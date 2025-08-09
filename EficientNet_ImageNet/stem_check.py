import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
import math
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==========================================================
# CÁC HÀM HỖ TRỢ
# ==========================================================
def read_hex_file_to_int8(filepath, shape):
    with open(filepath, 'r') as f:
        hex_vals = [int(line.strip(), 16) for line in f.readlines()]
    int8_array = np.array(hex_vals, dtype=np.uint8).view(np.int8)
    return int8_array.reshape(shape)

def read_params_file(filepath):
    params = {}
    with open(filepath, 'r') as f:
        for line in f:
            if ':' in line: key, value = line.split(':')
            else: continue
            params[key.strip()] = float(value.strip())
    return params.get('scale', 1.0), int(params.get('zero_point', 0))

def load_quantized_weights(layer_name, base_weights_dir):
    layer_dir = os.path.join(base_weights_dir, layer_name)
    kernel_scale, _ = read_params_file(os.path.join(layer_dir, "kernel_parameters.txt"))
    kernel_int8 = read_hex_file_to_int8(os.path.join(layer_dir, "kernel_quantized_int8.hex"), (3, 3, 3, 32))
    bias_scale, _ = read_params_file(os.path.join(layer_dir, "bias_parameters.txt"))
    bias_int8 = read_hex_file_to_int8(os.path.join(layer_dir, "bias_quantized_int8.hex"), (32,))
    return kernel_int8, kernel_scale, bias_int8, bias_scale

def manual_conv2d_int8(input_int8, kernel_int8, bias_int32, stride, padding):
    input_int8 = input_int8[0]
    h_in, w_in, c_in = input_int8.shape
    k_h, k_w, _, c_out = kernel_int8.shape
    if padding == 'same':
        h_out, w_out = int(math.ceil(h_in/stride)), int(math.ceil(w_in/stride))
        pad_h_total, pad_w_total = (h_out-1)*stride + k_h - h_in, (w_out-1)*stride + k_w - w_in
        pad_top, pad_left = pad_h_total//2, pad_w_total//2
    else: h_out, w_out, pad_top, pad_left = 0,0,0,0
    accumulator = np.zeros((h_out, w_out, c_out), dtype=np.int32)
    for c in range(c_out):
        for y in range(h_out):
            for x in range(w_out):
                acc = bias_int32[c]
                for ky in range(k_h):
                    for kx in range(k_w):
                        for kc in range(c_in):
                            iy, ix = y * stride + ky - pad_top, x * stride + kx - pad_left
                            if (iy >= 0 and iy < h_in and ix >= 0 and ix < w_in):
                                acc += int(input_int8[iy, ix, kc]) * int(kernel_int8[ky, kx, kc, c])
                accumulator[y, x, c] = acc
    return np.expand_dims(accumulator, axis=0)

def manual_swish(data):
    return data / (1.0 + np.exp(-data))

# ==========================================================
# CHƯƠNG TRÌNH CHÍNH
# ==========================================================
if __name__ == '__main__':
    IMAGE_SIZE = (224, 224)
    INPUT_HEX_PATH = 'input_files/input_quantized_int8.hex'
    INPUT_PARAMS_PATH = 'input_files/input_quantization_params.txt'
    OUTPUT_DIR = "stem_final_fixed_q_output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- BƯỚC 1: Đọc input và tạo tham chiếu ---
    print(f"--- Bước 1: Đọc input và tạo tham chiếu ---")
    try:
        input_int8 = read_hex_file_to_int8(INPUT_HEX_PATH, (1, *IMAGE_SIZE, 3))
        input_scale, input_zp = read_params_file(INPUT_PARAMS_PATH)
        input_dequantized_fp32 = (input_int8.astype(np.float32) - input_zp) * input_scale
        original_full_model = tf.keras.applications.EfficientNetV2B0(weights='imagenet', input_shape=IMAGE_SIZE + (3,))
        
        # Tạo model tham chiếu
        x_input_norm = original_full_model.get_layer('normalization').output
        x = original_full_model.get_layer('stem_conv')(x_input_norm)
        x = original_full_model.get_layer('stem_bn')(x)
        # MỚI: Giữ lại tensor trước activation
        output_before_activation = x 
        x = original_full_model.get_layer('stem_activation')(x)
        
        # Model 1: Lấy output cuối cùng (sau activation)
        golden_model_final = tf.keras.models.Model(inputs=x_input_norm, outputs=x)
        # MỚI: Model 2: Lấy output trung gian (trước activation)
        golden_model_before_activation = tf.keras.models.Model(inputs=x_input_norm, outputs=output_before_activation)
        
        # Chuẩn bị input cho các model tham chiếu con
        norm_model = tf.keras.models.Model(inputs=original_full_model.input, outputs=original_full_model.get_layer('normalization').output)
        norm_input_for_golden = norm_model.predict(input_dequantized_fp32, verbose=0)
        
        # Chạy để lấy cả hai kết quả
        golden_output = golden_model_final.predict(norm_input_for_golden, verbose=0)
        golden_output_before_act = golden_model_before_activation.predict(norm_input_for_golden, verbose=0)

        print("✅ Đã tạo output tham chiếu.")
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        exit()

    # --- BƯỚC 2: Mô phỏng tính toán int8 ---
    print("\n--- Bước 2: Mô phỏng tính toán int8 thủ công ---")
    base_weights_dir = "efficientnetv2b0_quantized_weights_8bit" 
    stem_kernel_int8, stem_kernel_scale, stem_bias_int8, stem_bias_scale = load_quantized_weights('stem_conv', base_weights_dir)
    accumulator_scale = input_scale * stem_kernel_scale
    bias_fp32 = stem_bias_int8.astype(np.float32) * stem_bias_scale
    bias_int32 = np.round(bias_fp32 / accumulator_scale).astype(np.int32)
    input_int8_symmetric = input_int8.astype(np.int32) - input_zp
    
    accumulator_output_int32 = manual_conv2d_int8(
        input_int8_symmetric.astype(np.int8), stem_kernel_int8, bias_int32, stride=2, padding='same'
    )
    manual_dequantized_output = accumulator_output_int32.astype(np.float32) * accumulator_scale
    manual_swish_output = manual_swish(manual_dequantized_output)
    print("✅ Tính toán thủ công hoàn tất.")

    # --- BƯỚC 3: Ép kết quả cuối cùng về định dạng Q cố định ---
    print("\n--- Bước 3: Ép output về định dạng Q cố định ---")
    TARGET_SCALE = 0.03125 
    q_min, q_max = -128, 127
    min_val, max_val = np.min(manual_swish_output), np.max(manual_swish_output)
    target_zp = round(q_min - min_val / TARGET_SCALE)
    target_zp = int(np.clip(target_zp, q_min, q_max))
    final_quantized_int8 = np.round(manual_swish_output / TARGET_SCALE) + target_zp
    final_quantized_int8 = np.clip(final_quantized_int8, q_min, q_max).astype(np.int8)
    manual_final_output = (final_quantized_int8.astype(np.float32) - target_zp) * TARGET_SCALE
    print(f"✅ Đã ép output về định dạng với scale={TARGET_SCALE}.")

    # --- BƯỚC 4: Ghi kết quả ra file ---
    print(f"\n--- Bước 4: Ghi kết quả ra thư mục '{OUTPUT_DIR}' ---")
    
    np.savetxt(os.path.join(OUTPUT_DIR, "accumulator_output_int32.txt"), accumulator_output_int32.flatten(), fmt='%d')
    print("  - Đã ghi: accumulator_output_int32.txt")
    np.savetxt(os.path.join(OUTPUT_DIR, "intermediate_dequantized_float32.txt"), manual_dequantized_output.flatten(), fmt='%.8f')
    print("  - Đã ghi: intermediate_dequantized_float32.txt")
    np.savetxt(os.path.join(OUTPUT_DIR, "custom_output_dequantized.txt"), manual_final_output.flatten(), fmt='%.8f')
    print("  - Đã ghi: custom_output_dequantized.txt")
    np.savetxt(os.path.join(OUTPUT_DIR, "custom_output_quantized_int8.hex"), final_quantized_int8.flatten().view(np.uint8), fmt='%02x')
    print("  - Đã ghi: custom_output_quantized_int8.hex")
    
    # MỚI: Ghi file output của model chuẩn trước activation
    np.savetxt(os.path.join(OUTPUT_DIR, "golden_output_before_activation.txt"), golden_output_before_act.flatten(), fmt='%.8f')
    print("  - Đã ghi: golden_output_before_activation.txt")
    
    np.savetxt(os.path.join(OUTPUT_DIR, "golden_output_final.txt"), golden_output.flatten(), fmt='%.8f')
    print("  - Đã ghi: golden_output_final.txt")
    
    with open(os.path.join(OUTPUT_DIR, "custom_output_params.txt"), 'w') as f:
        f.write(f"scale: {TARGET_SCALE}\n")
        f.write(f"zero_point: {target_zp}\n")
    print("  - Đã ghi: custom_output_params.txt")
    print("✅ Ghi file hoàn tất.")

    # --- BƯỚC 5: So sánh ---
    print("\n--- Bước 5: So sánh ---")
    mse = np.mean(np.square(golden_output - manual_final_output))
    print(f"\n=> Sai số bình phương trung bình (MSE): {mse:.8f}")

# import tensorflow as tf
# import numpy as np
# import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# # ==========================================================
# # CÁC HÀM HỖ TRỢ
# # ==========================================================
# def read_hex_file_to_int8(filepath, shape):
#     with open(filepath, 'r') as f:
#         hex_vals = [int(line.strip(), 16) for line in f.readlines()]
#     int8_array = np.array(hex_vals, dtype=np.uint8).view(np.int8)
#     return int8_array.reshape(shape)

# def read_params_file(filepath):
#     params = {}
#     with open(filepath, 'r') as f:
#         for line in f:
#             if ':' in line: key, value = line.split(':')
#             else: continue
#             params[key.strip()] = float(value.strip())
#     return params.get('scale', 1.0), int(params.get('zero_point', 0))

# def manual_swish(data):
#     return data / (1.0 + np.exp(-data))

# # MỚI: Hàm in ra chi tiết phép tính cho 1 pixel
# def print_detailed_conv_calculation(input_patch, kernel_slice, bias, data_type='float32'):
#     """In ra chi tiết 27 phép nhân và cộng cho 1 pixel output."""
#     acc = 0
#     print("  --- Bắt đầu tích chập chi tiết ---")
#     for ky in range(3):
#         for kx in range(3):
#             for kc in range(3):
#                 input_val = input_patch[ky, kx, kc]
#                 kernel_val = kernel_slice[ky, kx, kc]
#                 product = int(input_val) * int(kernel_val) if data_type == 'int8' else input_val * kernel_val
                
#                 if data_type == 'int8':
#                     print(f"    - Input[{kc}]({ky},{kx}) * Kernel({ky},{kx}) = {int(input_val):4d} * {int(kernel_val):4d} = {product}")
#                 else:
#                     print(f"    - Input[{kc}]({ky},{kx}) * Kernel({ky},{kx}) = {input_val: 10.6f} * {kernel_val: 10.6f} = {product:.6f}")

#                 acc += product
    
#     print("  ---------------------------------")
#     print(f"  - Tổng các phép nhân (Accumulator trước bias): {acc}")
#     print(f"  - Cộng với Bias ({bias:.6f})")
#     final_acc = acc + bias
#     print(f"  - KẾT QUẢ CUỐI CÙNG (Accumulator sau bias): {final_acc}")
#     return final_acc

# # ==========================================================
# # CHƯƠNG TRÌNH CHÍNH
# # ==========================================================
# if __name__ == '__main__':
#     # --- Chuẩn bị dữ liệu ---
#     print("--- Chuẩn bị dữ liệu và các tham số ---")
#     IMAGE_SIZE = (224, 224)
#     INPUT_HEX_PATH = 'input_files/input_quantized_int8.hex'
#     INPUT_PARAMS_PATH = 'input_files/input_quantization_params.txt'
#     WEIGHTS_DIR = "efficientnetv2b0_quantized_weights_8bit" 
    
#     input_int8, (input_scale, input_zp) = read_hex_file_to_int8(INPUT_HEX_PATH, (1, *IMAGE_SIZE, 3)), read_params_file(INPUT_PARAMS_PATH)
#     original_full_model = tf.keras.applications.EfficientNetV2B0(weights='imagenet', input_shape=IMAGE_SIZE + (3,))
    
#     stem_conv_orig = original_full_model.get_layer('stem_conv')
#     stem_bn_orig = original_full_model.get_layer('stem_bn')
#     original_conv_weights = stem_conv_orig.get_weights()
#     kernel_fp32_orig = original_conv_weights[0]
#     bias_fp32_orig = np.zeros(kernel_fp32_orig.shape[-1], dtype=np.float32)
#     gamma, beta, moving_mean, moving_variance = stem_bn_orig.get_weights()
#     epsilon = stem_bn_orig.epsilon

#     with open(os.path.join(WEIGHTS_DIR, 'stem_conv', "kernel_parameters.txt"), 'r') as f:
#         kernel_scale = float(f.readline().split(':')[1].strip())
#     kernel_int8 = read_hex_file_to_int8(os.path.join(WEIGHTS_DIR, 'stem_conv', "kernel_quantized_int8.hex"), kernel_fp32_orig.shape)
    
#     input_patch_int8 = input_int8[0, 0:3, 0:3, :]
#     input_patch_fp32 = (input_patch_int8.astype(np.float32) - input_zp) * input_scale
    
#     print("✅ Chuẩn bị xong.\n")
#     print("="*80)
#     print("PHÂN TÍCH PHÉP TÍNH CHO PIXEL[0,0] CỦA KÊNH[0]")
#     print("="*80)

#     # --- Luồng 1: Tính toán theo model chuẩn (Float32) ---
#     print("\n--- 1. Luồng Model Chuẩn (Float32) ---")
#     acc_fp32 = print_detailed_conv_calculation(input_patch_fp32, kernel_fp32_orig[:,:,:,0], bias_fp32_orig[0], 'float32')
#     bn_out_fp32 = gamma[0] * ((acc_fp32 - moving_mean[0]) / np.sqrt(moving_variance[0] + epsilon)) + beta[0]
#     golden_value = manual_swish(bn_out_fp32)
#     print(f"\n  - GIÁ TRỊ CHUẨN (sau BN và Swish): {golden_value:.8f}")

#     # --- Luồng 2: Mô phỏng tính toán int8 ---
#     print("\n\n--- 2. Luồng Mô Phỏng (int8) ---")
#     accumulator_scale = input_scale * kernel_scale
#     bias_fp32_fused = beta[0] + (bias_fp32_orig[0] - moving_mean[0]) * (gamma[0] / np.sqrt(moving_variance[0] + epsilon))
#     bias_int32 = np.round(bias_fp32_fused / accumulator_scale)
    
#     input_patch_symmetric = input_patch_int8.astype(np.int32) - input_zp
    
#     acc_int32 = print_detailed_conv_calculation(input_patch_symmetric, kernel_int8[:,:,:,0], bias_int32, 'int8')
    
#     dequantized_value = float(acc_int32) * accumulator_scale
#     custom_value = manual_swish(dequantized_value)
#     print(f"\n  - GIÁ TRỊ TÍNH TOÁN (sau De-quantize và Swish): {custom_value:.8f}")

#     # --- KẾT LUẬN ---
#     print("\n" + "="*80)
#     print("SO SÁNH KẾT QUẢ CUỐI CÙNG")
#     print("="*80)
#     print(f"  - Giá trị chuẩn:                {golden_value:.8f}")
#     print(f"  - Giá trị mô phỏng:              {custom_value:.8f}")
#     print(f"  - Sai lệch tuyệt đối:           {abs(golden_value - custom_value):.8f}")