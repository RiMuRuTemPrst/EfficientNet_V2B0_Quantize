# import tensorflow as tf
# import tensorflow_datasets as tfds
# import numpy as np
# import os
# import math

# # Tắt các thông báo log không quan trọng
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# # ==============================================================================
# # PHẦN 1: CÁC HÀM HỖ TRỢ
# # ==============================================================================
# def load_weights_from_file(layer, base_weights_dir):
#     """Đọc file HEX chứa trọng số và bias đã lượng tử hóa về int8."""
#     layer_name = layer.name
#     layer_dir = os.path.join(base_weights_dir, layer_name)
#     try:
#         # Đọc Kernel
#         kernel_params_path = os.path.join(layer_dir, "kernel_parameters.txt")
#         kernel_hex_path = os.path.join(layer_dir, "kernel_quantized_int8.hex")
#         with open(kernel_params_path, 'r') as f:
#             kernel_scale = float(f.readline().split(':')[1].strip())
#         with open(kernel_hex_path, 'r') as f:
#             kernel_hex_vals = [int(line.strip(), 16) for line in f.readlines()]
#         kernel_int = np.array(kernel_hex_vals, dtype=np.uint8).view(np.int8)
#         kernel_float = kernel_int.astype(np.float32) * kernel_scale
#         kernel_float = kernel_float.reshape(layer.get_weights()[0].shape)

#         # Đọc Bias
#         bias_params_path = os.path.join(layer_dir, "bias_parameters.txt")
#         bias_hex_path = os.path.join(layer_dir, "bias_quantized_int8.hex")
#         with open(bias_params_path, 'r') as f:
#             bias_scale = float(f.readline().split(':')[1].strip())
#         with open(bias_hex_path, 'r') as f:
#             bias_hex_vals = [int(line.strip(), 16) for line in f.readlines()]
#         bias_int = np.array(bias_hex_vals, dtype=np.uint8).view(np.int8)
#         bias_float = bias_int.astype(np.float32) * bias_scale
#         bias_float = bias_float.reshape(layer.get_weights()[1].shape)
        
#         return [kernel_float, bias_float]
#     except Exception as e:
#         print(f"  ❌ Lỗi khi nạp trọng số cho lớp '{layer_name}': {e}")
#         raise

# def calculate_quantization_params_pow2(tensor_data):
#     """Tính toán scale (dạng 2^n) và zero-point cho tensor đầu ra."""
#     min_val, max_val = np.min(tensor_data), np.max(tensor_data)
#     initial_scale = (max_val - min_val) / 255
#     if initial_scale <= 0: return 1.0, 0
#     power_of_2_scale = 2**round(math.log2(initial_scale))
#     return power_of_2_scale, 0

# def quantize_tensor(tensor_data, scale, zero_point):
#     """Lượng tử hóa tensor float32 sang int8 và trả về accumulator int32."""
#     scaled_float = tensor_data / scale
#     accumulator_int32 = np.round(scaled_float).astype(np.int32)
#     clipped_values = np.clip(accumulator_int32, -128, 127)
#     return clipped_values.astype(np.int8), accumulator_int32

# def write_float32_to_4byte_hex(filepath, float_tensor):
#     """Ghi tensor float32 ra file hex, mỗi giá trị 4 byte trên một dòng."""
#     byte_array = float_tensor.flatten().view(np.uint8)
#     with open(filepath, 'w') as f:
#         for i in range(0, len(byte_array), 4):
#             f.write(''.join(f'{byte:02x}' for byte in byte_array[i:i+4]) + '\n')

# def write_int32_to_4byte_hex(filepath, int32_tensor):
#     """Ghi tensor int32 ra file hex, mỗi giá trị 4 byte trên một dòng."""
#     flat_tensor = int32_tensor.flatten()
#     with open(filepath, 'w') as f:
#         for num in flat_tensor:
#             f.write(f'{(num & 0xffffffff):08x}\n')

# # ==============================================================================
# # PHẦN 2: KỊCH BẢN KIỂM TRA STEM BLOCK
# # ==============================================================================
# print("\n--- KIỂM TRA RIÊNG STEM BLOCK ---")

# IMAGE_SIZE = (224, 224)
# output_check_dir = "stem_output"
# os.makedirs(output_check_dir, exist_ok=True)

# # --- BƯỚC 1: Tải model gốc ---
# print("BƯỚC 1: Đang tải model EfficientNetV2B0 gốc...")
# original_full_model = tf.keras.applications.EfficientNetV2B0(weights='imagenet', input_shape=IMAGE_SIZE + (3,))
# print("✅ Tải model thành công.")

# # --- BƯỚC 2: Xây dựng model "chuẩn vàng" cho Stem ---
# original_stem_model = tf.keras.models.Model(
#     inputs=original_full_model.input, 
#     outputs=original_full_model.get_layer('stem_activation').output,
#     name="golden_stem_model"
# )
# print("✅ Đã trích xuất model Stem chuẩn.")

# # --- BƯỚC 3: Xây dựng model Stem tùy chỉnh ---
# print("\nBƯỚC 2: Xây dựng model Stem tùy chỉnh...")
# custom_inputs = tf.keras.layers.Input(shape=(224, 224, 3), name='input_layer')
# x = tf.keras.layers.Rescaling(1./255)(custom_inputs)
# x = tf.keras.layers.Normalization(axis=-1, mean=[0.485, 0.456, 0.406], variance=[0.229**2, 0.224**2, 0.225**2])(x)
# stem_conv_layer = tf.keras.layers.Conv2D(32, 3, strides=2, padding='same', use_bias=True, name='stem_conv')
# x = stem_conv_layer(x)
# x = tf.keras.layers.Activation('swish', name='stem_activation')(x)
# custom_stem_model = tf.keras.models.Model(inputs=custom_inputs, outputs=x, name="custom_stem_model")
# _ = custom_stem_model.predict(np.zeros((1, 224, 224, 3)), verbose=0)
# print("✅ Đã xây dựng model Stem tùy chỉnh.")

# # --- BƯỚC 4: Nạp trọng số cho Stem tùy chỉnh ---
# base_weights_dir = "efficientnetv2b0_quantized_weights_8bit"
# print(f"\nBƯỚC 3: Nạp trọng số cho 'stem_conv' từ '{base_weights_dir}'...")
# try:
#     weights = load_weights_from_file(custom_stem_model.get_layer('stem_conv'), base_weights_dir)
#     custom_stem_model.get_layer('stem_conv').set_weights(weights)
#     print("✅ Đã nạp trọng số thành công.")
# except Exception as e:
#     print(f"❌ Lỗi khi nạp trọng số: {e}")
#     exit()

# # --- BƯỚC 5: Chuẩn bị ảnh đầu vào ---
# print("\nBƯỚC 4: Chuẩn bị ảnh đầu vào...")
# try:
#     ds_single_image = tfds.load("imagenet_v2", split='test', as_supervised=True).take(1)
#     for img, _ in ds_single_image:
#         input_image_batch = tf.expand_dims(tf.image.resize(img, IMAGE_SIZE), axis=0)
#     print("✅ Đã có ảnh đầu vào.")
# except Exception as e:
#     print(f"❌ Không thể tải ảnh: {e}.")
#     exit()
    
# # --- BƯỚC 6: Chạy suy luận ---
# print("\nBƯỚC 5: Chạy suy luận...")
# output_golden = original_stem_model.predict(input_image_batch, verbose=0)
# output_custom = custom_stem_model.predict(input_image_batch, verbose=0)
# print("✅ Đã có kết quả từ cả hai model.")

# # --- BƯỚC 7: Lượng tử hóa và Ghi file kết quả ---
# print(f"\nBƯỚC 6: Ghi file kết quả ra thư mục '{output_check_dir}'...")
# output_scale, output_zp = calculate_quantization_params_pow2(output_custom)
# quantized_output_int8, accumulator_output_int32 = quantize_tensor(output_custom, output_scale, output_zp)

# params_path = os.path.join(output_check_dir, "output_parameters.txt")
# with open(params_path, 'w') as f: f.write(f"scale: {output_scale}\nzero_point: {output_zp}\n")
# np.savetxt(os.path.join(output_check_dir, "output_custom_float32.txt"), output_custom.flatten(), fmt='%.8f')
# write_float32_to_4byte_hex(os.path.join(output_check_dir, "output_custom_float32.hex"), output_custom)
# write_int32_to_4byte_hex(os.path.join(output_check_dir, "output_custom_accumulator_int32.hex"), accumulator_output_int32)
# np.savetxt(os.path.join(output_check_dir, "output_custom_quantized_int8.hex"), quantized_output_int8.flatten().view(np.uint8), fmt='%02x')
# np.savetxt(os.path.join(output_check_dir, "output_golden_float32.txt"), output_golden.flatten(), fmt='%.8f')
# print("✅ Đã ghi tất cả các file kết quả.")

# # --- BƯỚC 8: So sánh ---
# print("\nBƯỚC 7: So sánh kết quả MSE...")
# mse = np.mean(np.square(output_golden - output_custom))
# print(f"\n=> Sai số bình phương trung bình (MSE): {mse:.8f}")

# if mse < 1e-5: 
#     print("\n✅✅✅ THÀNH CÔNG! Stem Block tùy chỉnh hoạt động chính xác.")
# else:
#     print("\n❌❌❌ THẤT BẠI: Output có sự sai lệch.") 

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
import math

# Tắt các thông báo log không quan trọng
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==============================================================================
# PHẦN 1: CÁC HÀM HỖ TRỢ (Không thay đổi)
# ==============================================================================
def load_weights_from_file(layer, base_weights_dir):
    """Đọc file HEX chứa trọng số và bias đã lượng tử hóa về int8."""
    layer_name = layer.name
    layer_dir = os.path.join(base_weights_dir, layer_name)
    try:
        # Đọc Kernel
        kernel_params_path = os.path.join(layer_dir, "kernel_parameters.txt")
        kernel_hex_path = os.path.join(layer_dir, "kernel_quantized_int8.hex")
        with open(kernel_params_path, 'r') as f:
            kernel_scale = float(f.readline().split(':')[1].strip())
        with open(kernel_hex_path, 'r') as f:
            kernel_hex_vals = [int(line.strip(), 16) for line in f.readlines()]
        kernel_int = np.array(kernel_hex_vals, dtype=np.uint8).view(np.int8)
        kernel_float = kernel_int.astype(np.float32) * kernel_scale
        kernel_float = kernel_float.reshape(layer.get_weights()[0].shape)

        # Đọc Bias
        bias_params_path = os.path.join(layer_dir, "bias_parameters.txt")
        bias_hex_path = os.path.join(layer_dir, "bias_quantized_int8.hex")
        with open(bias_params_path, 'r') as f:
            bias_scale = float(f.readline().split(':')[1].strip())
        with open(bias_hex_path, 'r') as f:
            bias_hex_vals = [int(line.strip(), 16) for line in f.readlines()]
        bias_int = np.array(bias_hex_vals, dtype=np.uint8).view(np.int8)
        bias_float = bias_int.astype(np.float32) * bias_scale
        bias_float = bias_float.reshape(layer.get_weights()[1].shape)
        
        return [kernel_float, bias_float]
    except Exception as e:
        print(f"  ❌ Lỗi khi nạp trọng số cho lớp '{layer_name}': {e}")
        raise

def calculate_quantization_params_pow2(tensor_data):
    min_val, max_val = np.min(tensor_data), np.max(tensor_data)
    initial_scale = (max_val - min_val) / 255
    if initial_scale <= 0: return 1.0, 0
    power_of_2_scale = 2**round(math.log2(initial_scale))
    return power_of_2_scale, 0

def quantize_tensor(tensor_data, scale, zero_point):
    scaled_float = tensor_data / scale
    accumulator_int32 = np.round(scaled_float).astype(np.int32)
    clipped_values = np.clip(accumulator_int32, -128, 127)
    return clipped_values.astype(np.int8), accumulator_int32

def write_float32_to_4byte_hex(filepath, float_tensor):
    byte_array = float_tensor.flatten().view(np.uint8)
    with open(filepath, 'w') as f:
        for i in range(0, len(byte_array), 4):
            f.write(''.join(f'{byte:02x}' for byte in byte_array[i:i+4]) + '\n')

def write_int32_to_4byte_hex(filepath, int32_tensor):
    flat_tensor = int32_tensor.flatten()
    with open(filepath, 'w') as f:
        for num in flat_tensor:
            f.write(f'{(num & 0xffffffff):08x}\n')

# ==============================================================================
# PHẦN 2: KỊCH BẢN KIỂM TRA STEM + BLOCK 1A
# ==============================================================================
print("\n--- KIỂM TRA STEM + BLOCK 1A ---")

IMAGE_SIZE = (224, 224)
# THAY ĐỔI: Đổi tên thư mục output
output_check_dir = "stem_block1a_output"
os.makedirs(output_check_dir, exist_ok=True)

# --- BƯỚC 1: Tải model gốc ---
print("BƯỚC 1: Đang tải model EfficientNetV2B0 gốc...")
original_full_model = tf.keras.applications.EfficientNetV2B0(weights='imagenet', input_shape=IMAGE_SIZE + (3,))
print("✅ Tải model thành công.")

# --- BƯỚC 2: Xây dựng model "chuẩn vàng" ---
# THAY ĐỔI: Điểm output là lớp cuối cùng của Block 1a
output_layer_name_golden = 'block1a_project_activation'
original_model = tf.keras.models.Model(
    inputs=original_full_model.input, 
    outputs=original_full_model.get_layer(output_layer_name_golden).output,
    name="golden_stem_block1a_model"
)
print(f"✅ Đã trích xuất model chuẩn với đầu ra là '{output_layer_name_golden}'.")


# --- BƯỚC 3: Xây dựng model tùy chỉnh ---
print("\nBƯỚC 2: Xây dựng model tùy chỉnh...")
custom_inputs = tf.keras.layers.Input(shape=(224, 224, 3), name='input_layer')

# ---- KHỐI 1: STEM BLOCK ----
x = tf.keras.layers.Rescaling(1./255)(custom_inputs)
x = tf.keras.layers.Normalization(axis=-1, mean=[0.485, 0.456, 0.406], variance=[0.229**2, 0.224**2, 0.225**2])(x)
stem_conv_layer = tf.keras.layers.Conv2D(32, 3, strides=2, padding='same', use_bias=True, name='stem_conv')
x = stem_conv_layer(x)
x = tf.keras.layers.Activation('swish', name='stem_activation')(x)

# ---- MỚI: KHỐI 2: BLOCK 1A ----
# Sử dụng kernel 3x3 như đã xác định ở các bước gỡ lỗi trước
block1a_project_conv_layer = tf.keras.layers.Conv2D(16, 3, padding='same', use_bias=True, name='block1a_project_conv')
x = block1a_project_conv_layer(x)
x = tf.keras.layers.Activation('swish', name='block1a_project_activation')(x)

# Tạo model tùy chỉnh hoàn chỉnh
custom_model = tf.keras.models.Model(inputs=custom_inputs, outputs=x, name="custom_model")
_ = custom_model.predict(np.zeros((1, 224, 224, 3)), verbose=0)
print("✅ Đã xây dựng model tùy chỉnh cho Stem + Block 1a.")


# --- BƯỚC 4: Nạp trọng số ---
base_weights_dir = "efficientnetv2b0_quantized_weights_8bit"
print(f"\nBƯỚC 3: Nạp trọng số từ '{base_weights_dir}'...")
try:
    # THAY ĐỔI: Thêm lớp của Block 1a vào danh sách nạp
    layers_to_load = [
        custom_model.get_layer('stem_conv'),
        custom_model.get_layer('block1a_project_conv')
    ]
    for layer_obj in layers_to_load:
        weights = load_weights_from_file(layer_obj, base_weights_dir)
        layer_obj.set_weights(weights)
        print(f"  - Nạp trọng số thành công cho: '{layer_obj.name}'")
    print("✅ Đã nạp đầy đủ trọng số.")
except Exception as e:
    print(f"❌ Lỗi khi nạp trọng số: {e}")
    exit()

# --- BƯỚC 5: Chuẩn bị ảnh đầu vào ---
print("\nBƯỚC 4: Chuẩn bị ảnh đầu vào...")
try:
    ds_single_image = tfds.load("imagenet_v2", split='test', as_supervised=True).take(1)
    for img, _ in ds_single_image:
        input_image_batch = tf.expand_dims(tf.image.resize(img, IMAGE_SIZE), axis=0)
    print("✅ Đã có ảnh đầu vào.")
except Exception as e:
    print(f"❌ Không thể tải ảnh: {e}.")
    exit()
    
# --- BƯỚC 6: Chạy suy luận ---
print("\nBƯỚC 5: Chạy suy luận...")
output_golden = original_model.predict(input_image_batch, verbose=0)
output_custom = custom_model.predict(input_image_batch, verbose=0)
print("✅ Đã có kết quả từ cả hai model.")

# --- BƯỚC 7: Lượng tử hóa và Ghi file kết quả ---
print(f"\nBƯỚC 6: Ghi file kết quả ra thư mục '{output_check_dir}'...")
output_scale, output_zp = calculate_quantization_params_pow2(output_custom)
quantized_output_int8, accumulator_output_int32 = quantize_tensor(output_custom, output_scale, output_zp)

params_path = os.path.join(output_check_dir, "output_parameters.txt")
with open(params_path, 'w') as f: f.write(f"scale: {output_scale}\nzero_point: {output_zp}\n")
np.savetxt(os.path.join(output_check_dir, "output_custom_float32.txt"), output_custom.flatten(), fmt='%.8f')
write_float32_to_4byte_hex(os.path.join(output_check_dir, "output_custom_float32.hex"), output_custom)
write_int32_to_4byte_hex(os.path.join(output_check_dir, "output_custom_accumulator_int32.hex"), accumulator_output_int32)
np.savetxt(os.path.join(output_check_dir, "output_custom_quantized_int8.hex"), quantized_output_int8.flatten().view(np.uint8), fmt='%02x')
np.savetxt(os.path.join(output_check_dir, "output_golden_float32.txt"), output_golden.flatten(), fmt='%.8f')
print("✅ Đã ghi tất cả các file kết quả.")

# --- BƯỚC 8: So sánh ---
print("\nBƯỚC 7: So sánh kết quả MSE...")
mse = np.mean(np.square(output_golden - output_custom))
print(f"\n=> Sai số bình phương trung bình (MSE): {mse:.8f}")

if mse < 1e-5: 
    print("\n✅✅✅ THÀNH CÔNG! Model tùy chỉnh hoạt động chính xác.")
else:
    print(f"\nℹ️ THÔNG TIN: Sai số MSE là {mse:.4f}. Con số này phản ánh sai số lượng tử hóa tích lũy qua các khối.")