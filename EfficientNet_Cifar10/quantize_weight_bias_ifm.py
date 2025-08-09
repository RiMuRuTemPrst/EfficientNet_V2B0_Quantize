# Tên file: dynamic_inference_and_detailed_log.py
import tensorflow as tf
import numpy as np
import os
import time

# ==============================================================================
# 0. CẤU HÌNH
# ==============================================================================
# Buộc chạy trên CPU để tránh lỗi driver và tăng tính ổn định khi gỡ lỗi
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# ⚙️ CẤU HÌNH ĐÁNH GIÁ ⚙️
LOG_TENSORS_FOR_FIRST_IMAGE_ONLY = True
#   Số lượng ảnh để đánh giá 
NUM_IMAGES_TO_EVALUATE = 1000
# ⚙️ CẤU HÌNH DYNAMIC FIXED-POINT ⚙️
QUANT_CONFIG = {
    # Tên lớp : {'ifm_m': In, 'w_m': Weight, 'ofm_m': Out}
    'stem_conv': {'ifm_m': 7, 'w_m': 7, 'ofm_m': 6},
    'block1_conv': {'ifm_m': 6, 'w_m': 7, 'ofm_m': 6},
    'block2a_expand': {'ifm_m': 6, 'w_m': 3, 'ofm_m': 4},
    'block2a_project': {'ifm_m': 4, 'w_m': 7, 'ofm_m': 3},
    'block2b_expand': {'ifm_m': 3, 'w_m': 5, 'ofm_m': 8},
    'block2b_project': {'ifm_m': 8, 'w_m': 6, 'ofm_m': 4},
    'block3a_expand': {'ifm_m': 4, 'w_m': 4, 'ofm_m': 3},
    'block3a_project': {'ifm_m': 3, 'w_m': 6, 'ofm_m': 5},
    'block3b_expand': {'ifm_m': 5, 'w_m': 5, 'ofm_m': 4},
    'block3b_project': {'ifm_m': 4, 'w_m': 6, 'ofm_m': 4},
    'dense_128': {'ifm_m': 4, 'w_m': 6, 'ofm_m': 5},
    'predictions': {'ifm_m': 5, 'w_m': 7, 'ofm_m': 5},
}
N_BITS = 8

# --- Các hàm tiện ích ---
def quantize_dequantize(tensor_float, n_bits, m_frac_bits):
    scale = 2.0 ** m_frac_bits
    min_val, max_val = -2**(n_bits - 1), 2**(n_bits - 1) - 1
    quantized_values = np.clip(np.round(tensor_float * scale), min_val, max_val)
    return quantized_values.astype(np.float32) / scale

def requantize_dequantize_ofm(ofm_tensor_float, ifm_m_bits, weight_m_bits, ofm_m_bits, n_bits_out):
    in_total_m_bits = ifm_m_bits + weight_m_bits
    scale_in, scale_out = 2.0 ** in_total_m_bits, 2.0 ** ofm_m_bits
    shift_amount = in_total_m_bits - ofm_m_bits
    min_val, max_val = -2**(n_bits_out - 1), 2**(n_bits_out - 1) - 1
    intermediate_int = np.round(ofm_tensor_float * scale_in)
    if shift_amount > 0:
        requantized_int = np.round(intermediate_int / (2.0 ** shift_amount))
    else:
        requantized_int = np.round(intermediate_int * (2.0 ** -shift_amount))
    requantized_int_clipped = np.clip(requantized_int, min_val, max_val)
    return requantized_int_clipped.astype(np.float32) / scale_out

def save_tensor_to_txt(base_dir, layer_name, tensor_type, tensor):
    if tensor is None: return
    if hasattr(tensor, 'numpy'): tensor = tensor.numpy()
    if tensor.ndim == 4:
        tensor = np.transpose(tensor, (0, 3, 1, 2))
    os.makedirs(base_dir, exist_ok=True)
    filepath = os.path.join(base_dir, f"{layer_name}_{tensor_type}.txt")
    np.savetxt(filepath, tensor.flatten(), fmt='%.8f')


# ==============================================================================
# 1. TẢI MÔ HÌNH VÀ DỮ LIỆU
# ==============================================================================
print("--- BƯỚC 1: TẢI MÔ HÌNH VÀ DỮ LIỆU ---")
LOG_DIR = 'inference_logs_detailed'
FLOAT_MODEL_FILE = 'cifar10_custom_hswish_trained.keras'

# Định nghĩa hàm custom để TensorFlow hiểu khi tải mô hình
def custom_hard_swish(x):
    return x * tf.nn.relu6(x + 3) / 6

try:
    float_model = tf.keras.models.load_model(
        FLOAT_MODEL_FILE,
        custom_objects={'custom_hard_swish': custom_hard_swish}
    )
    print(f"✅ Đã tải thành công mô hình float32: {FLOAT_MODEL_FILE}")
except Exception as e:
    print(f"\n--- ❌ LỖI KHI TẢI MÔ HÌNH ---")
    print(f"Chi tiết lỗi: {e}")
    print("Vui lòng đảm bảo file model đã được tạo ra từ script training và không bị lỗi.")
    print("--------------------------------\n")
    exit()

# Bỏ bước clone_model không cần thiết và lấy layer trực tiếp
print("Lấy thông tin các layer từ mô hình...")
layers_quant_config = {layer.name: layer for layer in float_model.layers}
layers_float = {layer.name: layer for layer in float_model.layers}

(_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()


# ==============================================================================
# 2. HÀM XỬ LÝ & VÒNG LẶP ĐÁNH GIÁ
# ==============================================================================
# THAY THẾ HÀM CŨ BẰNG HÀM NÀY
def process_and_log_layer_detailed(layer_name, x_quant_in, x_float_in, log_to_file=False):
    float_layer = layers_float[layer_name]
    quant_config_layer = layers_quant_config[layer_name]
    config = QUANT_CONFIG[layer_name]
    ifm_m, w_m, ofm_m = config['ifm_m'], config['w_m'], config['ofm_m']

    # Luồng Float32 để so sánh
    x_float_out = float_layer(x_float_in)

    # Luồng Quantized (Tái tạo thủ công)
    float_weights = float_layer.get_weights()
    quant_w_kernel = quantize_dequantize(float_weights[0], N_BITS, w_m)
    quant_w_bias = None
    if len(float_weights) > 1:
      quant_w_bias = quantize_dequantize(float_weights[1], N_BITS, ifm_m + w_m)

    quant_input_snapped = quantize_dequantize(x_quant_in, N_BITS, ifm_m)

    if isinstance(quant_config_layer, tf.keras.layers.Conv2D):
        strides, padding = quant_config_layer.strides, quant_config_layer.padding.upper()
        output_after_conv = tf.nn.conv2d(quant_input_snapped, quant_w_kernel, strides=strides, padding=padding)
    elif isinstance(quant_config_layer, tf.keras.layers.Dense):
        output_after_conv = tf.matmul(quant_input_snapped, quant_w_kernel)
    else:
        output_after_conv = quant_input_snapped

    output_after_bias = output_after_conv + quant_w_bias if quant_w_bias is not None else output_after_conv
    output_after_activation = quant_config_layer.activation(output_after_bias)

    is_final_dense = (layer_name == 'predictions')
    x_quant_out = output_after_activation if is_final_dense else \
        requantize_dequantize_ofm(output_after_activation, ifm_m, w_m, ofm_m, N_BITS)

    # Ghi log
    if log_to_file:
        quant_log_dir = os.path.join(LOG_DIR, 'quantized')
        # ✅ THÊM DÒNG NÀY: Tạo thư mục cho log float32
        float_log_dir = os.path.join(LOG_DIR, 'float32')

        print(f"\n--- Ghi log chi tiết cho ảnh đầu tiên, lớp {layer_name} ---")

        # --- Ghi log cho luồng Quantized (giữ nguyên) ---
        save_tensor_to_txt(quant_log_dir, layer_name, 'input', quant_input_snapped)
        save_tensor_to_txt(quant_log_dir, layer_name, 'weight', quant_w_kernel)
        if quant_w_bias is not None:
          save_tensor_to_txt(quant_log_dir, layer_name, 'bias', quant_w_bias)
        # Đổi tên file output để đồng nhất
        save_tensor_to_txt(quant_log_dir, layer_name, 'output', x_quant_out)

        # ✅ THÊM KHỐI LỆNH NÀY: Ghi log cho luồng Float32 ---
        save_tensor_to_txt(float_log_dir, layer_name, 'input', x_float_in)
        save_tensor_to_txt(float_log_dir, layer_name, 'weight', float_weights[0])
        if len(float_weights) > 1:
          save_tensor_to_txt(float_log_dir, layer_name, 'bias', float_weights[1])
        save_tensor_to_txt(float_log_dir, layer_name, 'output', x_float_out)

    return x_quant_out, x_float_out

# Vòng lặp đánh giá chính
print(f"\n--- BƯỚC 2: BẮT ĐẦU ĐÁNH GIÁ TRÊN {NUM_IMAGES_TO_EVALUATE} ẢNH ---")
correct_float, correct_quant = 0, 0
start_time = time.time()

for i in range(NUM_IMAGES_TO_EVALUATE):
    print(f"\rĐang xử lý ảnh {i + 1}/{NUM_IMAGES_TO_EVALUATE}...", end="")
    sample_image, true_label_index = x_test[i:i+1], y_test[i][0]
    x_input = tf.image.resize(sample_image, (224, 224))
    should_log = (i == 0 and LOG_TENSORS_FOR_FIRST_IMAGE_ONLY)

    # --- Bắt đầu luồng inference tuần tự cho 1 ảnh ---
    x_quant, x_float = x_input, x_input
    x_quant, x_float = layers_quant_config['rescaling'](x_quant), layers_float['rescaling'](x_float)
    x_quant, x_float = process_and_log_layer_detailed('stem_conv', x_quant, x_float, should_log)
    x_quant, x_float = process_and_log_layer_detailed('block1_conv', x_quant, x_float, should_log)
    x_quant, x_float = process_and_log_layer_detailed('block2a_expand', x_quant, x_float, should_log)
    x_quant, x_float = process_and_log_layer_detailed('block2a_project', x_quant, x_float, should_log)
    x_skip_2b_quant, x_skip_2b_float = x_quant, x_float
    x_quant, x_float = process_and_log_layer_detailed('block2b_expand', x_quant, x_float, should_log)
    x_quant, x_float = process_and_log_layer_detailed('block2b_project', x_quant, x_float, should_log)

    x_quant = layers_quant_config['block2b_drop'](x_quant, training=False)
    x_float = layers_float['block2b_drop'](x_float, training=False)
    
    x_quant, x_float = layers_quant_config['block2b_add']([x_quant, x_skip_2b_quant]), layers_float['block2b_add']([x_float, x_skip_2b_float])
    x_quant, x_float = process_and_log_layer_detailed('block3a_expand', x_quant, x_float, should_log)
    x_quant, x_float = process_and_log_layer_detailed('block3a_project', x_quant, x_float, should_log)
    x_skip_3b_quant, x_skip_3b_float = x_quant, x_float
    x_quant, x_float = process_and_log_layer_detailed('block3b_expand', x_quant, x_float, should_log)
    x_quant, x_float = process_and_log_layer_detailed('block3b_project', x_quant, x_float, should_log)

    x_quant = layers_quant_config['block3b_drop'](x_quant, training=False)
    x_float = layers_float['block3b_drop'](x_float, training=False)

    x_quant, x_float = layers_quant_config['block3b_add']([x_quant, x_skip_3b_quant]), layers_float['block3b_add']([x_float, x_skip_3b_float])
    x_quant, x_float = layers_quant_config['global_avg_pool'](x_quant), layers_float['global_avg_pool'](x_float)
    x_quant, x_float = process_and_log_layer_detailed('dense_128', x_quant, x_float, should_log)

    x_quant = layers_quant_config['dropout_final'](x_quant, training=False)
    x_float = layers_float['dropout_final'](x_float, training=False)

    final_logits_quant, final_logits_float = process_and_log_layer_detailed('predictions', x_quant, x_float, should_log)

    if np.argmax(tf.nn.softmax(final_logits_float)) == true_label_index: correct_float += 1
    if np.argmax(tf.nn.softmax(final_logits_quant)) == true_label_index: correct_quant += 1

end_time = time.time()
print(f"\nĐánh giá hoàn tất. Tổng thời gian: {end_time - start_time:.2f} giây.")


# ==============================================================================
# 3. HIỂN THỊ KẾT QUẢ ĐỘ CHÍNH XÁC
# ==============================================================================
print("\n--- BƯỚC 3: KẾT QUẢ ĐỘ CHÍNH XÁC ---")
acc_float = (correct_float / NUM_IMAGES_TO_EVALUATE) * 100
acc_quant = (correct_quant / NUM_IMAGES_TO_EVALUATE) * 100
acc_drop = acc_float - acc_quant

print(f"Số lượng ảnh đánh giá: {NUM_IMAGES_TO_EVALUATE}")
print("-" * 50)
print(f"✅ Độ chính xác mô hình Float32: {acc_float:.2f}% ({correct_float}/{NUM_IMAGES_TO_EVALUATE})")
print(f"⚙️ Độ chính xác mô hình Quantized: {acc_quant:.2f}% ({correct_quant}/{NUM_IMAGES_TO_EVALUATE})")
print("-" * 50)
print(f"📉 Độ sụt giảm chính xác: {acc_drop:.2f}%")