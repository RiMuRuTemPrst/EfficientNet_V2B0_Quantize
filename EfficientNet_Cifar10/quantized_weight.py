# Quantize Weight
import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==============================================================================
# 1. CÁC HÀM HỖ TRỢ LƯỢNG TỬ HÓA (ĐỊNH DẠNG Q3.5 - GIỮ NGUYÊN)
# ==============================================================================
def quantize_weights_q3_5(weights):
    """
    Lượng tử hóa trọng số float32 sang định dạng Q3.5 (trên int8).
    Scale cố định là 2^5 = 32.
    """
    scale = 32.0
    quantized_values = np.round(weights * scale)
    quantized_weights = np.clip(quantized_values, -128, 127).astype(np.int8)
    return quantized_weights, scale

def dequantize_weights_q_format(quantized_weights, scale):
    """
    Giải lượng tử hóa từ định dạng Q-format về float32.
    """
    return quantized_weights.astype(np.float32) / scale

# ==============================================================================
# 2. HÀM ĐỆ QUY ĐỂ ÁP DỤNG LƯỢNG TỬ HÓA (GIỮ NGUYÊN)
# ==============================================================================
def apply_quantization_recursively(layer, f_before, f_after):
    if hasattr(layer, 'layers'):
        for sub_layer in layer.layers:
            apply_quantization_recursively(sub_layer, f_before, f_after)

    # Chỉ lượng tử hóa các lớp Conv2D và Dense có trọng số
    if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)) and layer.get_weights():
        original_weights_and_biases = layer.get_weights()

        # Bỏ qua các lớp không có kernel (ví dụ: lớp bias-only, hiếm gặp)
        if not original_weights_and_biases:
            return

        kernel_float32 = original_weights_and_biases[0]

        f_before.write(f"--- Layer: {layer.name} ---\n")
        f_before.write(np.array2string(kernel_float32.flatten()[:10], precision=8, separator=', '))
        f_before.write("\n\n")

        kernel_int8, scale = quantize_weights_q3_5(kernel_float32)
        kernel_dequantized = dequantize_weights_q_format(kernel_int8, scale)

        f_after.write(f"--- Layer: {layer.name} ---\n")
        f_after.write(f"Fixed Scale (Q3.5): {scale}\n")
        f_after.write(np.array2string(kernel_dequantized.flatten()[:10], precision=8, separator=', '))
        f_after.write("\n\n")

        new_weights = [kernel_dequantized]
        if len(original_weights_and_biases) > 1:
            new_weights.append(original_weights_and_biases[1]) # Giữ nguyên bias

        layer.set_weights(new_weights)
        print(f"Đã lượng tử hóa (Q3.5) layer: {layer.name}")

# ==============================================================================
# 3. TẢI VÀ ĐÁNH GIÁ MÔ HÌNH H-SWISH GỐC
# ==============================================================================
print("--- BƯỚC 1: ĐÁNH GIÁ MÔ HÌNH H-SWISH GỐC ---")

# *** THAY ĐỔI: Tải đúng file model h-swish đã huấn luyện ***
MODEL_FILE = '/home/tquocanh/Code/EfficientNetV2B0_Quantize/EfficientNet_Cifar10/cifar10_custom_hswish_trained/model.weights.h5'

try:
    original_model = tf.keras.models.load_model(MODEL_FILE)
except (OSError, IOError):
    print(f"LỖI: Không tìm thấy file '{MODEL_FILE}'.")
    print("Vui lòng đảm bảo bạn đã chạy script training cho mô hình h-swish trước.")
    exit()

# Tải dữ liệu CIFAR-10
(_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# *** THAY ĐỔI: Tiền xử lý dữ liệu cho phù hợp với mô hình h-swish ***
# Mô hình này có lớp Rescaling bên trong, nên ta chỉ cần resize ảnh.
# Không dùng `efficientnet.preprocess_input`.
x_test_resized = tf.image.resize(x_test, (224, 224))

# Chuyển nhãn sang one-hot để khớp với cách train
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Biên dịch và đánh giá mô hình gốc
original_model.compile(optimizer='adam',
                       loss='categorical_crossentropy', # Phải khớp với lúc train
                       metrics=['accuracy'])
loss_original, acc_original = original_model.evaluate(x_test_resized, y_test_cat, verbose=1)
print(f"Độ chính xác mô hình h-swish gốc (float32): {acc_original:.4f}")

# ==============================================================================
# 4. ÁP DỤNG LƯỢNG TỬ HÓA Q3.5
# ==============================================================================
print("\n--- BƯỚC 2: MÔ PHỎNG LƯỢNG TỬ HÓA (Q3.5) ---")
quantized_sim_model = tf.keras.models.clone_model(original_model)
quantized_sim_model.set_weights(original_model.get_weights())

with open('hswish_weights_before_q3_5.txt', 'w') as f_before, open('hswish_weights_after_q3_5.txt', 'w') as f_after:
    apply_quantization_recursively(quantized_sim_model, f_before, f_after)

print("\nĐã cập nhật các trọng số mô phỏng lượng tử hóa Q3.5 vào mô hình mới.")

# ==============================================================================
# 5. ĐÁNH GIÁ MÔ HÌNH MỚI VÀ SO SÁNH
# ==============================================================================
print("\n--- BƯỚC 3: ĐÁNH GIÁ VÀ SO SÁNH ---")
quantized_sim_model_path = 'cifar10_hswish_post_quant_sim_q3_5.keras'
quantized_sim_model.save(quantized_sim_model_path)
print(f"Đã lưu mô hình mô phỏng tại: '{quantized_sim_model_path}'")

quantized_sim_model.compile(optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])
loss_quant, acc_quant = quantized_sim_model.evaluate(x_test_resized, y_test_cat, verbose=1)

print("\n--- SO SÁNH HIỆU SUẤT ---")
print(f"Mô hình h-swish gốc:      {acc_original:.4f}")
print(f"Mô hình mô phỏng (Q3.5):   {acc_quant:.4f}")
print(f"Độ sụt giảm chính xác:     {acc_original - acc_quant:.4f}")