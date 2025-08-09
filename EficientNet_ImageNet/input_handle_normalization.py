import os
import random
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_datasets as tfds
import math

# Tắt các thông báo log không quan trọng
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==== CÁC HÀM HỖ TRỢ TỪ TRƯỚC ==== #
def calculate_quantization_params_pow2(tensor_data):
    """Tính toán scale (dạng 2^n) và zero-point cho tensor."""
    min_val, max_val = np.min(tensor_data), np.max(tensor_data)
    q_min, q_max = -128, 127
    initial_scale = (max_val - min_val) / (q_max - q_min)
    
    if initial_scale <= 0: return 1.0, 0
    
    power_of_2_scale = 2**round(math.log2(initial_scale))
    zero_point = round(q_min - min_val / power_of_2_scale)
    
    return power_of_2_scale, int(np.clip(zero_point, q_min, q_max))

def quantize_tensor_asymmetric(tensor_data, scale, zero_point):
    """Lượng tử hóa bất đối xứng."""
    quantized_val = np.round(tensor_data / scale) + zero_point
    return np.clip(quantized_val, -128, 127).astype(np.int8)


# ==== CẤU HÌNH CHƯƠNG TRÌNH ==== #
INPUT_SIZE = (224, 224)
OUTPUT_DIR = "input_files"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==== 1) Load ảnh ngẫu nhiên từ ImageNet ==== #
print("[+] Đang tải ảnh ngẫu nhiên từ ImageNet...")
try:
    ds = tfds.load("imagenet_v2", split='test', as_supervised=True)
    image_tensor, _ = next(iter(ds.take(1)))
    img_array_float32 = tf.image.resize(image_tensor, INPUT_SIZE, method='bilinear').numpy()
    print("✅ Đã tải ảnh.")
except Exception as e:
    print(f"❌ Lỗi khi tải ảnh: {e}.")
    exit()

Image.fromarray(img_array_float32.astype(np.uint8)).save(os.path.join(OUTPUT_DIR, "input_image_visual.png"))
print(f"✅ Đã lưu ảnh trực quan.")


# ==== 2) Tiền xử lý ảnh (Rescaling & Normalization) ==== #
print("\n[+] Đang tiền xử lý ảnh...")
preprocessing_input = tf.keras.layers.Input(shape=(224, 224, 3))
x = tf.keras.layers.Rescaling(1./255)(preprocessing_input)
x = tf.keras.layers.Normalization(
    axis=-1, 
    mean=[0.485, 0.456, 0.406],
    variance=[0.229**2, 0.224**2, 0.225**2]
)(x)
preprocessing_model = tf.keras.models.Model(inputs=preprocessing_input, outputs=x)

input_batch = np.expand_dims(img_array_float32, axis=0)
normalized_output_float32 = preprocessing_model.predict(input_batch, verbose=0)
print("✅ Đã tiền xử lý ảnh xong.")


# ==== 3) Lượng tử hóa input đã được chuẩn hóa ==== #
print("\n[+] Đang lượng tử hóa input đã chuẩn hóa...")
input_scale, input_zero_point = calculate_quantization_params_pow2(normalized_output_float32)
input_quantized_int8 = quantize_tensor_asymmetric(normalized_output_float32, input_scale, input_zero_point)
print(f"✅ Lượng tử hóa xong với scale={input_scale}, zero_point={input_zero_point}")


# ==== 4) Phân Tích Sai Lệch và Xuất File ==== #
print("\n[+] Phân tích sai lệch và xuất các file output...")

# De-quantize để so sánh
dequantized_output_float32 = (input_quantized_int8.astype(np.float32) - input_zero_point) * input_scale

# --- In ra các thông số thống kê ---
min_before = np.min(normalized_output_float32)
max_before = np.max(normalized_output_float32)
std_before = np.std(normalized_output_float32)
print(f"  - Trước Quantize (float32): Min={min_before:.4f}, Max={max_before:.4f}, Độ lệch chuẩn={std_before:.4f}")

min_after = np.min(dequantized_output_float32)
max_after = np.max(dequantized_output_float32)
std_after = np.std(dequantized_output_float32)
print(f"  - Sau Quantize (de-quantized): Min={min_after:.4f}, Max={max_after:.4f}, Độ lệch chuẩn={std_after:.4f}")

mse = np.mean(np.square(normalized_output_float32 - dequantized_output_float32))
print(f"  - MSE giữa hai phiên bản: {mse:.6f}")

# --- Xuất các file ---
# File input cuối cùng (int8) cho phần cứng
with open(os.path.join(OUTPUT_DIR, "input_quantized_int8.hex"), 'w') as f:
    for val in input_quantized_int8.flatten():
        f.write(f"{val & 0xFF:02X}\n")

# File chứa các tham số lượng tử hóa của input
with open(os.path.join(OUTPUT_DIR, "input_quantization_params.txt"), 'w') as f:
    f.write(f"scale: {input_scale}\n")
    f.write(f"zero_point: {input_zero_point}\n")

# File float32 gốc (đã chuẩn hóa) để tham khảo
with open(os.path.join(OUTPUT_DIR, "input_normalized_float32.txt"), 'w') as f:
    for val in normalized_output_float32.flatten():
        f.write(f"{val}\n")

# MỚI: File float32 sau khi de-quantize để so sánh
with open(os.path.join(OUTPUT_DIR, "input_dequantized_float32.txt"), 'w') as f:
    for val in dequantized_output_float32.flatten():
        f.write(f"{val}\n")

print("\n--- Hoàn tất ---")