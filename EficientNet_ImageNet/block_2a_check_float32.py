import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
import math

# Tắt các thông báo log không quan trọng
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==============================================================================
# PHẦN 1: CÁC HÀM HỖ TRỢ
# ==============================================================================
def calculate_quantization_params_pow2(tensor_data):
    """Tính toán scale (dạng 2^n) và zero-point cho tensor đầu ra."""
    min_val, max_val = np.min(tensor_data), np.max(tensor_data)
    initial_scale = (max_val - min_val) / 255
    if initial_scale <= 0: return 1.0, 0
    power_of_2_scale = 2**round(math.log2(initial_scale))
    return power_of_2_scale, 0

def quantize_tensor(tensor_data, scale, zero_point):
    """Lượng tử hóa tensor float32 sang int8 và trả về accumulator int32."""
    scaled_float = tensor_data / scale
    accumulator_int32 = np.round(scaled_float).astype(np.int32)
    clipped_values = np.clip(accumulator_int32, -128, 127)
    return clipped_values.astype(np.int8), accumulator_int32

def write_float32_to_4byte_hex(filepath, float_tensor):
    """Ghi tensor float32 ra file hex, mỗi giá trị 4 byte trên một dòng."""
    byte_array = float_tensor.flatten().view(np.uint8)
    with open(filepath, 'w') as f:
        for i in range(0, len(byte_array), 4):
            f.write(''.join(f'{byte:02x}' for byte in byte_array[i:i+4]) + '\n')

def write_int32_to_4byte_hex(filepath, int32_tensor):
    """Ghi tensor int32 ra file hex, mỗi giá trị 4 byte trên một dòng."""
    flat_tensor = int32_tensor.flatten()
    with open(filepath, 'w') as f:
        for num in flat_tensor:
            f.write(f'{(num & 0xffffffff):08x}\n')

# ==============================================================================
# PHẦN 2: KỊCH BẢN KIỂM THỬ HOÀN CHỈNH (KHÔNG GỘP LỚP)
# ==============================================================================
print("\n--- KIỂM TRA KIẾN TRÚC GỐC (KHÔNG GỘP LỚP) ---")

IMAGE_SIZE = (224, 224)
output_check_dir = "block2a_output"
os.makedirs(output_check_dir, exist_ok=True)

# --- BƯỚC 1: Tải model gốc ---
print("BƯỚC 1: Đang tải model EfficientNetV2B0 gốc...")
original_full_model = tf.keras.applications.EfficientNetV2B0(weights='imagenet', input_shape=IMAGE_SIZE + (3,))
print("✅ Tải model thành công.")

# --- BƯỚC 2: Xây dựng model "chuẩn vàng" ---
output_layer_name_golden = 'block2a_project_bn'
original_model = tf.keras.models.Model(
    inputs=original_full_model.input,
    outputs=original_full_model.get_layer(output_layer_name_golden).output,
    name="golden_reference_model"
)
print(f"✅ Đã trích xuất model chuẩn với đầu ra là '{output_layer_name_golden}'.")


# --- BƯỚC 3: Xây dựng model tùy chỉnh với các lớp Conv2D và BN riêng biệt ---
print("\nBƯỚC 2: Xây dựng model tùy chỉnh với kiến trúc đầy đủ...")
custom_inputs = tf.keras.layers.Input(shape=(224, 224, 3), name='input_layer')
# Stem
x = tf.keras.layers.Rescaling(1./255)(custom_inputs)
x = tf.keras.layers.Normalization(axis=-1, mean=[0.485, 0.456, 0.406], variance=[0.229**2, 0.224**2, 0.225**2])(x)
x = tf.keras.layers.Conv2D(32, 3, strides=2, padding='same', use_bias=False, name='stem_conv_custom')(x)
x = tf.keras.layers.BatchNormalization(epsilon=1.001e-5, name='stem_bn_custom')(x)
x = tf.keras.layers.Activation('swish')(x)
# Block 1a
x = tf.keras.layers.Conv2D(16, 3, padding='same', use_bias=False, name='block1a_project_conv_custom')(x)
x = tf.keras.layers.BatchNormalization(epsilon=1.001e-5, name='block1a_project_bn_custom')(x)
x = tf.keras.layers.Activation('swish')(x)
# Block 2a
x = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1)))(x)
x = tf.keras.layers.Conv2D(64, 3, strides=2, padding='valid', use_bias=False, name='block2a_expand_conv_custom')(x)
x = tf.keras.layers.BatchNormalization(epsilon=1.001e-5, name='block2a_expand_bn_custom')(x)
x = tf.keras.layers.Activation('swish')(x)
x = tf.keras.layers.Conv2D(32, 1, padding='same', use_bias=False, name='block2a_project_conv_custom')(x)
x = tf.keras.layers.BatchNormalization(epsilon=1.001e-5, name='block2a_project_bn_custom')(x)

custom_model = tf.keras.models.Model(inputs=custom_inputs, outputs=x, name="custom_model_no_fusion")
print("✅ Đã xây dựng model tùy chỉnh (không gộp lớp).")


# --- BƯỚC 4: Sao chép trọng số gốc ---
print(f"\nBƯỚC 3: Sao chép trọng số gốc...")
layer_mapping = {
    'stem_conv_custom': 'stem_conv', 'stem_bn_custom': 'stem_bn',
    'block1a_project_conv_custom': 'block1a_project_conv', 'block1a_project_bn_custom': 'block1a_project_bn',
    'block2a_expand_conv_custom': 'block2a_expand_conv', 'block2a_expand_bn_custom': 'block2a_expand_bn',
    'block2a_project_conv_custom': 'block2a_project_conv', 'block2a_project_bn_custom': 'block2a_project_bn',
}
for custom_name, original_name in layer_mapping.items():
    custom_model.get_layer(custom_name).set_weights(original_full_model.get_layer(original_name).get_weights())
print("✅ Đã sao chép đầy đủ trọng số gốc.")


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


# --- MỚI: Bổ sung lại phần ghi file ---
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


# --- BƯỚC 7: So sánh ---
print("\nBƯỚC 7: So sánh kết quả MSE...")
mse = np.mean(np.square(output_golden - output_custom))
print(f"\n=> Sai số bình phương trung bình (MSE): {mse:.15f}")

if mse < 1e-10: 
    print("\n✅✅✅ THÀNH CÔNG! Kiến trúc tùy chỉnh hoạt động chính xác.")
else:
    print("\n❌❌❌ THẤT BẠI: Vẫn có sai lệch.")