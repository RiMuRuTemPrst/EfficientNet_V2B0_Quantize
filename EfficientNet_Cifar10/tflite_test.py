# Tên file: evaluate_tflite_imagenet_auto_download.py
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as tfhub
import numpy as np
import os
import time

# Tối ưu hóa hiệu suất
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==============================================================================
# 0. CẤU HÌNH
# ==============================================================================
# ✅ MÔ HÌNH SẼ ĐƯỢC TẢI TỰ ĐỘNG TỪ URL NÀY
TFLITE_MODEL_URL = "https://tfhub.dev/google/lite-model/mobilenet_v2_1.0_224/1/default/1"

# ✅ ĐÁNH GIÁ TRÊN TOÀN BỘ 10,000 ẢNH
NUM_IMAGES_TO_EVALUATE = 10000

# ==============================================================================
# 1. TẢI MÔ HÌNH TFLITE VÀ LẤY THÔNG TIN
# ==============================================================================
print(f"--- Đang tải mô hình TFLite từ TensorFlow Hub... ---")
try:
    # Dùng TF Hub để tải mô hình và lấy đường dẫn tới file đã được cache
    tflite_model_path = tfhub.resolve(TFLITE_MODEL_URL)
    
    # Tải mô hình vào Interpreter
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
except Exception as e:
    print(f"LỖI: Không thể tải mô hình từ TensorFlow Hub. Vui lòng kiểm tra kết nối mạng.")
    print(f"Chi tiết lỗi: {e}")
    exit()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_height = input_details[0]['shape'][1]
input_width = input_details[0]['shape'][2]
is_quantized_model = input_details[0]['dtype'] == np.uint8

print(f"Mô hình yêu cầu đầu vào kích thước: ({input_height}, {input_width})")

# ==============================================================================
# 2. TẢI VÀ CHUẨN BỊ DỮ LIỆU (ImageNetV2)
# ==============================================================================
print("\n--- Đang tải và chuẩn bị dữ liệu ImageNetV2 (chỉ lần đầu) ---")
dataset, info = tfds.load(
    'imagenet_v2/matched-frequency',
    split='test',
    as_supervised=True,
    with_info=True
)

def preprocess_image(image, label):
    image = tf.image.resize(image, (input_height, input_width))
    if not is_quantized_model:
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = tf.expand_dims(image, axis=0)
    return image, label

dataset = dataset.take(NUM_IMAGES_TO_EVALUATE)

# ==============================================================================
# 3. VÒNG LẶP ĐÁNH GIÁ
# ==============================================================================
print(f"\n--- Bắt đầu đánh giá trên {NUM_IMAGES_TO_EVALUATE} ảnh ---")
correct_predictions = 0
start_time = time.time()

for i, (image, label) in enumerate(dataset):
    processed_image, _ = preprocess_image(image, label)
    interpreter.set_tensor(input_details[0]['index'], processed_image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Do MobileNet trên TF Hub được huấn luyện với 1001 lớp (lớp 0 là background)
    # nên ta cần trừ 1 để khớp với nhãn của ImageNet (0-999)
    predicted_index = np.argmax(output_data[0]) - 1

    if predicted_index == label.numpy():
        correct_predictions += 1
    
    print(f"\rĐã xử lý ảnh: {i + 1}/{NUM_IMAGES_TO_EVALUATE}", end='')

end_time = time.time()
print(f"\nĐánh giá hoàn tất trong {end_time - start_time:.2f} giây.")

# ==============================================================================
# 4. HIỂN THỊ KẾT QUẢ
# ==============================================================================
if NUM_IMAGES_TO_EVALUATE > 0:
    accuracy = (correct_predictions / NUM_IMAGES_TO_EVALUATE) * 100
    print("\n--- KẾT QUẢ ĐỘ CHÍNH XÁC ---")
    print(f"Số ảnh đã đánh giá: {NUM_IMAGES_TO_EVALUATE}")
    print(f"Số dự đoán đúng:    {correct_predictions}")
    print(f"✅ Độ chính xác Top-1: {accuracy:.2f}%")
else:
    print("Không có ảnh nào được đánh giá.")