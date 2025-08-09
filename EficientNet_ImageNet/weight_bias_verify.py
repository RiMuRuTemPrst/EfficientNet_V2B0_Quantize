import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==============================================================================
# BƯỚC 1: TẢI MODEL
# ==============================================================================
print("BƯỚC 1: Đang tải model EfficientNetV2-B0 (bản đầy đủ)...")
IMAGE_SIZE = (224, 224)

model = tf.keras.applications.EfficientNetV2B0(
    include_top=True,
    weights='imagenet',
    input_shape=IMAGE_SIZE + (3,)
)
print("✅ Tải model thành công.")

# ==============================================================================
# BƯỚC 2: CHUẨN BỊ DỮ LIỆU (TẢI 1 LẦN DUY NHẤT)
# ==============================================================================
print("\nBƯỚC 2: Đang tải và chuẩn bị bộ dữ liệu ImageNet...")
dataset_name = "imagenet_v2" 
# Tải dữ liệu một lần và lưu vào biến ds_raw
ds_raw, ds_info = tfds.load(dataset_name, split='test', with_info=True, as_supervised=True)
print(f"✅ Tải xong bộ dữ liệu.")

# ==============================================================================
# BƯỚC 3: QUANTIZE MODEL
# ==============================================================================
print("\nBƯỚC 3: Đang thực hiện quantize model...")

# Dùng ds_raw đã tải để tạo dữ liệu hiệu chỉnh
def preprocess_for_representative(image, label):
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.keras.applications.efficientnet_v2.preprocess_input(image)
    return image

representative_ds_preprocessed = ds_raw.map(preprocess_for_representative, num_parallel_calls=tf.data.AUTOTUNE)
representative_ds_for_gen = representative_ds_preprocessed.take(200).batch(1)

def representative_dataset_gen():
    for x in representative_ds_for_gen:
        yield [x]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
tflite_quant_model = converter.convert()
print("✅ Quantize thành công!")

# ==============================================================================
# BƯỚC 4: CHUẨN BỊ DỮ LIỆU ĐÁNH GIÁ
# ==============================================================================
print("\nBƯỚC 4: Đang chuẩn bị dữ liệu ImageNet để đánh giá...")

def preprocess_for_eval(image, label):
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.keras.applications.efficientnet_v2.preprocess_input(image)
    label = tf.cast(label, tf.int64)
    return image, label

# Dùng lại ds_raw đã tải, không cần tải lại
eval_ds = ds_raw.map(preprocess_for_eval, num_parallel_calls=tf.data.AUTOTUNE)

num_images_to_run = 10000
eval_ds = eval_ds.take(num_images_to_run)
eval_ds = eval_ds.batch(1)
print("✅ Chuẩn bị dữ liệu đánh giá thành công.")

# ==============================================================================
# BƯỚC 5: ĐÁNH GIÁ ĐỘ CHÍNH XÁC
# ==============================================================================
print(f"\nBƯỚC 5: Bắt đầu đánh giá độ chính xác trên {num_images_to_run} ảnh...")
interpreter = tf.lite.Interpreter(model_content=tflite_quant_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

num_correct = 0
for image, label in tqdm(eval_ds, total=num_images_to_run, desc="Đang đánh giá"):
    interpreter.set_tensor(input_details['index'], image.numpy())
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details['index'])
    predicted_class = np.argmax(prediction[0])

    if predicted_class == label.numpy()[0]:
        num_correct += 1

accuracy = (num_correct / num_images_to_run) * 100
print("\n" + "="*50)
print("KẾT QUẢ ACCURACY THAM CHIẾU (GOLDEN REFERENCE)")
print(f"-> Số ảnh đúng: {num_correct}/{num_images_to_run}")
print(f"-> Độ chính xác của model 8-bit: {accuracy:.2f}%")
print("="*50)