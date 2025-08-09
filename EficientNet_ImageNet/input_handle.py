import os
import random
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_datasets as tfds

# Tắt các thông báo log không quan trọng
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==== CẤU HÌNH CHƯƠNG TRÌNH ==== #
IMAGENET_VAL_DIR = '/path/to/imagenet/val'            # Vui lòng thay đổi đường dẫn này nếu bạn có ImageNet cục bộ!
INPUT_SIZE = (224, 224)  # Kích thước đầu vào của EfficientNet-V2 B0
RAW_HEX_PATH = 'input_raw_pixels.hex'
Q_HEX_PATH  = 'input_quantized_pixels.hex'
RAW_DEC_PATH = 'input_raw_pixels.txt'             # <--- FILE MỚI
Q_DEC_PATH = 'input_quantized_pixels.txt'         # <--- FILE MỚI
VISUAL_IMAGE_PATH = 'input_image_visual.png' 

# ==== 1) Load ảnh ngẫu nhiên từ ImageNet (sử dụng tfds.load cho tiện) ==== #
print("[+] Đang tải và chọn ảnh ngẫu nhiên từ ImageNet...")
try:
    # Tải bộ dữ liệu ImageNetV2 (chỉ lấy 1 ảnh)
    # Lần đầu chạy có thể tải dữ liệu, mất thời gian.
    ds = tfds.load("imagenet_v2", split='test', as_supervised=True)
    image_tensor, _ = next(iter(ds.take(1)))
    img_array = tf.image.resize(image_tensor, INPUT_SIZE, method='bilinear').numpy().astype(np.uint8)
    print("✅ Đã tải ảnh ngẫu nhiên từ ImageNetV2.")

except Exception as e:
    print(f"❌ Lỗi khi tải ảnh từ tfds.load: {e}")
    print(f"   Vui lòng kiểm tra kết nối internet hoặc cài đặt tfds.")
    print(f"   Thử tải ảnh từ thư mục cục bộ '{IMAGENET_VAL_DIR}' (nếu có).")
    
    # Fallback: Nếu không tải được từ tfds, thử từ thư mục cục bộ
    if os.path.isdir(IMAGENET_VAL_DIR):
        classes = [d for d in os.listdir(IMAGENET_VAL_DIR) if os.path.isdir(os.path.join(IMAGENET_VAL_DIR, d))]
        if not classes:
            raise RuntimeError(f"Không tìm thấy thư mục con nào trong {IMAGENET_VAL_DIR}")
        cls = random.choice(classes)
        cls_dir = os.path.join(IMAGENET_VAL_DIR, cls)
        imgs = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.png'))]
        if not imgs:
            raise RuntimeError(f"Không tìm thấy ảnh trong {cls_dir}")
        img_path = os.path.join(cls_dir, random.choice(imgs))
        img = Image.open(img_path).convert('RGB').resize(INPUT_SIZE)
        img_array = np.array(img, dtype=np.uint8)
        print(f"✅ Đã tải ảnh ngẫu nhiên từ thư mục cục bộ: {img_path}")
    else:
        raise RuntimeError("Không thể tải ảnh từ ImageNet. Vui lòng kiểm tra cấu hình IMAGENET_VAL_DIR hoặc kết nối internet.")

# Lưu ảnh để trực quan
Image.fromarray(img_array).save(VISUAL_IMAGE_PATH)
print(f"✅ Đã lưu ảnh trực quan vào: {VISUAL_IMAGE_PATH}")


# ==== 2) Xuất mảng pixel thô (uint8) ra file hex và dec ==== #
# Xuất raw hex
with open(RAW_HEX_PATH, 'w') as f:
    for val in img_array.flatten():
        f.write(f"{val:02X}\n") # Định dạng 2 ký tự hex, thêm 0 nếu cần
print(f"[+] Đã xuất raw input pixels (0-255) vào: {RAW_HEX_PATH}")

# Xuất raw decimal <--- FILE MỚI
with open(RAW_DEC_PATH, 'w') as f:
    for val in img_array.flatten():
        f.write(f"{val}\n") # Ghi giá trị thập phân
print(f"[+] Đã xuất raw input pixels (0-255, thập phân) vào: {RAW_DEC_PATH}")


# ==== 3) Quantize đầu vào bằng cách đơn giản nhất (uint8 -> int8) ==== #
arr_quantized_int8 = img_array.astype(np.int8)


# ==== 4) Xuất mảng đã quantize (int8) ra file hex và dec ==== #
# Xuất quantized hex
with open(Q_HEX_PATH, 'w') as f:
    for val in arr_quantized_int8.flatten():
        f.write(f"{val & 0xFF:02X}\n") # Chuyển đổi số nguyên có dấu 8-bit sang biểu diễn hex của byte không dấu
print(f"[+] Đã xuất quantized input (int8) vào: {Q_HEX_PATH}")

# Xuất quantized decimal <--- FILE MỚI
with open(Q_DEC_PATH, 'w') as f:
    for val in arr_quantized_int8.flatten():
        f.write(f"{val}\n") # Ghi giá trị thập phân
print(f"[+] Đã xuất quantized input (int8, thập phân) vào: {Q_DEC_PATH}")


print("\n--- Hoàn tất quá trình tạo file input ---")
print(f"File ảnh trực quan: {VISUAL_IMAGE_PATH}")
print(f"File hex raw (uint8): {RAW_HEX_PATH}")
print(f"File dec raw (uint8): {RAW_DEC_PATH}")
print(f"File hex quantized (int8): {Q_HEX_PATH}")
print(f"File dec quantized (int8): {Q_DEC_PATH}")

# --- Ví dụ minh họa nhỏ ---
print("\n--- Ví dụ so sánh 5 giá trị pixel đầu tiên ---")
first_5_raw_dec = img_array.flatten()[:5]
first_5_raw_hex = [f"{val:02X}" for val in first_5_raw_dec]
first_5_quant_dec = arr_quantized_int8.flatten()[:5]
first_5_quant_hex = [f"{val & 0xFF:02X}" for val in first_5_quant_dec]

print(f"Raw uint8 (thập phân): {first_5_raw_dec}")
print(f"Raw uint8 (hex): {first_5_raw_hex}")
print(f"Quantized int8 (thập phân): {first_5_quant_dec}")
print(f"Quantized int8 (hex): {first_5_quant_hex}")