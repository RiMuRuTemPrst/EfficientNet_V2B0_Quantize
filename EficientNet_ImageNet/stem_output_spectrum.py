import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Tắt các thông báo log không quan trọng
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# THAY ĐỔI: Hàm vẽ biểu đồ đã được nâng cấp
def plot_histogram(data, title, filename):
    """
    Vẽ histogram của một mảng dữ liệu, tự động zoom và thêm các thông số thống kê.
    """
    flat_data = data.flatten()
    
    # --- MỚI: Tính toán các thông số thống kê ---
    mean_val = np.mean(flat_data)
    std_val = np.std(flat_data)
    median_val = np.median(flat_data)
    p99_val = np.percentile(flat_data, 99) # Lấy giá trị ở phân vị 99%
    max_val = np.max(flat_data)

    # Tạo chuỗi text để hiển thị trên biểu đồ
    stats_text = (
        f"Mean: {mean_val:.4f}\n"
        f"Std Dev: {std_val:.4f}\n"
        f"Median: {median_val:.4f}\n"
        f"Max: {max_val:.4f}\n"
        f"99th Percentile: {p99_val:.4f}"
    )

    # --- Bắt đầu vẽ ---
    plt.figure(figsize=(12, 7))
    sns.histplot(flat_data, bins=100, kde=True)
    plt.title(title, fontsize=16)
    plt.xlabel("Giá trị", fontsize=12)
    plt.ylabel("Tần suất", fontsize=12)
    plt.grid(True)

    # --- MỚI: Giới hạn trục X để zoom vào vùng chính ---
    # Giới hạn đến 110% của giá trị ở phân vị 99
    plt.xlim(np.min(flat_data), p99_val * 1.1)

    # --- MỚI: Thêm box chứa thông số thống kê ---
    # `ha='right', va='top'` đặt box ở góc trên bên phải
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', horizontalalignment='right', 
             bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    plt.savefig(filename)
    plt.close()
    print(f"✅ Đã vẽ và lưu biểu đồ cải thiện vào: {filename}")


def main():
    """Hàm chính để thực hiện công việc."""
    IMAGE_SIZE = (224, 224)
    INPUT_HEX_FILE = 'input_raw_pixels.hex' 
    OUTPUT_DIR = "golden_stem_analysis"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- Đọc và tái tạo ảnh từ file hex ---
    print(f"--- Đang đọc file: {INPUT_HEX_FILE} ---")
    try:
        with open(INPUT_HEX_FILE, 'r') as f:
            hex_vals = [int(line.strip(), 16) for line in f.readlines()]
        pixel_array = np.array(hex_vals, dtype=np.uint8)
        image_data = pixel_array.reshape((1, *IMAGE_SIZE, 3))
        print("✅ Đọc và tái tạo ảnh thành công.")
    except Exception as e:
        print(f"❌ Lỗi: Không thể đọc file '{INPUT_HEX_FILE}'.\n{e}")
        return

    # --- Tải model, chạy suy luận và lấy output ---
    print("\n--- Đang chạy model Stem chuẩn...")
    try:
        input_for_model = image_data.astype(np.float32)
        original_full_model = tf.keras.applications.EfficientNetV2B0(weights='imagenet', input_shape=IMAGE_SIZE + (3,))
        golden_output_tensor = original_full_model.get_layer('stem_activation').output
        golden_model = tf.keras.models.Model(inputs=original_full_model.input, outputs=golden_output_tensor)
        golden_output = golden_model.predict(input_for_model, verbose=0)
        print("✅ Suy luận hoàn tất.")
    except Exception as e:
        print(f"❌ Lỗi khi chạy model: {e}")
        return

    # --- Vẽ biểu đồ ---
    print("\n--- Đang vẽ biểu đồ...")
    plot_histogram(
        data=golden_output, 
        title="Phổ Giá Trị Output Của Stem Block (Đã Zoom)",
        filename=os.path.join(OUTPUT_DIR, "golden_stem_output_histogram_improved.png")
    )
    
    print("\n🎉 Hoàn tất!")

if __name__ == '__main__':
    main()