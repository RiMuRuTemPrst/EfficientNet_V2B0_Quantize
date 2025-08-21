# --- 0. IMPORT CÁC THƯ VIỆN CẦN THIẾT ---
import tensorflow as tf
from tensorflow import keras
import numpy as np
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')

# --- 1. CẤU HÌNH THÍ NGHIỆM ---
# Tỷ lệ của giá trị tuyệt đối max mà chúng ta muốn giữ lại
# Ví dụ: 80.0 có nghĩa là giữ lại phổ giá trị trong khoảng [-80%*max_abs, +80%*max_abs]
CLIP_PERCENTAGE_OF_MAX = 80.0 


# --- 2. CÁC HÀM TIỆN ÍCH (VIẾT LẠI) ---

def plot_max_abs_clipping_decision(model, pdf_filename, clip_percentage):
    """
    Vẽ biểu đồ phân phối gốc và đánh dấu vùng clipping dựa trên % của giá trị tuyệt đối max.
    """
    print(f"\nĐang tạo file PDF thể hiện quyết định clipping theo max_abs: {pdf_filename}...")
    
    with PdfPages(pdf_filename) as pdf:
        for layer in model.layers:
            if isinstance(layer, (keras.layers.Conv2D, keras.layers.Dense)):
                weights_and_biases = layer.get_weights()
                
                if not weights_and_biases or len(weights_and_biases[0]) == 0:
                    continue

                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                fig.suptitle(f'Layer: {layer.name} ({type(layer).__name__})', fontsize=16)

                # --- Xử lý Weights ---
                w = weights_and_biases[0].flatten()
                ax = axes[0]
                
                ax.hist(w, bins=100, color='royalblue', label='Original Distribution')
                
                # Tính toán ngưỡng dựa trên giá trị tuyệt đối max
                max_abs_val = np.max(np.abs(w))
                threshold = max_abs_val * (clip_percentage / 100.0)
                
                # Vẽ vùng được giữ lại
                ax.axvline(-threshold, color='green', linestyle='--')
                ax.axvline(threshold, color='green', linestyle='--')
                ax.axvspan(-threshold, threshold, alpha=0.2, color='gray', label=f'Clip Region (+/- {threshold:.2f})')

                ax.set_title("Weights")
                ax.set_xlabel("Value")
                ax.set_ylabel("Frequency")
                ax.legend()

                # --- Xử lý Biases (nếu có) ---
                if len(weights_and_biases) > 1 and weights_and_biases[1].size > 0:
                    b = weights_and_biases[1].flatten()
                    ax = axes[1]
                    
                    ax.hist(b, bins=50, color='coral', label='Original Distribution')
                    
                    max_abs_val_b = np.max(np.abs(b))
                    threshold_b = max_abs_val_b * (clip_percentage / 100.0)

                    ax.axvline(-threshold_b, color='green', linestyle='--')
                    ax.axvline(threshold_b, color='green', linestyle='--')
                    ax.axvspan(-threshold_b, threshold_b, alpha=0.2, color='gray', label=f'Clip Region (+/- {threshold_b:.2f})')

                    ax.set_title("Biases")
                    ax.set_xlabel("Value")
                    ax.legend()
                else:
                    axes[1].axis('off')
                    axes[1].set_title("No Biases")
                
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                pdf.savefig(fig)
                plt.close(fig)
                
    print(f"Đã tạo thành công file: {pdf_filename}")


def clip_weights_by_max_abs_value(model_to_clip, clip_percentage):
    """
    Ép các trọng số vào khoảng [-threshold, +threshold], với threshold được tính bằng
    % của giá trị tuyệt đối lớn nhất.
    """
    print(f"\nBắt đầu 'ép' trọng số theo {clip_percentage}% của giá trị tuyệt đối max...")
    
    for layer in model_to_clip.layers:
        weights_and_biases = layer.get_weights()
        if not weights_and_biases:
            continue
            
        new_weights_and_biases = []
        for w_matrix in weights_and_biases:
            if w_matrix is None or w_matrix.size == 0:
                new_weights_and_biases.append(w_matrix)
                continue

            # 1. Tìm giá trị tuyệt đối lớn nhất
            max_abs_val = np.max(np.abs(w_matrix))
            
            # 2. Tính ngưỡng clipping mới
            threshold = max_abs_val * (clip_percentage / 100.0)
            
            # 3. Ép trọng số vào khoảng [-threshold, threshold]
            new_w_matrix = np.clip(w_matrix, a_min=-threshold, a_max=threshold)
            
            new_weights_and_biases.append(new_w_matrix)

        layer.set_weights(new_weights_and_biases)
    
    print("Hoàn tất việc 'ép' trọng số.")
    return model_to_clip


# --- 3. CHUẨN BỊ DỮ LIỆU VÀ CÀI ĐẶT ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def preprocess_data(image, label):
    image = tf.image.resize(image, IMG_SIZE, method='bicubic')
    image = tf.keras.applications.efficientnet_v2.preprocess_input(image)
    return image, label

print("Đang tải và chuẩn bị bộ dữ liệu ImageNetV2...")
validation_dataset = tfds.load(
    'imagenet_v2/matched-frequency', 
    split='test', 
    shuffle_files=False, 
    as_supervised=True
)
validation_dataset = validation_dataset.map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.batch(BATCH_SIZE)
validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
print("Đã chuẩn bị xong dữ liệu.")

# --- 4. ĐÁNH GIÁ MÔ HÌNH GỐC (BASELINE) ---
print("\nĐang tải mô hình EfficientNetV2B0 hoàn chỉnh từ Keras...")
custom_model = tf.keras.applications.EfficientNetV2B0(
    weights="imagenet",
    include_top=True
)
custom_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n--- Đánh giá mô hình gốc trên ImageNetV2 ---")
original_loss, original_acc = custom_model.evaluate(validation_dataset, verbose=1)
print(f"Độ chính xác gốc: {original_acc:.2%}")

# --- 5. ÁP DỤNG Ý TƯỞNG CỦA BẠN VÀ ĐÁNH GIÁ ---
clipped_model = tf.keras.models.clone_model(custom_model)
clipped_model.set_weights(custom_model.get_weights())
clipped_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Gọi hàm clipping ĐÃ ĐƯỢC VIẾT LẠI theo đúng ý tưởng
clipped_model = clip_weights_by_max_abs_value(clipped_model, clip_percentage=CLIP_PERCENTAGE_OF_MAX)

print(f"\n--- Đánh giá mô hình sau khi 'ép' trọng số ---")
clipped_loss, clipped_acc = clipped_model.evaluate(validation_dataset, verbose=1)
print(f"Độ chính xác của mô hình đã 'ép': {clipped_acc:.2%}")

# --- 6. IN KẾT QUẢ TỔNG KẾT ---
print("\n--- TỔNG KẾT (Clipping by % of Max Absolute Value) ---")
print(f"Giữ lại phổ giá trị bằng {CLIP_PERCENTAGE_OF_MAX}% của giá trị tuyệt đối max")
print(f"Độ chính xác mô hình gốc:   {original_acc:.2%}")
print(f"Độ chính xác mô hình đã 'ép': {clipped_acc:.2%}")
accuracy_loss_pp = (original_acc - clipped_acc) * 100
relative_loss = (original_acc - clipped_acc) / original_acc if original_acc != 0 else 0
print(f"Độ sụt giảm chính xác:      {accuracy_loss_pp:.2f} điểm phần trăm")
print(f"Tỷ lệ sụt giảm tương đối:   {relative_loss:.2%}")

# --- 7. TẠO FILE PDF TRỰC QUAN HÓA ---
pdf_filename = f"max_abs_clipping_{CLIP_PERCENTAGE_OF_MAX}percent.pdf"
# Gọi hàm vẽ biểu đồ ĐÃ ĐƯỢC VIẾT LẠI
plot_max_abs_clipping_decision(custom_model, pdf_filename, clip_percentage=CLIP_PERCENTAGE_OF_MAX)

print("\nHoàn tất toàn bộ quá trình!")