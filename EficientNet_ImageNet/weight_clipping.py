# --- 0. IMPORT CÁC THƯ VIỆN CẦN THIẾT ---
import tensorflow as tf
from tensorflow import keras
import numpy as np
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import warnings

# Tắt các cảnh báo không quan trọng từ matplotlib để output gọn gàng hơn
warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')


# --- 1. CÁC HÀM TIỆN ÍCH ---

def plot_clipping_decision_to_pdf(model, pdf_filename):
    """
    Vẽ biểu đồ phân phối gốc và đánh dấu (tô màu) vùng 80% trung tâm sẽ được giữ lại.
    Đây là phiên bản trực quan nhất để thể hiện ý tưởng.
    """
    print(f"\nĐang tạo file PDF thể hiện quyết định clipping: {pdf_filename}...")
    
    with PdfPages(pdf_filename) as pdf:
        for layer in model.layers:
            if isinstance(layer, (keras.layers.Conv2D, keras.layers.Dense)):
                weights_and_biases = layer.get_weights()
                
                if not weights_and_biases or len(weights_and_biases[0]) == 0:
                    continue

                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                fig.suptitle(f'Layer: {layer.name} ({type(layer).__name__}) - Clipping Decision', fontsize=16)

                # --- Xử lý Weights ---
                w = weights_and_biases[0].flatten()
                ax = axes[0]
                
                ax.hist(w, bins=100, color='royalblue', label='Original Distribution')
                
                orig_min, orig_max = w.min(), w.max()
                ax.axvline(orig_min, color='red', linestyle=':')
                ax.axvline(orig_max, color='red', linestyle=':')
                y_pos_red = ax.get_ylim()[1] * 0.9
                ax.text(orig_min, y_pos_red, f'orig_min={orig_min:.4f}', color='red', ha='left')
                ax.text(orig_max, y_pos_red, f'orig_max={orig_max:.4f}', color='red', ha='right')

                lower_bound = np.percentile(w, 10.0)
                upper_bound = np.percentile(w, 90.0)
                ax.axvline(lower_bound, color='green', linestyle='--')
                ax.axvline(upper_bound, color='green', linestyle='--')
                
                ax.axvspan(lower_bound, upper_bound, alpha=0.2, color='gray', label='80% Kept Region')
                
                # BỎ COMMENT DÒNG DƯỚI ĐÂY NẾU BẠN MUỐN TRỤC X VỪA KHÍT DỮ LIỆU
                # ax.set_xlim(orig_min, orig_max)

                ax.set_title("Weights")
                ax.set_xlabel("Value")
                ax.set_ylabel("Frequency")
                ax.legend()

                # --- Xử lý Biases (nếu có) ---
                if len(weights_and_biases) > 1 and weights_and_biases[1].size > 0:
                    b = weights_and_biases[1].flatten()
                    ax = axes[1]
                    
                    ax.hist(b, bins=50, color='coral', label='Original Distribution')
                    orig_min, orig_max = b.min(), b.max()
                    ax.axvline(orig_min, color='red', linestyle=':')
                    ax.axvline(orig_max, color='red', linestyle=':')
                    
                    lower_bound = np.percentile(b, 10.0)
                    upper_bound = np.percentile(b, 90.0)
                    ax.axvspan(lower_bound, upper_bound, alpha=0.2, color='gray', label='80% Kept Region')
                    ax.axvline(lower_bound, color='green', linestyle='--')
                    ax.axvline(upper_bound, color='green', linestyle='--')

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


def clip_weights_by_percentile(model_to_clip):
    """
    Ép các trọng số nằm ngoài vùng trung tâm 80% về giá trị biên.
    """
    print(f"\nBắt đầu 'ép' trọng số về biên theo ý tưởng của bạn...")
    
    for layer in model_to_clip.layers:
        weights_and_biases = layer.get_weights()
        if not weights_and_biases:
            continue
            
        new_weights_and_biases = []
        for w_matrix in weights_and_biases:
            if w_matrix is None or w_matrix.size == 0:
                new_weights_and_biases.append(w_matrix)
                continue

            lower_bound = np.percentile(w_matrix, 10.0)
            upper_bound = np.percentile(w_matrix, 90.0)
            
            new_w_matrix = np.clip(w_matrix, a_min=lower_bound, a_max=upper_bound)
            
            new_weights_and_biases.append(new_w_matrix)

        layer.set_weights(new_weights_and_biases)
    
    print("Hoàn tất việc 'ép' trọng số.")
    return model_to_clip


# --- 2. CHUẨN BỊ DỮ LIỆU VÀ CÀI ĐẶT ---
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

# --- 3. ĐÁNH GIÁ MÔ HÌNH GỐC (BASELINE) ---
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

# --- 4. ÁP DỤNG Ý TƯỞNG CỦA BẠN VÀ ĐÁNH GIÁ ---
clipped_model = tf.keras.models.clone_model(custom_model)
clipped_model.set_weights(custom_model.get_weights())
clipped_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Gọi hàm clipping theo ý tưởng của bạn
clipped_model = clip_weights_by_percentile(clipped_model)

print(f"\n--- Đánh giá mô hình sau khi 'ép' trọng số ---")
clipped_loss, clipped_acc = clipped_model.evaluate(validation_dataset, verbose=1)
print(f"Độ chính xác của mô hình đã 'ép': {clipped_acc:.2%}")

# --- 5. IN KẾT QUẢ TỔNG KẾT ---
print("\n--- TỔNG KẾT (Weight Clipping) ---")
print(f"Độ chính xác mô hình gốc:   {original_acc:.2%}")
print(f"Độ chính xác mô hình đã 'ép': {clipped_acc:.2%}")
accuracy_loss_pp = (original_acc - clipped_acc) * 100
relative_loss = (original_acc - clipped_acc) / original_acc if original_acc != 0 else 0
print(f"Độ sụt giảm chính xác:      {accuracy_loss_pp:.2f} điểm phần trăm")
print(f"Tỷ lệ sụt giảm tương đối:   {relative_loss:.2%}")

# --- 6. TẠO FILE PDF TRỰC QUAN HÓA ---
plot_clipping_decision_to_pdf(custom_model, "clipping_decision_visualization.pdf")

print("\nHoàn tất toàn bộ quá trình!")