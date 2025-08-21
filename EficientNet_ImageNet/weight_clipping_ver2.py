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
# Thay đổi giá trị này để thử nghiệm các tỷ lệ khác nhau (ví dụ: 70.0, 90.0, 95.0)
CENTRAL_PERCENT_TO_KEEP = 99.0 


# --- 2. CÁC HÀM TIỆN ÍCH ---

def plot_clipping_decision_to_pdf_simplified(model, pdf_filename, central_percent):
    """
    Vẽ biểu đồ phân phối gốc và đánh dấu vùng trung tâm sẽ được giữ lại.
    """
    print(f"\nĐang tạo file PDF đơn giản hóa: {pdf_filename}...")
    
    lower_percentile = (100.0 - central_percent) / 2.0
    upper_percentile = 100.0 - lower_percentile
    
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
                
                lower_bound = np.percentile(w, lower_percentile)
                upper_bound = np.percentile(w, upper_percentile)
                ax.axvline(lower_bound, color='green', linestyle='--')
                ax.axvline(upper_bound, color='green', linestyle='--')
                
                ax.axvspan(lower_bound, upper_bound, alpha=0.2, color='gray', label=f'{central_percent}% Kept Region')

                ax.set_title("Weights")
                ax.set_xlabel("Value")
                ax.set_ylabel("Frequency")
                ax.legend()

                # --- Xử lý Biases (nếu có) ---
                if len(weights_and_biases) > 1 and weights_and_biases[1].size > 0:
                    b = weights_and_biases[1].flatten()
                    ax = axes[1]
                    
                    ax.hist(b, bins=50, color='coral', label='Original Distribution')
                    
                    lower_bound = np.percentile(b, lower_percentile)
                    upper_bound = np.percentile(b, upper_percentile)
                    ax.axvspan(lower_bound, upper_bound, alpha=0.2, color='gray', label=f'{central_percent}% Kept Region')
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


def clip_weights_by_percentile(model_to_clip, central_percent):
    """
    Ép các trọng số nằm ngoài vùng trung tâm được xác định bởi central_percent.
    """
    print(f"\nBắt đầu 'ép' trọng số về biên, giữ lại {central_percent}% ở trung tâm...")
    
    lower_percentile = (100.0 - central_percent) / 2.0
    upper_percentile = 100.0 - lower_percentile

    for layer in model_to_clip.layers:
        weights_and_biases = layer.get_weights()
        if not weights_and_biases:
            continue
            
        new_weights_and_biases = []
        for w_matrix in weights_and_biases:
            if w_matrix is None or w_matrix.size == 0:
                new_weights_and_biases.append(w_matrix)
                continue

            lower_bound = np.percentile(w_matrix, lower_percentile)
            upper_bound = np.percentile(w_matrix, upper_percentile)
            
            new_w_matrix = np.clip(w_matrix, a_min=lower_bound, a_max=upper_bound)
            
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
print("\nĐang tải mô hình EfficientNetV2B₀ hoàn chỉnh từ Keras...")
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

# Gọi hàm clipping, sử dụng biến cấu hình ở đầu file
clipped_model = clip_weights_by_percentile(clipped_model, central_percent=CENTRAL_PERCENT_TO_KEEP)

print(f"\n--- Đánh giá mô hình sau khi 'ép' trọng số ---")
clipped_loss, clipped_acc = clipped_model.evaluate(validation_dataset, verbose=1)
print(f"Độ chính xác của mô hình đã 'ép': {clipped_acc:.2%}")

# --- 6. IN KẾT QUẢ TỔNG KẾT ---
print("\n--- TỔNG KẾT (Weight Clipping) ---")
print(f"Giữ lại {CENTRAL_PERCENT_TO_KEEP}% trọng số ở trung tâm")
print(f"Độ chính xác mô hình gốc:   {original_acc:.2%}")
print(f"Độ chính xác mô hình đã 'ép': {clipped_acc:.2%}")
accuracy_loss_pp = (original_acc - clipped_acc) * 100
relative_loss = (original_acc - clipped_acc) / original_acc if original_acc != 0 else 0
print(f"Độ sụt giảm chính xác:      {accuracy_loss_pp:.2f} điểm phần trăm")
print(f"Tỷ lệ sụt giảm tương đối:   {relative_loss:.2%}")

# --- 7. TẠO FILE PDF TRỰC QUAN HÓA ---
pdf_filename = f"clipping_decision_{CENTRAL_PERCENT_TO_KEEP}percent.pdf"
# Gọi hàm vẽ biểu đồ, sử dụng biến cấu hình ở đầu file
plot_clipping_decision_to_pdf_simplified(custom_model, pdf_filename, central_percent=CENTRAL_PERCENT_TO_KEEP)

print("\nHoàn tất toàn bộ quá trình!")