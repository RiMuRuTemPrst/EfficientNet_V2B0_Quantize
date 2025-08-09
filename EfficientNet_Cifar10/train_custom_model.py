import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import os

# Tối ưu hóa hiệu suất
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==============================================================================
# 1. HÀM XÂY DỰNG MÔ HÌNH VỚI H-SWISH
# ==============================================================================

# ✅ SỬA LỖI: Tự định nghĩa hàm hard_swish để không phụ thuộc vào phiên bản TF
def custom_hard_swish(x):
    """Tự triển khai hàm hard-swish theo công thức."""
    return x * tf.nn.relu6(x + 3) / 6

def build_model():
    """Xây dựng mô hình với hàm kích hoạt h-swish để tối ưu cho phần cứng."""
    inputs = layers.Input(shape=(224, 224, 3), name='input_layer')

    x = layers.Rescaling(1./255, name='rescaling')(inputs)

    # Sử dụng hàm tự định nghĩa
    H_SWISH_ACTIVATION = custom_hard_swish

    # Stem
    x = layers.Conv2D(32, 3, strides=2, padding='same', use_bias=True, activation=H_SWISH_ACTIVATION, name='stem_conv')(x)

    # Block 1
    x = layers.Conv2D(16, 3, padding='same', use_bias=True, activation=H_SWISH_ACTIVATION, name='block1_conv')(x)

    # Block 2a
    x = layers.Conv2D(64, 3, strides=2, padding='same', use_bias=True, activation=H_SWISH_ACTIVATION, name='block2a_expand')(x)
    x = layers.Conv2D(32, 1, padding='same', use_bias=True, name='block2a_project')(x)

    # Block 2b (Residual)
    x_skip = x
    x = layers.Conv2D(128, 3, padding='same', use_bias=True, activation=H_SWISH_ACTIVATION, name='block2b_expand')(x)
    x = layers.Conv2D(32, 1, padding='same', use_bias=True, name='block2b_project')(x)
    x = layers.Dropout(0.0, name='block2b_drop')(x)
    x = layers.Add(name='block2b_add')([x, x_skip])

    # Block 3a
    x = layers.Conv2D(128, 3, strides=2, padding='same', use_bias=True, activation=H_SWISH_ACTIVATION, name='block3a_expand')(x)
    x = layers.Conv2D(48, 1, padding='same', use_bias=True, name='block3a_project')(x)

    # Block 3b (Residual)
    x_skip = x
    x = layers.Conv2D(192, 3, padding='same', use_bias=True, activation=H_SWISH_ACTIVATION, name='block3b_expand')(x)
    x = layers.Conv2D(48, 1, padding='same', use_bias=True, name='block3b_project')(x)
    x = layers.Dropout(0.0, name='block3b_drop')(x)
    x = layers.Add(name='block3b_add')([x, x_skip])

    # MỚI: Block 4a
    # Giảm kích thước không gian (strides=2) và tăng độ sâu của feature map lên 96.
    x = layers.Conv2D(288, 3, strides=2, padding='same', use_bias=True, activation=H_SWISH_ACTIVATION, name='block4a_expand')(x)
    x = layers.Conv2D(96, 1, padding='same', use_bias=True, name='block4a_project')(x)

    # MỚI: Block 4b (Residual)
    # Tăng cường học đặc trưng ở độ sâu 96.
    x_skip = x
    x = layers.Conv2D(576, 3, padding='same', use_bias=True, activation=H_SWISH_ACTIVATION, name='block4b_expand')(x)
    x = layers.Conv2D(96, 1, padding='same', use_bias=True, name='block4b_project')(x)
    x = layers.Dropout(0.0, name='block4b_drop')(x)
    x = layers.Add(name='block4b_add')([x, x_skip])

    # MỚI: Block 4c (Residual)
    # Tiếp tục học sâu hơn ở cùng độ sâu 96.
    x_skip = x
    x = layers.Conv2D(576, 3, padding='same', use_bias=True, activation=H_SWISH_ACTIVATION, name='block4c_expand')(x)
    x = layers.Conv2D(96, 1, padding='same', use_bias=True, name='block4c_project')(x)
    x = layers.Dropout(0.0, name='block4c_drop')(x)
    x = layers.Add(name='block4c_add')([x, x_skip])

    # Classification head
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = layers.Dense(128, activation=H_SWISH_ACTIVATION, name='dense_128')(x)
    x = layers.Dropout(0.5, name='dropout_final')(x)
    outputs = layers.Dense(10, activation='softmax', name='predictions')(x)

    return models.Model(inputs=inputs, outputs=outputs, name='EffNet_Custom_hswish_CIFAR10_v2')

# ==============================================================================
# 2. TẢI VÀ TIỀN XỬ LÝ DỮ LIỆU CIFAR-10
# ==============================================================================
print("Đang tải dữ liệu CIFAR-10...")
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

def preprocess_data(images, labels):
    images = tf.image.resize(images, (224, 224))
    return images, labels

# Sử dụng tf.data để tối ưu pipeline
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

BATCH_SIZE = 32

train_dataset = train_dataset.map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)\
                             .shuffle(buffer_size=1024)\
                             .batch(BATCH_SIZE)\
                             .prefetch(buffer_size=tf.data.AUTOTUNE)

test_dataset = test_dataset.map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)\
                            .batch(BATCH_SIZE)\
                            .prefetch(buffer_size=tf.data.AUTOTUNE)

# ==============================================================================
# 3. XÂY DỰNG VÀ BIÊN DỊCH MÔ HÌNH
# ==============================================================================
print("Đang xây dựng mô hình với h-swish...")
model = build_model()
model.summary()

print("Biên dịch mô hình...")
model.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ==============================================================================
# 4. HUẤN LUYỆN MÔ HÌNH
# ==============================================================================
print("Bắt đầu huấn luyện...")
EPOCHS = 15 # Lưu ý: Mô hình sâu hơn có thể cần nhiều epochs hơn để hội tụ
history = model.fit(train_dataset,
                    epochs=EPOCHS,
                    validation_data=test_dataset)

# ==============================================================================
# 5. ĐÁNH GIÁ MÔ HÌNH
# ==============================================================================
print("\nĐánh giá mô hình trên tập test...")
test_loss, test_acc = model.evaluate(test_dataset)
print(f"\nKết quả đánh giá trên tập test: Loss = {test_loss:.4f}, Accuracy = {test_acc:.4f}")

# ==============================================================================
# 6. LƯU MÔ HÌNH VÀ VẼ BIỂU ĐỒ
# ==============================================================================
# CẬP NHẬT: Hàm plot_history giờ sẽ lưu biểu đồ vào thư mục
def plot_history(history, save_dir='plots'):
    """Vẽ và lưu biểu đồ lịch sử huấn luyện."""
    # Tạo thư mục nếu nó chưa tồn tại
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Đã tạo thư mục '{save_dir}'")
        
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, 'bo-', label='Training Accuracy')
    plt.plot(epochs_range, val_acc, 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, 'bo-', label='Training Loss')
    plt.plot(epochs_range, val_loss, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.suptitle('Model Training History (with h-swish)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Lưu biểu đồ vào file
    file_name = 'training_history.png'
    save_path = os.path.join(save_dir, file_name)
    plt.savefig(save_path)
    print(f"✅ Đã lưu biểu đồ vào '{save_path}'")
    
    # Hiển thị biểu đồ
    plt.show()

print("Đang lưu mô hình...")
model_path = 'cifar10_custom_hswish_trained_v2.keras' # Đổi tên file để không ghi đè
model.save(model_path)
print(f"✅ Đã lưu mô hình thành công tại '{model_path}'")

# Vẽ và lưu biểu đồ lịch sử huấn luyện
plot_history(history)