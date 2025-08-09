import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os

# Tắt các thông báo log không quan trọng
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==============================================================================
# PHẦN 1: ĐỊNH NGHĨA MODEL VÀ CÁC HÀM HỖ TRỢ
# ==============================================================================

def build_efficientnetv2_b0_full():
    """
    Xây dựng model EfficientNetV2-B0 với các lớp Batch Normalization đã được loại bỏ.
    Kiến trúc này khớp với các trọng số đã được gộp (fused) mà chúng ta đã xuất ra.
    """
    inputs = tf.keras.layers.Input(shape=(224, 224, 3), name='input_layer')

    # Tiền xử lý
    x = tf.keras.layers.Rescaling(1./255, name='rescaling')(inputs)
    normalizer = tf.keras.layers.Normalization(
        axis=-1,
        mean=[0.485, 0.456, 0.406],
        variance=[0.229**2, 0.224**2, 0.225**2],
        name='normalization'
    )
    x = normalizer(x)

    # Stem
    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding='same', use_bias=True, name='stem_conv')(x)
    x = tf.keras.layers.Activation('swish', name='stem_activation')(x)

    # Block 1
    x = tf.keras.layers.Conv2D(16, 3, padding='same', use_bias=True, name='block1a_project_conv')(x)
    x = tf.keras.layers.Activation('swish', name='block1a_project_activation')(x)

    # Block 2
    x = tf.keras.layers.Conv2D(64, 3, strides=2, padding='same', use_bias=True, name='block2a_expand_conv')(x)
    x = tf.keras.layers.Activation('swish', name='block2a_expand_activation')(x)
    x = tf.keras.layers.Conv2D(32, 1, padding='same', use_bias=True, name='block2a_project_conv')(x)
    x_skip = x
    x = tf.keras.layers.Conv2D(128, 3, padding='same', use_bias=True, name='block2b_expand_conv')(x)
    x = tf.keras.layers.Activation('swish', name='block2b_expand_activation')(x)
    x = tf.keras.layers.Conv2D(32, 1, padding='same', use_bias=True, name='block2b_project_conv')(x)
    x = tf.keras.layers.Dropout(0.0, name='block2b_drop')(x)
    x = tf.keras.layers.Add(name='block2b_add')([x, x_skip])

    # Block 3
    x = tf.keras.layers.Conv2D(128, 3, strides=2, padding='same', use_bias=True, name='block3a_expand_conv')(x)
    x = tf.keras.layers.Activation('swish', name='block3a_expand_activation')(x)
    x = tf.keras.layers.Conv2D(48, 1, padding='same', use_bias=True, name='block3a_project_conv')(x)
    x_skip = x
    x = tf.keras.layers.Conv2D(192, 3, padding='same', use_bias=True, name='block3b_expand_conv')(x)
    x = tf.keras.layers.Activation('swish', name='block3b_expand_activation')(x)
    x = tf.keras.layers.Conv2D(48, 1, padding='same', use_bias=True, name='block3b_project_conv')(x)
    x = tf.keras.layers.Dropout(0.0, name='block3b_drop')(x)
    x = tf.keras.layers.Add(name='block3b_add')([x, x_skip])

    # Block 4
    x = tf.keras.layers.Conv2D(192, 1, padding='same', use_bias=True, name='block4a_expand_conv')(x)
    x = tf.keras.layers.Activation('swish', name='block4a_expand_activation')(x)
    x = tf.keras.layers.DepthwiseConv2D(3, strides=2, padding='same', use_bias=True, name='block4a_dwconv2')(x)
    x = tf.keras.layers.Activation('swish', name='block4a_activation')(x)
    se = tf.keras.layers.GlobalAveragePooling2D(name='block4a_se_squeeze')(x)
    se = tf.keras.layers.Reshape((1, 1, 192), name='block4a_se_reshape')(se)
    se = tf.keras.layers.Conv2D(12, 1, activation='swish', name='block4a_se_reduce')(se)
    se = tf.keras.layers.Conv2D(192, 1, activation='sigmoid', name='block4a_se_expand')(se)
    x = tf.keras.layers.Multiply(name='block4a_se_excite')([x, se])
    x = tf.keras.layers.Conv2D(96, 1, padding='same', use_bias=True, name='block4a_project_conv')(x)
    x_skip = x
    x = tf.keras.layers.Conv2D(384, 1, padding='same', use_bias=True, name='block4b_expand_conv')(x)
    x = tf.keras.layers.Activation('swish', name='block4b_expand_activation')(x)
    x = tf.keras.layers.DepthwiseConv2D(3, strides=1, padding='same', use_bias=True, name='block4b_dwconv2')(x)
    x = tf.keras.layers.Activation('swish', name='block4b_activation')(x)
    se = tf.keras.layers.GlobalAveragePooling2D(name='block4b_se_squeeze')(x)
    se = tf.keras.layers.Reshape((1, 1, 384), name='block4b_se_reshape')(se)
    se = tf.keras.layers.Conv2D(24, 1, activation='swish', name='block4b_se_reduce')(se)
    se = tf.keras.layers.Conv2D(384, 1, activation='sigmoid', name='block4b_se_expand')(se)
    x = tf.keras.layers.Multiply(name='block4b_se_excite')([x, se])
    x = tf.keras.layers.Conv2D(96, 1, padding='same', use_bias=True, name='block4b_project_conv')(x)
    x = tf.keras.layers.Dropout(0.0, name='block4b_drop')(x)
    x = tf.keras.layers.Add(name='block4b_add')([x, x_skip])
    x_skip = x
    x = tf.keras.layers.Conv2D(384, 1, padding='same', use_bias=True, name='block4c_expand_conv')(x)
    x = tf.keras.layers.Activation('swish', name='block4c_expand_activation')(x)
    x = tf.keras.layers.DepthwiseConv2D(3, strides=1, padding='same', use_bias=True, name='block4c_dwconv2')(x)
    x = tf.keras.layers.Activation('swish', name='block4c_activation')(x)
    se = tf.keras.layers.GlobalAveragePooling2D(name='block4c_se_squeeze')(x)
    se = tf.keras.layers.Reshape((1, 1, 384), name='block4c_se_reshape')(se)
    se = tf.keras.layers.Conv2D(24, 1, activation='swish', name='block4c_se_reduce')(se)
    se = tf.keras.layers.Conv2D(384, 1, activation='sigmoid', name='block4c_se_expand')(se)
    x = tf.keras.layers.Multiply(name='block4c_se_excite')([x, se])
    x = tf.keras.layers.Conv2D(96, 1, padding='same', use_bias=True, name='block4c_project_conv')(x)
    x = tf.keras.layers.Dropout(0.0, name='block4c_drop')(x)
    x = tf.keras.layers.Add(name='block4c_add')([x, x_skip])

    # Block 5
    x = tf.keras.layers.Conv2D(576, 1, padding='same', use_bias=True, name='block5a_expand_conv')(x)
    x = tf.keras.layers.Activation('swish', name='block5a_expand_activation')(x)
    x = tf.keras.layers.DepthwiseConv2D(3, strides=1, padding='same', use_bias=True, name='block5a_dwconv2')(x)
    x = tf.keras.layers.Activation('swish', name='block5a_activation')(x)
    se = tf.keras.layers.GlobalAveragePooling2D(name='block5a_se_squeeze')(x)
    se = tf.keras.layers.Reshape((1, 1, 576), name='block5a_se_reshape')(se)
    se = tf.keras.layers.Conv2D(24, 1, activation='swish', name='block5a_se_reduce')(se)
    se = tf.keras.layers.Conv2D(576, 1, activation='sigmoid', name='block5a_se_expand')(se)
    x = tf.keras.layers.Multiply(name='block5a_se_excite')([x, se])
    x = tf.keras.layers.Conv2D(112, 1, padding='same', use_bias=True, name='block5a_project_conv')(x)
    x_skip = x
    x = tf.keras.layers.Conv2D(672, 1, padding='same', use_bias=True, name='block5b_expand_conv')(x)
    x = tf.keras.layers.Activation('swish', name='block5b_expand_activation')(x)
    x = tf.keras.layers.DepthwiseConv2D(3, strides=1, padding='same', use_bias=True, name='block5b_dwconv2')(x)
    x = tf.keras.layers.Activation('swish', name='block5b_activation')(x)
    se = tf.keras.layers.GlobalAveragePooling2D(name='block5b_se_squeeze')(x)
    se = tf.keras.layers.Reshape((1, 1, 672), name='block5b_se_reshape')(se)
    se = tf.keras.layers.Conv2D(28, 1, activation='swish', name='block5b_se_reduce')(se)
    se = tf.keras.layers.Conv2D(672, 1, activation='sigmoid', name='block5b_se_expand')(se)
    x = tf.keras.layers.Multiply(name='block5b_se_excite')([x, se])
    x = tf.keras.layers.Conv2D(112, 1, padding='same', use_bias=True, name='block5b_project_conv')(x)
    x = tf.keras.layers.Dropout(0.0, name='block5b_drop')(x)
    x = tf.keras.layers.Add(name='block5b_add')([x, x_skip])
    x_skip = x
    x = tf.keras.layers.Conv2D(672, 1, padding='same', use_bias=True, name='block5c_expand_conv')(x)
    x = tf.keras.layers.Activation('swish', name='block5c_expand_activation')(x)
    x = tf.keras.layers.DepthwiseConv2D(3, strides=1, padding='same', use_bias=True, name='block5c_dwconv2')(x)
    x = tf.keras.layers.Activation('swish', name='block5c_activation')(x)
    se = tf.keras.layers.GlobalAveragePooling2D(name='block5c_se_squeeze')(x)
    se = tf.keras.layers.Reshape((1, 1, 672), name='block5c_se_reshape')(se)
    se = tf.keras.layers.Conv2D(28, 1, activation='swish', name='block5c_se_reduce')(se)
    se = tf.keras.layers.Conv2D(672, 1, activation='sigmoid', name='block5c_se_expand')(se)
    x = tf.keras.layers.Multiply(name='block5c_se_excite')([x, se])
    x = tf.keras.layers.Conv2D(112, 1, padding='same', use_bias=True, name='block5c_project_conv')(x)
    x = tf.keras.layers.Dropout(0.0, name='block5c_drop')(x)
    x = tf.keras.layers.Add(name='block5c_add')([x, x_skip])

    # Block 6
    x = tf.keras.layers.Conv2D(672, 1, padding='same', use_bias=True, name='block6a_expand_conv')(x)
    x = tf.keras.layers.Activation('swish', name='block6a_expand_activation')(x)
    x = tf.keras.layers.DepthwiseConv2D(3, strides=2, padding='same', use_bias=True, name='block6a_dwconv2')(x)
    x = tf.keras.layers.Activation('swish', name='block6a_activation')(x)
    se = tf.keras.layers.GlobalAveragePooling2D(name='block6a_se_squeeze')(x)
    se = tf.keras.layers.Reshape((1, 1, 672), name='block6a_se_reshape')(se)
    se = tf.keras.layers.Conv2D(28, 1, activation='swish', name='block6a_se_reduce')(se)
    se = tf.keras.layers.Conv2D(672, 1, activation='sigmoid', name='block6a_se_expand')(se)
    x = tf.keras.layers.Multiply(name='block6a_se_excite')([x, se])
    x = tf.keras.layers.Conv2D(192, 1, padding='same', use_bias=True, name='block6a_project_conv')(x)
    x_skip = x
    x = tf.keras.layers.Conv2D(1152, 1, padding='same', use_bias=True, name='block6b_expand_conv')(x)
    x = tf.keras.layers.Activation('swish', name='block6b_expand_activation')(x)
    x = tf.keras.layers.DepthwiseConv2D(3, strides=1, padding='same', use_bias=True, name='block6b_dwconv2')(x)
    x = tf.keras.layers.Activation('swish', name='block6b_activation')(x)
    se = tf.keras.layers.GlobalAveragePooling2D(name='block6b_se_squeeze')(x)
    se = tf.keras.layers.Reshape((1, 1, 1152), name='block6b_se_reshape')(se)
    se = tf.keras.layers.Conv2D(48, 1, activation='swish', name='block6b_se_reduce')(se)
    se = tf.keras.layers.Conv2D(1152, 1, activation='sigmoid', name='block6b_se_expand')(se)
    x = tf.keras.layers.Multiply(name='block6b_se_excite')([x, se])
    x = tf.keras.layers.Conv2D(192, 1, padding='same', use_bias=True, name='block6b_project_conv')(x)
    x = tf.keras.layers.Dropout(0.0, name='block6b_drop')(x)
    x = tf.keras.layers.Add(name='block6b_add')([x, x_skip])
    x_skip = x
    x = tf.keras.layers.Conv2D(1152, 1, padding='same', use_bias=True, name='block6c_expand_conv')(x)
    x = tf.keras.layers.Activation('swish', name='block6c_expand_activation')(x)
    x = tf.keras.layers.DepthwiseConv2D(3, strides=1, padding='same', use_bias=True, name='block6c_dwconv2')(x)
    x = tf.keras.layers.Activation('swish', name='block6c_activation')(x)
    se = tf.keras.layers.GlobalAveragePooling2D(name='block6c_se_squeeze')(x)
    se = tf.keras.layers.Reshape((1, 1, 1152), name='block6c_se_reshape')(se)
    se = tf.keras.layers.Conv2D(48, 1, activation='swish', name='block6c_se_reduce')(se)
    se = tf.keras.layers.Conv2D(1152, 1, activation='sigmoid', name='block6c_se_expand')(se)
    x = tf.keras.layers.Multiply(name='block6c_se_excite')([x, se])
    x = tf.keras.layers.Conv2D(192, 1, padding='same', use_bias=True, name='block6c_project_conv')(x)
    x = tf.keras.layers.Dropout(0.0, name='block6c_drop')(x)
    x = tf.keras.layers.Add(name='block6c_add')([x, x_skip])
    x_skip = x
    x = tf.keras.layers.Conv2D(1152, 1, padding='same', use_bias=True, name='block6d_expand_conv')(x)
    x = tf.keras.layers.Activation('swish', name='block6d_expand_activation')(x)
    x = tf.keras.layers.DepthwiseConv2D(3, strides=1, padding='same', use_bias=True, name='block6d_dwconv2')(x)
    x = tf.keras.layers.Activation('swish', name='block6d_activation')(x)
    se = tf.keras.layers.GlobalAveragePooling2D(name='block6d_se_squeeze')(x)
    se = tf.keras.layers.Reshape((1, 1, 1152), name='block6d_se_reshape')(se)
    se = tf.keras.layers.Conv2D(48, 1, activation='swish', name='block6d_se_reduce')(se)
    se = tf.keras.layers.Conv2D(1152, 1, activation='sigmoid', name='block6d_se_expand')(se)
    x = tf.keras.layers.Multiply(name='block6d_se_excite')([x, se])
    x = tf.keras.layers.Conv2D(192, 1, padding='same', use_bias=True, name='block6d_project_conv')(x)
    x = tf.keras.layers.Dropout(0.0, name='block6d_drop')(x)
    x = tf.keras.layers.Add(name='block6d_add')([x, x_skip])

    # Head
    x = tf.keras.layers.Conv2D(1280, 1, padding='same', use_bias=True, name='top_conv')(x)
    x = tf.keras.layers.Activation('swish', name='top_activation')(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = tf.keras.layers.Dropout(0.2, name='top_dropout')(x)
    outputs = tf.keras.layers.Dense(1000, activation='softmax', name='predictions')(x)

    return tf.keras.models.Model(inputs, outputs, name='efficientnetv2_b0_fused')


def load_weights_from_file(layer, base_weights_dir):
    """
    Đọc file txt chứa trọng số và bias đã lượng tử hóa,
    de-quantize chúng và trả về dưới dạng numpy array.
    Hàm này dùng chung cho cả Conv2D, DepthwiseConv2D và Dense.
    """
    layer_name = layer.name
    layer_dir = os.path.join(base_weights_dir, layer_name)

    # --- Đọc và de-quantize Kernel (Trọng số) ---
    with open(os.path.join(layer_dir, f"{layer_name}_kernel_parameters.txt"), 'r') as f:
        kernel_scale = float(f.readline().split(':')[1].strip())
    with open(os.path.join(layer_dir, f"{layer_name}_kernel_quantized_int8.txt"), 'r') as f:
        kernel_int = np.array([int(line.strip()) for line in f.readlines()], dtype=np.int8)
    
    kernel_float = kernel_int.astype(np.float32) * kernel_scale
    kernel_float = kernel_float.reshape(layer.get_weights()[0].shape)

    # --- Đọc và de-quantize Bias ---
    with open(os.path.join(layer_dir, f"{layer_name}_bias_parameters.txt"), 'r') as f:
        bias_scale = float(f.readline().split(':')[1].strip())
    with open(os.path.join(layer_dir, f"{layer_name}_bias_quantized_int32.txt"), 'r') as f:
        bias_int = np.array([int(line.strip()) for line in f.readlines()], dtype=np.int32)

    bias_float = bias_int.astype(np.float32) * bias_scale

    return [kernel_float, bias_float]

def load_all_weights(model, base_weights_dir):
    """
    Lặp qua tất cả các lớp của model và nạp trọng số từ file.
    """
    print(f"Bắt đầu nạp trọng số từ thư mục: '{base_weights_dir}'")
    loaded_layers_count = 0
    for layer in model.layers:
        # Chỉ nạp trọng số cho các lớp có file đã xuất
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D, tf.keras.layers.Dense)):
            try:
                weights = load_weights_from_file(layer, base_weights_dir)
                layer.set_weights(weights)
                print(f"  ✅ Đã nạp trọng số cho lớp: {layer.name}")
                loaded_layers_count += 1
            except FileNotFoundError:
                print(f"  ⚠️ Cảnh báo: Không tìm thấy file trọng số cho lớp '{layer.name}'. Bỏ qua.")
            
    print(f"\n🎉 Hoàn tất nạp trọng số cho {loaded_layers_count} lớp.")
    if loaded_layers_count == 0:
        print("🛑 CẢNH BÁO: Không có trọng số nào được nạp. Model đang sử dụng trọng số ngẫu nhiên.")
        print(f"   Hãy chắc chắn rằng thư mục '{base_weights_dir}' tồn tại và có chứa các file trọng số.")


# ==============================================================================
# PHẦN 2: KHỞI TẠO MODEL VÀ TẢI TRỌNG SỐ
# ==============================================================================
print("--- PHẦN 2: KHỞI TẠO MODEL VÀ TẢI TRỌNG SỐ ---")
model = build_efficientnetv2_b0_full()

base_weights_dir = "quantized_model_weights_and_biases_per_layer"

load_all_weights(model, base_weights_dir)


# ==============================================================================
# PHẦN 3: TẢI DỮ LIỆU VÀ CHẠY THỬ NGHIỆM
# ==============================================================================
print("\n--- PHẦN 3: TẢI DỮ LIỆU VÀ CHẠY DỰ ĐOÁN ---")

try:
    ds, info = tfds.load("imagenet_v2", split='test', with_info=True, as_supervised=True)
    test_dataset = ds.take(100)
    print("✅ Tải dữ liệu ImageNetV2 thành công.")
    
    labels = info.features['label'].names
    
    num_images_to_test = 100
    correct_predictions = 0
    image_count = 0
    
    for image, label_index in test_dataset:
        image_count += 1
        image_resized = tf.image.resize(image, (224, 224))
        image_batch = tf.expand_dims(image_resized, axis=0)
        
        predictions = model.predict(image_batch, verbose=0)
        
        predicted_index = np.argmax(predictions[0])
        predicted_label_name = labels[predicted_index]
        true_label_name = labels[label_index]
        
        is_correct = (predicted_index == label_index.numpy())
        if is_correct:
            correct_predictions += 1
            
        print(f"Ảnh #{image_count}: Dự đoán: '{predicted_label_name}', Nhãn thật: '{true_label_name}' -> {'ĐÚNG' if is_correct else 'SAI'}")

    # ==============================================================================
    # PHẦN 4: ĐÁNH GIÁ KẾT QUẢ
    # ==============================================================================
    print("\n--- PHẦN 4: ĐÁNH GIÁ ---")
    accuracy = (correct_predictions / num_images_to_test) * 100
    print(f"Tổng kết trên {num_images_to_test} ảnh:")
    print(f"Số dự đoán đúng: {correct_predictions}")
    print(f"Độ chính xác (Top-1 Accuracy): {accuracy:.2f}%")

except Exception as e:
    print(f"\n❌ Đã xảy ra lỗi trong quá trình tải dữ liệu hoặc chạy thử nghiệm: {e}")
    print("Vui lòng kiểm tra kết nối mạng và đảm bảo tensorflow_datasets đã được cài đặt (`pip install tensorflow_datasets`).")
