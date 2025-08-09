import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_weights_from_file_per_channel(layer, base_weights_dir):
    """Đọc file trọng số được lượng tử hóa theo từng kênh."""
    layer_name = layer.name
    layer_dir = os.path.join(base_weights_dir, layer_name)
    try:
        # Đọc mảng Kernel scales
        kernel_params_path = os.path.join(layer_dir, "kernel_parameters.txt")
        kernel_scales = np.loadtxt(kernel_params_path, dtype=np.float32)

        # Đọc kernel int8
        kernel_hex_path = os.path.join(layer_dir, "kernel_quantized_int8.hex")
        with open(kernel_hex_path, 'r') as f:
            kernel_hex_vals = [int(line.strip(), 16) for line in f.readlines()]
        kernel_int = np.array(kernel_hex_vals, dtype=np.uint8).view(np.int8)
        
        kernel_int_reshaped = kernel_int.reshape(layer.get_weights()[0].shape)
        scale_reshape_shape = [1] * kernel_int_reshaped.ndim
        scale_reshape_shape[-1] = -1
        kernel_scales_reshaped = kernel_scales.reshape(scale_reshape_shape)
        kernel_float = kernel_int_reshaped.astype(np.float32) * kernel_scales_reshaped

        # Đọc mảng Bias scales
        bias_params_path = os.path.join(layer_dir, "bias_parameters.txt")
        bias_scales = np.loadtxt(bias_params_path, dtype=np.float32)

        # Đọc bias int8
        bias_hex_path = os.path.join(layer_dir, "bias_quantized_int8.hex")
        with open(bias_hex_path, 'r') as f:
            bias_hex_vals = [int(line.strip(), 16) for line in f.readlines()]
        bias_int = np.array(bias_hex_vals, dtype=np.uint8).view(np.int8)
        
        bias_float = bias_int.astype(np.float32) * bias_scales
        
        return [kernel_float, bias_float]
    except Exception as e:
        print(f"  ❌ Lỗi khi nạp trọng số cho lớp '{layer_name}': {e}")
        raise

print("\n--- KIỂM TRA (PER-CHANNEL) ĐẾN HẾT BLOCK 2A ---")
IMAGE_SIZE = (224, 224)

original_full_model = tf.keras.applications.EfficientNetV2B0(weights='imagenet', input_shape=IMAGE_SIZE + (3,))
original_model = tf.keras.models.Model(
    inputs=original_full_model.input, outputs=original_full_model.get_layer('block2a_project_bn').output)

custom_inputs = tf.keras.layers.Input(shape=(224, 224, 3))
x = tf.keras.layers.Rescaling(1./255)(custom_inputs)
x = tf.keras.layers.Normalization(axis=-1, mean=[0.485, 0.456, 0.406], variance=[0.229**2, 0.224**2, 0.225**2])(x)
x = tf.keras.layers.Conv2D(32, 3, strides=2, padding='same', use_bias=True, name='stem_conv')(x)
x = tf.keras.layers.Activation('swish', name='stem_activation')(x)
x = tf.keras.layers.Conv2D(16, 3, padding='same', use_bias=True, name='block1a_project_conv')(x)
x = tf.keras.layers.Activation('swish', name='block1a_project_activation')(x)
x = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1)))(x)
x = tf.keras.layers.Conv2D(64, 3, strides=2, padding='valid', use_bias=True, name='block2a_expand_conv')(x)
x = tf.keras.layers.Activation('swish', name='block2a_expand_activation')(x)
x = tf.keras.layers.Conv2D(32, 1, padding='same', use_bias=True, name='block2a_project_conv')(x)
custom_model = tf.keras.models.Model(inputs=custom_inputs, outputs=x)
_ = custom_model.predict(np.zeros((1, 224, 224, 3)), verbose=0)

base_weights_dir = "efficientnetv2b0_quantized_weights_float_scale"
print(f"\nNạp trọng số từ '{base_weights_dir}'...")
try:
    layers_to_load = [
        custom_model.get_layer('stem_conv'), custom_model.get_layer('block1a_project_conv'),
        custom_model.get_layer('block2a_expand_conv'), custom_model.get_layer('block2a_project_conv')
    ]
    for layer_obj in layers_to_load:
        weights = load_weights_from_file_per_channel(layer_obj, base_weights_dir)
        layer_obj.set_weights(weights)
    print("✅ Đã nạp đầy đủ trọng số.")
except Exception as e:
    print(f"❌ Lỗi khi nạp trọng số: {e}")
    exit()

# ---- PHẦN BỊ THIẾU: SUY LUẬN VÀ SO SÁNH ---- #
print("\nBƯỚC 4: Chuẩn bị ảnh đầu vào...")
try:
    ds_single_image = tfds.load("imagenet_v2", split='test', as_supervised=True).take(1)
    for img, _ in ds_single_image:
        input_image_batch = tf.expand_dims(tf.image.resize(img, IMAGE_SIZE), axis=0)
    print("✅ Đã có ảnh đầu vào.")
except Exception as e:
    print(f"❌ Không thể tải ảnh: {e}.")
    exit()
    
print("\nBƯỚC 5: Chạy suy luận...")
output_golden = original_model.predict(input_image_batch, verbose=0)
output_custom = custom_model.predict(input_image_batch, verbose=0)
print("✅ Đã có kết quả từ cả hai model.")

print("\nBƯỚC 6: So sánh kết quả MSE...")
mse = np.mean(np.square(output_golden - output_custom))
print(f"\n=> Sai số bình phương trung bình (MSE): {mse:.8f}")

if mse < 1e-4: # Nới lỏng ngưỡng một chút cho sai số lượng tử hóa
    print("\n✅✅✅ THÀNH CÔNG! Sai số thấp ở mức chấp nhận được.")
else:
    print("\n❌❌❌ THẤT BẠI: Output vẫn có sự sai lệch lớn.")