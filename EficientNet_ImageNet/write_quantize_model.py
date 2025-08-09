import tensorflow as tf
import numpy as np
import os
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def quantize_symmetric_pow2_per_tensor(fp32_tensor):
    """Lượng tử hóa đối xứng trên toàn bộ tensor, scale dạng 2^n."""
    max_abs_val = np.max(np.abs(fp32_tensor))
    if max_abs_val == 0:
        return np.zeros_like(fp32_tensor, dtype=np.int8), 1.0, 0
    initial_scale = max_abs_val / 127.0
    power_of_2_scale = 2**round(math.log2(initial_scale)) if initial_scale > 0 else 1.0
    quantized_int8_tensor = np.round(fp32_tensor / power_of_2_scale).astype(np.int8)
    return quantized_int8_tensor, power_of_2_scale, 0

# THAY ĐỔI: Thêm các tham số mới để lưu file
def save_layer_data(base_dir, layer_name, 
                    int8_kernel, kernel_scale, 
                    int8_bias, bias_scale,
                    fused_kernel_fp32, dequantized_kernel_fp32,
                    fused_bias_fp32, dequantized_bias_fp32):
    """Lưu tất cả các loại dữ liệu của một lớp."""
    layer_dir = os.path.join(base_dir, layer_name)
    os.makedirs(layer_dir, exist_ok=True)
    
    # Lưu các file hex và tham số như cũ
    np.savetxt(os.path.join(layer_dir, 'kernel_quantized_int8.hex'), int8_kernel.flatten().view(np.uint8), fmt='%02x')
    np.savetxt(os.path.join(layer_dir, 'bias_quantized_int8.hex'), int8_bias.flatten().view(np.uint8), fmt='%02x')
    with open(os.path.join(layer_dir, 'kernel_parameters.txt'), 'w') as f:
        f.write(f"scale: {kernel_scale}\n")
        f.write("zero_point: 0\n")
    with open(os.path.join(layer_dir, 'bias_parameters.txt'), 'w') as f:
        f.write(f"scale: {bias_scale}\n")
        f.write("zero_point: 0\n")

    # MỚI: Lưu 2 file float32 cho kernel
    np.savetxt(os.path.join(layer_dir, 'kernel_fused_float32.txt'), fused_kernel_fp32.flatten(), fmt='%.8f')
    np.savetxt(os.path.join(layer_dir, 'kernel_dequantized_float32.txt'), dequantized_kernel_fp32.flatten(), fmt='%.8f')
    
    # MỚI: Lưu 2 file float32 cho bias
    np.savetxt(os.path.join(layer_dir, 'bias_fused_float32.txt'), fused_bias_fp32.flatten(), fmt='%.8f')
    np.savetxt(os.path.join(layer_dir, 'bias_dequantized_float32.txt'), dequantized_bias_fp32.flatten(), fmt='%.8f')
    
    print(f"  ✅ Đã lưu đầy đủ dữ liệu cho lớp '{layer_name}'")


def main():
    print("BƯỚC 1: Đang tải model...")
    model = tf.keras.applications.EfficientNetV2B0(include_top=False, weights='imagenet')
    print("✅ Tải model thành công.")
    
    output_base_dir = "efficientnetv2b0_quantized_weights_per_tensor_pow2"
    os.makedirs(output_base_dir, exist_ok=True)
    print(f"\nBƯỚC 2: Bắt đầu xử lý và lưu trọng số (Per-Tensor, scale 2^n) vào '{output_base_dir}'...")
    
    layers = model.layers
    for i in range(len(layers)):
        layer = layers[i]
        if not isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)) or len(layer.get_weights()) == 0:
            continue
        
        kernel, bias = (layer.get_weights()[0], layer.get_weights()[1]) if len(layer.get_weights()) > 1 else (layer.get_weights()[0], np.zeros(layer.get_weights()[0].shape[-1]))
        
        # Đây là trọng số "trước khi" quantize
        fused_kernel_fp32, fused_bias_fp32 = kernel, bias
        
        next_layer = layers[i + 1] if (i + 1) < len(layers) else None
        if next_layer and isinstance(next_layer, tf.keras.layers.BatchNormalization):
            bn_layer = next_layer
            gamma, beta, moving_mean, moving_variance = bn_layer.get_weights()
            epsilon = bn_layer.epsilon
            scale_bn = gamma / np.sqrt(moving_variance + epsilon)
            reshape_shape = (1, 1, -1, 1) if isinstance(layer, tf.keras.layers.DepthwiseConv2D) else (1, 1, 1, -1)
            fused_kernel_fp32 = kernel * scale_bn.reshape(reshape_shape)
            fused_bias_fp32 = beta + (bias - moving_mean) * scale_bn
            print(f"  - Đã gộp lớp '{bn_layer.name}' vào '{layer.name}'")

        # Lượng tử hóa
        quant_kernel, kernel_scale, _ = quantize_symmetric_pow2_per_tensor(fused_kernel_fp32)
        quant_bias, bias_scale, _ = quantize_symmetric_pow2_per_tensor(fused_bias_fp32)
        
        # MỚI: De-quantize ngay để có dữ liệu "sau khi" quantize
        dequantized_kernel_fp32 = quant_kernel.astype(np.float32) * kernel_scale
        dequantized_bias_fp32 = quant_bias.astype(np.float32) * bias_scale
        
        # Lưu tất cả các loại dữ liệu
        save_layer_data(
            output_base_dir, layer.name, 
            quant_kernel, kernel_scale, 
            quant_bias, bias_scale,
            fused_kernel_fp32, dequantized_kernel_fp32,
            fused_bias_fp32, dequantized_bias_fp32
        )
        
    print("\n🎉 Hoàn tất!")

if __name__ == '__main__':
    main()