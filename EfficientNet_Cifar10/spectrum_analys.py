# Tên file: analyze_value_distributions.py
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_error

# --- CẤU HÌNH ---
LAYER_TO_ANALYZE = 'block3b_project'
LOG_DIR = 'inference_logs_detailed' # Đảm bảo tên thư mục khớp
QUANT_LOG_DIR = os.path.join(LOG_DIR, 'quantized')
FLOAT_LOG_DIR = os.path.join(LOG_DIR, 'float32')

# --- HÀM PHÂN TÍCH ---
def analyze_and_plot_distributions(tensor_type, layer_name):
    float_file = os.path.join(FLOAT_LOG_DIR, f"{layer_name}_{tensor_type}.txt")
    quant_file = os.path.join(QUANT_LOG_DIR, f"{layer_name}_{tensor_type}.txt")

    print(f"\n===== Phân tích: {tensor_type.upper()} của lớp {layer_name} =====")

    if not os.path.exists(float_file) or not os.path.exists(quant_file):
        print(f"LỖI: Không tìm thấy file log cho '{tensor_type}'. Bỏ qua.")
        return

    data_float = np.loadtxt(float_file)
    data_quant = np.loadtxt(quant_file)

    print("--- Thống kê dữ liệu Float32 (Gốc) ---")
    print(f"  - Min: {np.min(data_float):.6f} | Max: {np.max(data_float):.6f} | Mean: {np.mean(data_float):.6f} | StdDev: {np.std(data_float):.6f}")

    print("--- Thống kê dữ liệu Quantized (Mô phỏng) ---")
    print(f"  - Min: {np.min(data_quant):.6f} | Max: {np.max(data_quant):.6f} | Mean: {np.mean(data_quant):.6f} | StdDev: {np.std(data_quant):.6f}")

    mae = mean_absolute_error(data_float, data_quant)
    print(f"--- Sai số ---")
    print(f"  - Mean Absolute Error (MAE): {mae:.8f}")

    plt.figure(figsize=(14, 7))
    combined_min = min(np.min(data_float), np.min(data_quant))
    combined_max = max(np.max(data_float), np.max(data_quant))
    bins = np.linspace(combined_min, combined_max, 100)

    plt.hist(data_float, bins=bins, color='dodgerblue', alpha=0.6, label='Float32 (Gốc)', density=True)
    plt.hist(data_quant, bins=bins, color='red', alpha=0.9, label='Quantized (Mô phỏng)', density=True, histtype='step', linewidth=2)

    plt.title(f"So Sánh Phân Phối - {tensor_type.capitalize()} của Lớp '{layer_name}'", fontsize=16)
    plt.xlabel("Giá trị của Tensor")
    plt.ylabel("Mật độ (Density)")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

# --- THỰC THI ---
if __name__ == "__main__":
    if not os.path.exists(LOG_DIR):
        print(f"LỖI: Thư mục log '{LOG_DIR}' không tồn tại.")
    else:
        print(f"Phân tích cho lớp: '{LAYER_TO_ANALYZE}'")
        analyze_and_plot_distributions('input', LAYER_TO_ANALYZE)
        analyze_and_plot_distributions('weight', LAYER_TO_ANALYZE)
        analyze_and_plot_distributions('bias', LAYER_TO_ANALYZE)
        analyze_and_plot_distributions('output', LAYER_TO_ANALYZE)