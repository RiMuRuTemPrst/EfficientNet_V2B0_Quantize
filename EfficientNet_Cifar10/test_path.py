import os

# Đường dẫn file y hệt như trong thông báo lỗi của bạn
file_path = 'cifar10_custom_hswish_trained.keras'

print(f"Đang kiểm tra đường dẫn: '{file_path}'")

if os.path.exists(file_path):
    print(">>> KẾT QUẢ: ✅ THÀNH CÔNG! Python cơ bản có thể thấy file.")
else:
    print(">>> KẾT QUẢ: ❌ THẤT BẠI! Python cơ bản không thấy file này.")