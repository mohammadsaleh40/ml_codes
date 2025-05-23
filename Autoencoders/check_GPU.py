import tensorflow as tf

# نمایش نسخه TensorFlow
print(f"TensorFlow Version: {tf.__version__}")

# بررسی وجود GPU
gpu_devices = tf.config.list_physical_devices('GPU')

if len(gpu_devices) > 0:
    print("\nGPU تشخیص داده شد! ✅")
    print("دستگاه‌های GPU موجود:")
    for device in gpu_devices:
        print(f"- {device}")
else:
    print("\nGPU تشخیص داده نشد! ❌")
    print("توجه: مدل ممکن است روی CPU اجرا شود")

# اطلاعات تکمیلی (اختیاری)
print("\nاطلاعات تکمیلی:")
print(tf.sysconfig.get_build_info())