import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 1. بارگذاری مدل آموزش‌دیده
model = load_model("autoencoder_final_gpu.h5", compile=False)

# 2. تابع بارگذاری چند تصویر نمونه از پوشه data/
def load_sample_images(num_samples=5):
    image_files = [f for f in os.listdir("data") if f.endswith('.png')][:num_samples]
    images = []
    for file in image_files:
        img = cv2.imread(os.path.join("data", file), cv2.IMREAD_GRAYSCALE)
        images.append(img)
    return np.array(images)

# 3. تابع اضافه کردن نویز (با استفاده از NumPy)
def add_noise_np(images, noise_factor=0.2):
    noise = np.random.normal(size=images.shape, scale=1.0)
    return np.clip(images + noise_factor * noise, 0., 1.)

# 4. بارگذاری و پیش‌پردازش تصاویر
original_images = load_sample_images(num_samples=5)

# نرمالایز و اضافه کردن بعد کانال
original_normalized = original_images.astype('float32') / 255.0
original_normalized = np.expand_dims(original_normalized, -1)  # شکل: (batch_size, 64, 64, 1)

# اضافه کردن نویز به تصاویر
noisy_images = add_noise_np(original_normalized)

# 5. پیش‌بینی (بازسازی تصویر با استفاده از تصاویر نویزدار)
reconstructed_images = model.predict(noisy_images)

# 6. نمایش تصاویر با OpenCV
for i in range(len(original_images)):
    # تصویر اصلی (0-255)
    original = original_images[i]

    # تصویر نویزدار (0-1 → 0-255)
    noisy = (noisy_images[i].squeeze() * 255).astype(np.uint8)

    # تصویر بازسازی شده (0-1 → 0-255)
    reconstructed = (reconstructed_images[i].squeeze() * 255).astype(np.uint8)

    # ترکیب تصاویر در یک خط (Horizontal Stack)
    combined = np.hstack([original, noisy, reconstructed])

    # نمایش در پنجره OpenCV
    cv2.imshow(f"Sample {i+1} - Original | Noisy | Reconstructed", combined)

    # انتظار برای فشردن کلید (برای بستن پنجره)
    cv2.waitKey(0)
    cv2.destroyAllWindows()