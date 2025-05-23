# %%
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
# %%
# 1. بارگذاری مدل آموزش‌دیده
model = load_model("autoencoder_final_gpu.h5", compile=False)
# %%
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

# 6. نمایش تصاویر: اصلی، نویزدار و بازسازی شده
plt.figure(figsize=(15, 7))  # افزایش ارتفاع برای سه ردیف

for i in range(len(original_images)):
    # تصویر اصلی
    ax = plt.subplot(3, len(original_images), i + 1)
    plt.imshow(original_images[i], cmap="gray")
    plt.title("تصویر اصلی")
    plt.axis("off")

    # تصویر نویزدار (ورودی به مدل)
    ax = plt.subplot(3, len(original_images), i + len(original_images) + 1)
    plt.imshow(noisy_images[i].squeeze(), cmap="gray")
    plt.title("نویزدار ورودی")
    plt.axis("off")

    # تصویر بازسازی شده
    ax = plt.subplot(3, len(original_images), i + 2 * len(original_images) + 1)
    plt.imshow(reconstructed_images[i].squeeze(), cmap="gray")
    plt.title("بازسازی شده")
    plt.axis("off")

plt.tight_layout()
plt.show()
# %%
