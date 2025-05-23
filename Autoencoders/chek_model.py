
# %%
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf
# %%
# 1. بارگذاری مدل آموزش‌دیده
model = load_model("autoencoder_final_gpu.h5", compile=False)# %%
# 2. تابع بارگذاری چند تصویر نمونه از پوشه data/
def load_sample_images(num_samples=5):
    image_files = [f for f in os.listdir("data") if f.endswith('.png')][:num_samples]
    images = []
    for file in image_files:
        img = cv2.imread(os.path.join("data", file), cv2.IMREAD_GRAYSCALE)
        images.append(img)
    return np.array(images)

# 3. بارگذاری تصاویر و پیش‌پردازش
original_images = load_sample_images(num_samples=5)
noisy_images = original_images.astype('float32') / 255.0
noisy_images = np.expand_dims(noisy_images, -1)  # شکل: (batch_size, 64, 64, 1)

# 4. پیش‌بینی (بازسازی تصویر)
reconstructed_images = model.predict(noisy_images)
reconstructed_images_binary = (reconstructed_images > 0.5).astype(np.float32)  # تبدیل به 0 و 1

# 5. نمایش تصاویر اصلی و بازسازی شده
plt.figure(figsize=(15, 5))
for i in range(len(original_images)):
    # تصویر اصلی
    ax = plt.subplot(2, len(original_images), i + 1)
    plt.imshow(original_images[i], cmap="gray")
    plt.title("تصویر اصلی")
    plt.axis("off")

    # تصویر بازسازی شده
    ax = plt.subplot(2, len(original_images), i + len(original_images) + 1)
    plt.imshow(reconstructed_images[i].squeeze(), cmap="gray")
    plt.title("بازسازی شده")
    plt.axis("off")

plt.tight_layout()
plt.show()
# %%
