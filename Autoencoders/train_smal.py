# %%
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import tensorflow as tf
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split

# %%
# ----------------------------
# 1. Hyperparameters
# ----------------------------
INPUT_DIM = 64 * 64  # 4096
LATENT_DIM = 2      # کوچکترین فضای منطقی
EPOCHS = 100
ENCODER_UNITS = [32, 16, 8, 4]  # لایه‌های Encoder
DECODER_UNITS = [4, 8, 16, 32]  # لایه‌های Decoder

# ----------------------------
# 2. Encoder
# ----------------------------
def build_encoder():
    return tf.keras.Sequential([
        layers.Flatten(input_shape=(64, 64, 1)),
        layers.Dense(ENCODER_UNITS[0], activation='relu'),
        layers.Dense(ENCODER_UNITS[1], activation='relu'),
        layers.Dense(ENCODER_UNITS[2], activation='relu'),
        layers.Dense(ENCODER_UNITS[3], activation='relu'),
        layers.Dense(LATENT_DIM, activation='linear', name='latent')  # فضای کاهش یافته
    ], name="Encoder")

# ----------------------------
# 3. Decoder
# ----------------------------
def build_decoder():
    return tf.keras.Sequential([
        layers.Dense(DECODER_UNITS[0], activation='relu', input_shape=(LATENT_DIM,)),
        layers.Dense(DECODER_UNITS[1], activation='relu'),
        layers.Dense(DECODER_UNITS[2], activation='relu'),
        layers.Dense(DECODER_UNITS[3], activation='relu'),
        layers.Dense(INPUT_DIM, activation='sigmoid'),
        layers.Reshape((64, 64, 1))
    ], name="Decoder")

# ----------------------------
# 4. Autoencoder
# ----------------------------
class SimpleAutoencoder(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# ----------------------------
# 5. ساخت و کامپایل مدل
# ----------------------------
encoder = build_encoder()
decoder = build_decoder()
autoencoder = SimpleAutoencoder(encoder, decoder)

autoencoder.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',  # تغییر این قسمت
    metrics=['mae']
)

# ایجاد callback برای مانیتورینگ
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        'best_autoencoder_gpu.h5',
        save_best_only=True,
        monitor='val_loss'
    ),
    tf.keras.callbacks.EarlyStopping(
        patience=10,
        restore_best_weights=True
    )
]

# نمایش خلاصه
autoencoder.build((None, 64, 64, 1))
autoencoder.summary()
# %%
# پیکربندی اولیه برای GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # تنظیم رشد حافظه پویا برای GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"{len(gpus)} GPU فیزیکی تشخیص داده شد")
        print(f"{len(logical_gpus)} GPU منطقی ساخته شد")
    except RuntimeError as e:
        print(e)
else:
    print("⚠️ هیچ GPU یافت نشد! از CPU استفاده می‌شود")
def load_data():
    images = []
    files = [f for f in os.listdir("data") if f.endswith('.png')]
    for file in files:
        img = cv2.imread(os.path.join("data", file), cv2.IMREAD_GRAYSCALE)
        images.append(img)
    return np.array(images)

# بارگذاری داده (همانند کد قبلی شما)
images = load_data()  # تابع شما برای بارگذاری داده
images = images.astype('float32') / 255.0
images = np.expand_dims(images, -1)  # شکل: (N, 64, 64, 1)

# تقسیم داده
X_train, X_test = train_test_split(images, test_size=0.2, random_state=42)

# Dataset API
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, X_train)).shuffle(1024).batch(64).prefetch(tf.data.AUTOTUNE)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, X_test)).batch(64)
def add_noise_np(images, noise_factor=0.2):
    noise = np.random.normal(size=images.shape, scale=1.0)
    return np.clip(images + noise_factor * noise, 0., 1.)

def add_noise(images, noise_factor=0.2):
    noise = tf.random.normal(shape=tf.shape(images), mean=0.0, stddev=1.0, dtype=images.dtype)
    noisy_images = images + noise_factor * noise
    return tf.clip_by_value(noisy_images, 0., 1.)
X_train_noisy = add_noise(X_train)
X_test_noisy = add_noise(X_test)
# آموزش
history = autoencoder.fit(
    train_dataset.map(lambda x, y: (add_noise(x), y)),
    epochs=EPOCHS,
    validation_data=test_dataset.map(lambda x, y: (add_noise(x), y)),
    callbacks=callbacks,
    verbose=1
)
# %%
# کد نمایش Latent Space
import matplotlib.pyplot as plt

# تمام داده‌ها را در Encoder فید کن
latents = encoder.predict(images)

# i و j هر تصویر را استخراج کن (از اسم فایل‌ها)
i_values = []
j_values = []

for file in os.listdir("data"):
    if file.endswith(".png"):
        parts = file.split("_")
        i = int(parts[0])
        j = int(parts[1].replace(".png", ""))
        i_values.append(i)
        j_values.append(j)

# نمایش Latent Space به رنگ i و j
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
scatter = plt.scatter(latents[:, 0], latents[:, 1], c=i_values, cmap='viridis', s=5)
plt.title("Latent Space (رنگ‌گذاری شده با i)")
plt.xlabel("z1")
plt.ylabel("z2")
plt.colorbar(scatter, label="i")

plt.subplot(1, 2, 2)
scatter = plt.scatter(latents[:, 0], latents[:, 1], c=j_values, cmap='plasma', s=5)
plt.title("Latent Space (رنگ‌گذاری شده با j)")
plt.xlabel("z1")
plt.ylabel("z2")
plt.colorbar(scatter, label="j")

plt.tight_layout()
plt.show()
# %%
# 5 تصویر تصادفی
sample_idx = np.random.choice(len(X_test_noisy), 5)
print(sample_idx)
test_samples = [X_test_noisy[i] for i in sample_idx]
# نرمالایز و اضافه کردن بعد کانال
original_normalized = X_test.astype('float32') / 255.0
original_normalized = np.expand_dims(original_normalized, -1)  # شکل: (batch_size, 64, 64, 1)

# اضافه کردن نویز به تصاویر
noisy_images = add_noise_np(original_normalized)

# 5. پیش‌بینی (بازسازی تصویر با استفاده از تصاویر نویزدار)
reconstructed_images = autoencoder.predict(X_test)

# نمایش با OpenCV
for i in range(5):
    original = (noisy_images[i].squeeze() * 255).astype(np.uint8)
    reconstructed = (reconstructed_images[i].squeeze() * 255).astype(np.uint8)
    np.expand_dims(noisy_images, -1)  # شکل: (batch_size, 64, 64, 1)
    combined = np.hstack([original, reconstructed])
    cv2.imshow(f"Original vs Reconstructed {i}", combined)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
# %%

test_samples[1].shape
# %%
autoencoder.shape
# %%
for idx in sample_idx:
    print(f"Shape of X_test_noisy[{idx}]: {X_test_noisy[idx].shape}")

# %%
