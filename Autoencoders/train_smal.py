# %%
import tensorflow as tf
from tensorflow.keras import layers, Model
import tensorflow.keras.backend as K
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
# %%
# ----------------------------
# 1. Hyperparameters
# ----------------------------
INPUT_DIM = 64 * 64  # 4096
LATENT_DIM = 2       # کوچکترین فضای منطقی
EPOCHS = 200  # یا 300
BATCH_SIZE = 128  # تغییر این مقدار (مثلاً 128، 256، 512)
# %%
# ----------------------------
# 2. Encoder/Decoder/Autoencoder
# ----------------------------
def build_encoder():
    return tf.keras.Sequential([
        layers.Flatten(input_shape=(64, 64, 1)),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(2, activation='linear')  # فضای Latent
    ])

def build_decoder():
    return tf.keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(2,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(INPUT_DIM, activation='sigmoid'),
        layers.Reshape((64, 64, 1))
    ])

class SimpleAutoencoder(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# ----------------------------
# 3. ساخت مدل
# ----------------------------
encoder = build_encoder()
decoder = build_decoder()
autoencoder = SimpleAutoencoder(encoder, decoder)

def custom_loss(y_true, y_pred):
    # محاسبه ضریب برای پیکسل‌های سفید
    weight = tf.where(y_true > 0.5, 10.0, 1.0)

    # استفاده از K.binary_crossentropy برای بازگشت بدون کاهش
    bce = K.binary_crossentropy(y_true, y_pred)

    # اعمال ضریب و میانگین گیری کلی
    return tf.reduce_mean(weight * bce)

autoencoder.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),  # تغییر این قسمت
    loss=custom_loss,
    metrics=['mae']
)

# %%
# ----------------------------
# 4. پیکربندی GPU
# ----------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"{len(gpus)} GPU فیزیکی تشخیص داده شد")
        print(f"{len(logical_gpus)} GPU منطقی ساخته شد")
    except RuntimeError as e:
        print(e)
else:
    print("⚠️ هیچ GPU یافت نشد! از CPU استفاده می‌شود")

# ----------------------------
# 5. بارگذاری داده
# ----------------------------
def load_data():
    images = []
    files = [f for f in os.listdir("data") if f.endswith('.png')]
    for file in files:
        img = cv2.imread(os.path.join("data", file), cv2.IMREAD_GRAYSCALE)
        images.append(img)
    return np.array(images)

images = load_data()
images = images.astype('float32') / 255.0
images = np.expand_dims(images, -1)  # شکل: (N, 64, 64, 1)
# %%
X_train, X_test = train_test_split(images, test_size=0.2, random_state=42)

# ----------------------------
# 6. تابع اضافه کردن نویز برای NumPy
# ----------------------------
def add_numpy_noise(images, noise_factor=0.2):
    noise = np.random.normal(size=images.shape, scale=1.0)
    return np.clip(images + noise_factor * noise, 0., 1.)

# ----------------------------
# 7. Dataset API با نویز
# ----------------------------
def tf_add_noise(images, noise_factor=0.2):
    noise = tf.random.normal(shape=tf.shape(images), mean=0.0, stddev=1.0, dtype=images.dtype)
    return tf.clip_by_value(images + noise_factor * noise, 0., 1.)

# Train Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, X_train))
train_dataset = train_dataset.map(lambda x, y: (tf_add_noise(x), y), tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(1024).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Test Dataset
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, X_test))
test_dataset = test_dataset.map(lambda x, y: (tf_add_noise(x), y), tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE)

# ----------------------------
# 8. آموزش مدل
# ----------------------------

callbacks = [
    tf.keras.callbacks.ModelCheckpoint('best_autoencoder_gpu.h5', save_best_only=True, monitor='val_loss'),
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
]
# %%
history = autoencoder.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=test_dataset,
    callbacks=callbacks,
    verbose=1
)
# %%
# ----------------------------
# 9. نمایش Latent Space
# ----------------------------
import matplotlib.pyplot as plt

latents = encoder.predict(images)
i_values = []
j_values = []

for file in os.listdir("data"):
    if file.endswith(".png"):
        parts = file.split("_")
        i = int(parts[0])
        j = int(parts[1].replace(".png", ""))
        i_values.append(i)
        j_values.append(j)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(latents[:, 0], latents[:, 1], c=i_values, cmap='viridis', s=5)
plt.colorbar(label="i")

plt.subplot(1, 2, 2)
plt.scatter(latents[:, 0], latents[:, 1], c=j_values, cmap='plasma', s=5)
plt.colorbar(label="j")
plt.tight_layout()
plt.show()
# %%
# ----------------------------
# 10. نمایش تصاویر بازسازی شده
# ----------------------------
# ایجاد تصاویر نویزدار تست با استفاده از NumPy
X_test_noisy = add_numpy_noise(X_test)

# 5 تصویر تصادفی
sample_idx = np.random.choice(len(X_test), 5).tolist()
test_samples = X_test_noisy[sample_idx]

# بازسازی
reconstructions = autoencoder.predict(test_samples)

# نمایش
import cv2
for i in range(len(test_samples)):
    original = (test_samples[i].squeeze() * 255).astype(np.uint8)
    reconstructed = (reconstructions[i].squeeze() * 255).astype(np.uint8)
    combined = np.hstack([original, reconstructed])
    cv2.imshow(f"Sample {i}", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# %%
