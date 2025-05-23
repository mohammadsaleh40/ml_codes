import tensorflow as tf
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split

# 1. پیکربندی اولیه برای GPU
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

# 2. پارامترهای مدل
IMG_SIZE = 64
BATCH_SIZE = 64
EPOCHS = 100
LATENT_DIM = 64

# 3. بارگذاری و پیش‌پردازش داده‌ها
def load_data():
    images = []
    files = [f for f in os.listdir("data") if f.endswith('.png')]
    for file in files:
        img = cv2.imread(os.path.join("data", file), cv2.IMREAD_GRAYSCALE)
        images.append(img)
    return np.array(images)

images = load_data()
images = images.astype('float32') / 255.0
images = np.expand_dims(images, -1)  # (64,64,1)

# تقسیم داده‌ها
X_train, X_test = train_test_split(images, test_size=0.2, random_state=42)

# 4. ساخت مدل با بهینه‌سازی GPU
with tf.device('/GPU:0' if gpus else '/CPU:0'):
    # Encoder
    encoder = tf.keras.Sequential([
        tf.keras.layers.InputLayer((IMG_SIZE, IMG_SIZE, 1)),
        tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),  # لایه جدید
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(LATENT_DIM, activation='relu')
    ])

    # Decoder
    decoder = tf.keras.Sequential([
        tf.keras.layers.InputLayer((LATENT_DIM,)),
        tf.keras.layers.Dense(8*8*128, activation='relu'),  # تغییر این قسمت
        tf.keras.layers.Reshape((8, 8, 128)),  # تغییر این قسمت
        tf.keras.layers.Conv2DTranspose(128, 3, strides=2, activation='relu', padding='same'),
        tf.keras.layers.Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same'),
        tf.keras.layers.Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(1, 3, activation='sigmoid', padding='same')
    ])
    
    autoencoder = tf.keras.Sequential([encoder, decoder])

# 5. کامپایل و آموزش با GPU
autoencoder.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']  # افزودن این قسمت
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
def add_noise(images, noise_factor=0.2):
    noise = tf.random.normal(shape=tf.shape(images), mean=0.0, stddev=1.0, dtype=images.dtype)
    noisy_images = images + noise_factor * noise
    return tf.clip_by_value(noisy_images, 0., 1.)
X_train_noisy = add_noise(X_train)
X_test_noisy = add_noise(X_test)

# آموزش با استفاده از Dataset API برای بهینه‌سازی حافظه
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, X_train))
train_dataset = train_dataset.shuffle(1024).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((X_test, X_test))
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

history = autoencoder.fit(
    train_dataset.map(lambda x, y: (add_noise(x), y)),  # اضافه کردن نویز در حین آموزش
    epochs=EPOCHS,
    validation_data=test_dataset.map(lambda x, y: (add_noise(x), y)),
    callbacks=callbacks,
    verbose=1
)

# 6. ذخیره مدل نهایی
autoencoder.save("autoencoder_final_gpu.h5")

print("✅ آموزش با موفقیت انجام شد!")