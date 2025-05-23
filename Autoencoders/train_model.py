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
LATENT_DIM = 32

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
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(LATENT_DIM, activation='relu')
    ])
    
    # Decoder
    decoder = tf.keras.Sequential([
        tf.keras.layers.InputLayer((LATENT_DIM,)),
        tf.keras.layers.Dense(16*16*64, activation='relu'),
        tf.keras.layers.Reshape((16, 16, 64)),
        tf.keras.layers.Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same'),
        tf.keras.layers.Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(1, 3, activation='sigmoid', padding='same')
    ])
    
    autoencoder = tf.keras.Sequential([encoder, decoder])

# 5. کامپایل و آموزش با GPU
autoencoder.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='mse',
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

# آموزش با استفاده از Dataset API برای بهینه‌سازی حافظه
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, X_train))
train_dataset = train_dataset.shuffle(1024).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((X_test, X_test))
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

history = autoencoder.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=test_dataset,
    callbacks=callbacks,
    verbose=1
)

# 6. ذخیره مدل نهایی
autoencoder.save("autoencoder_final_gpu.h5")

print("✅ آموزش با موفقیت انجام شد!")