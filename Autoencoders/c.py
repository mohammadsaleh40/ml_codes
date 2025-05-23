import cv2
import numpy as np
import os

# ایجاد پوشه داده
os.makedirs("data", exist_ok=True)

for i in range(64):
    for j in range(64):
        ekhtelaf = j - i
        bala_raftan = ekhtelaf / 44

        img = np.zeros((64, 64), dtype=np.uint8)
        img[0:10, i] = 255
        img[54:64, j] = 255
        
        for m in range(44):
            x_pos = int(bala_raftan * m) + i
            # جلوگیری از خروج از محدوده آرایه
            if 0 <= x_pos < 64:
                img[m + 10, x_pos] = 255
        
        # ذخیره تصویر
        cv2.imwrite(f"data/{i}_{j}.png", img)