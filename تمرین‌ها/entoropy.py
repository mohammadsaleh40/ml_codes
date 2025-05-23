import numpy as np
import matplotlib.pyplot as plt

# تابع آنتروپی (شانون)
def shannon_entropy(w1):
    w2 = 1 - w1
    if w1 == 0 or w2 == 0:
        return 0
    return -(w1 * np.log2(w1) + w2 * np.log2(w2))

# تابع شبه آنتروپی (پراکندگی مربعی)
def entropy_like(w1):
    return 1 - (w1**2 + (1 - w1)**2)

# تولید مقادیر بین 0 و 1
w1_values = np.linspace(0.001, 0.999, 500)  # برای جلوگیری از log(0)

# محاسبه مقادیر توابع
entropy_vals = [shannon_entropy(w1) for w1 in w1_values]
entropy_like_vals = [entropy_like(w1) for w1 in w1_values]

# رسم نمودار
plt.figure(figsize=(8, 5))
plt.plot(w1_values, entropy_vals, label='Shannon Entropy', color='blue')
plt.plot(w1_values, entropy_like_vals, label='Entropy-like Function', color='red', linestyle='--')
plt.xlabel('$w_1$')
plt.ylabel('Function Value')
plt.title('Entropy and Entropy-like Function')
plt.grid(True)
plt.legend()
plt.tight_layout()

# ذخیره به عنوان فایل PNG برای درج در LaTeX
plt.savefig('entropy_plot.png', dpi=300)

# نمایش
plt.show()
