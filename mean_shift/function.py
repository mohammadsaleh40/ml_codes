import numpy as np
def distans_pythagoras(a,b):
    if len(a) != len(b):
        # ارسال ارور
        raise ValueError("تعداد عناصر متنی با توجه به هم برابر نیست")
    
    return (sum((x - y) ** 2 for x, y in zip(a, b))) ** 0.5
def gaussian(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

