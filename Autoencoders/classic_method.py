# %%
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# %%
def find_i_j(img):
    i_kh = 0
    j_kh = 0
    for i in range(64):
        
        if (img[0:10,i]==[255, 255, 255, 255, 255, 255, 255, 255, 255, 255]).all():
            i_kh = i
        if (img[54:64,i]==[255, 255, 255, 255, 255, 255, 255, 255, 255, 255]).all():
            j_kh = i
    return i_kh , j_kh
# %%
for i in range(64):
    for j in range(64):
        file_name = f"{i}_{j}.png"
        directory = "data"
        file_path = os.path.join(directory, file_name)
        
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        i_kh , j_kh = find_i_j(img)
        
        print(f"i , j : {i} , {j} , i_kh , j_kh : {i_kh} , {j_kh}")

# %%
