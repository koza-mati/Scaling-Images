import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from kernels import h_rect_sym, h_tri
from utils import reduce_image, reduce_image_maxpool, enlarge_image_kernel, mse



# 1. Wczytanie obrazu
img_path = "pies.jpg"
img = io.imread(img_path, as_gray=True)

print("Original shape:", img.shape)


# 2. Pomniejszenie obrazu
small_img = reduce_image(img, n=4, m=4, N=16)
small_img_max = reduce_image_maxpool(img, n=4, m=4)
print("Reduced shape:", small_img.shape)

# 2. Pomniejszenie — metoda średniej
small_img_avg = reduce_image(img, n=4, m=4, N=16)

# 2b. Pomniejszenie — metoda max pooling
small_img_max = reduce_image_maxpool(img, n=4, m=4)


# 3. Powiększenie obrazu (kernel jądrowy)
target_shape = img.shape  # (521, 800)

enlarged_rect = enlarge_image_kernel(small_img, target_shape, h_rect_sym)
enlarged_tri  = enlarge_image_kernel(small_img, target_shape, h_tri)

print("Upscaled (rect) shape:", enlarged_rect.shape)
print("Upscaled (tri) shape:", enlarged_tri.shape)


# 4. Obliczenie MSE
mse_rect = mse(img, enlarged_rect)
mse_tri = mse(img, enlarged_tri)

print("MSE (Rectangular kernel):", mse_rect)
print("MSE (Triangular kernel):", mse_tri)


# 5. Rysowanie wyników
plt.figure(figsize=(16, 12))

plt.subplot(3, 3, 1)
plt.imshow(img, cmap='gray')
plt.title("Original")

plt.subplot(3, 3, 2)
plt.imshow(small_img_avg, cmap='gray')
plt.title("Reduced (avg)")

plt.subplot(3, 3, 3)
plt.imshow(small_img_max, cmap='gray')
plt.title("Reduced (max pool)")

plt.subplot(3, 3, 7)
plt.imshow(enlarged_rect, cmap='gray')
plt.title(f"Increased - kernel rect\nMSE={mse_rect:.4f}")

plt.subplot(3, 3, 8)
plt.imshow(enlarged_tri, cmap='gray')
plt.title(f"Increased - kernel tri\nMSE={mse_tri:.4f}")

plt.tight_layout()
plt.show()
