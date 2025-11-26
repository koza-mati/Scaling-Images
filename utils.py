import numpy as np
import kernels

# Pobranie piksela uwzgledniając brzegi
def get_pixel(img, i, j):
    if 0 <= i < img.shape[0] and 0 <= j < img.shape[1]:
        return img[i, j]
    return 0.0

# Splot
def splot(image, kernel):
    # Ustalanie rozmiarów obrazu i kernela
    height, width = image.shape
    k_height, k_width = kernel.shape
    # Wyliczanie odstępów
    pad_h = k_height // 2
    pad_w = k_width // 2
    
    # Inicjalizacja obrazu wyjściowego
    output = np.zeros_like(image, dtype=float)
    
    # Wykonywanie splotu za pomocą podwójnej pętli
    for i in range(height):
        for j in range(width):
            s = 0.0 # Inicjalizacja sumy 
            for ki in range(k_height):
                for kj in range(k_width):
                    s= get_pixel(image, i + ki - pad_h, j + kj - pad_w) * kernel[ki, kj] # Mnożenie i sumowanie 
            output[i, j] = s
    return output

# Pomniejszanie obrazu metodą średniej
def reduce_image(image, n, m, N):
    kernel = (1 / N) * np.ones((n, m))
    blurred = splot(image, kernel)
    return blurred[::n, ::m]

# Interpolacja wykorzystana z poprzedniego zadania
def interpolate_kernel(x, y, t, kernel):
    dx = np.median(np.diff(x))
    K = kernel((t[:, None] - x[None, :]) / dx)
    W = K.sum(axis=1)
    W[W == 0] = 1
    return (K @ y) / W.ravel()

# Powiększanie obrazu z użyciem jądra
def enlarge_image_kernel(image, target_shape, kernel):
    h_in, w_in = image.shape
    h_out, w_out = target_shape

    # siatki wejściowe
    x_in = np.arange(w_in)
    y_in = np.arange(h_in)

    # siatki wyjściowe
    x_out = np.linspace(0, w_in - 1, w_out)
    y_out = np.linspace(0, h_in - 1, h_out)

    # interpolacja poziomo
    temp = np.zeros((h_in, w_out))
    for i in range(h_in):
        temp[i, :] = interpolate_kernel(x_in, image[i, :], x_out, kernel)

    # interpolacja pionowo
    out = np.zeros((h_out, w_out))
    for j in range(w_out):
        out[:, j] = interpolate_kernel(y_in, temp[:, j], y_out, kernel)

    return out

# Obliczanie MSE 
def mse(img1, img2):
    return np.mean((img1.astype(float) - img2.astype(float)) ** 2)

# Pomniejszanie obrazu metodą max-pooling
def reduce_image_maxpool(image, n, m):
    H, W = image.shape

    # Przycięcie obrazu tak, aby był całkowicie podzielny
    H2 = H - (H % n)
    W2 = W - (W % m)
    img = image[:H2, :W2]

    # Dzielenie na bloki i max pooling
    pooled = img.reshape(H2 // n, n, W2 // m, m).max(axis=(1, 3))

    return pooled
