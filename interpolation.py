import numpy as np
from scipy import signal

# Jądro liniowej interpolacji
def linear_kernel(t):
    return (t >= 0) & (t < 1)

# Jądro kubicznej interpolacji
def cubic_kernel(t):
    return (t >= -0.5) & (t < 0.5)

# Jądro interpolacji sinc
def sinc_kernel(t):
    return np.clip(1 - np.abs(t), 0, None)


#  Interpolacja przez splot z jądrem
def apply_convolution_interpolation(image, scale_factor, kernel_func, kernel_radius=2):
    h, w = image.shape
    new_h, new_w = h * scale_factor, w * scale_factor
    
    expanded = np.zeros((new_h, new_w), dtype=np.float32)
    expanded[::scale_factor, ::scale_factor] = image
    
    kernel_size = 2 * kernel_radius * scale_factor + 1
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    
    center = kernel_size // 2
    for i in range(kernel_size):
        for j in range(kernel_size):
            dx = (i - center) / scale_factor
            dy = (j - center) / scale_factor
            kernel[i, j] = kernel_func(dx) * kernel_func(dy)
    
    kernel = kernel / np.sum(kernel)
    interpolated = signal.convolve2d(expanded, kernel, mode='same', boundary='symm')
    
    return interpolated

def upsample_linear(image, scale_factor=2):
    return apply_convolution_interpolation(image, scale_factor, linear_kernel, kernel_radius=1)

def upsample_cubic(image, scale_factor=2):
    return apply_convolution_interpolation(image, scale_factor, cubic_kernel, kernel_radius=2)

def upsample_sinc(image, scale_factor=2):
    return apply_convolution_interpolation(image, scale_factor, sinc_kernel, kernel_radius=3)

# Sekwencyjne powiększanie
def sequential_upscale(image, scale_factor, method, steps=2):
    result = image.copy()
    for i in range(steps):
        result = method(result, 2)
    return result