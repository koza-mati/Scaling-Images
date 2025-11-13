import numpy as np

# Pomniejszanie obrazu przez uśrednianie
def downsample_average(image, scale_factor=2):
    h, w = image.shape
    new_h, new_w = h // scale_factor, w // scale_factor
    
    downsampled = np.zeros((new_h, new_w), dtype=np.float32)
    
    for i in range(new_h):
        for j in range(new_w):
            block = image[i*scale_factor:(i+1)*scale_factor, 
                         j*scale_factor:(j+1)*scale_factor]
            downsampled[i, j] = np.mean(block)
    
    return downsampled

# Pomniejszanie obrazu przez max pooling
def downsample_max_pooling(image, scale_factor=2):
    h, w = image.shape
    new_h, new_w = h // scale_factor, w // scale_factor
    
    downsampled = np.zeros((new_h, new_w), dtype=np.float32)
    
    for i in range(new_h):
        for j in range(new_w):
            block = image[i*scale_factor:(i+1)*scale_factor, 
                         j*scale_factor:(j+1)*scale_factor]
            downsampled[i, j] = np.max(block)
    
    return downsampled

# Skalowanie o niecałkowitą wielokroność
def non_integer_scale(image, scale_x, scale_y):
    from scipy.interpolate import RegularGridInterpolator
    
    h, w = image.shape
    new_h, new_w = int(h * scale_y), int(w * scale_x)
    
    # Tworzenie siatki oryginalnych współrzędnych
    x_orig = np.arange(w)
    y_orig = np.arange(h)
    
    # Tworzenie siatki nowych współrzędnych
    x_new = np.linspace(0, w-1, new_w)
    y_new = np.linspace(0, h-1, new_h)
    
    # Tworzenie interpolator
    interpolator = RegularGridInterpolator((y_orig, x_orig), image, method='linear', bounds_error=False, fill_value=0)
    
    # Tworzenie siatki punktów do interpolacji
    Y_new, X_new = np.meshgrid(y_new, x_new, indexing='ij')
    points = np.stack([Y_new, X_new], axis=-1)
    
    # Interpolacja
    scaled = interpolator(points)
    
    return scaled