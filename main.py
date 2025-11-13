import matplotlib.pyplot as plt
from utils import create_sample_image, load_image_with_pil, calculate_mse
from image_processing import downsample_average, downsample_max_pooling, non_integer_scale
from interpolation import upsample_linear, upsample_cubic, upsample_sinc, sequential_upscale

def main():
    # 1. Wczytanie obrazu - próbujemy PIL, jak nie to tworzymy szachownicę
    img = load_image_with_pil('obraz.jpg')
    if img is None:
        img = create_sample_image()
    
    print(f"Oryginalny rozmiar: {img.shape}")
    
    # 2. Pomniejszenie
    print("Pomniejszanie...")
    small_avg = downsample_average(img, 2)
    small_max = downsample_max_pooling(img, 2)
    
    print(f"Pomniejszony rozmiar: {small_avg.shape}")
    
    # 3. Powiększanie różnymi metodami
    print("Powiększanie...")
    big_linear = upsample_linear(small_avg, 2)
    big_cubic = upsample_cubic(small_avg, 2)
    big_sinc = upsample_sinc(small_avg, 2)
    
    # 4. Sekwencyjne powiększanie
    print("Sekwencyjne powiększanie...")
    big_seq = sequential_upscale(small_avg, 4, upsample_linear, steps=2)
    
    # 5. Skalowanie niecałkowite
    print("Skalowanie niecałkowite...")
    big_non_int = non_integer_scale(img, 1.7, 1.3)
    
    # 6. Obliczanie MSE
    print("Obliczanie MSE...")
    mse_linear = calculate_mse(img, big_linear)
    mse_cubic = calculate_mse(img, big_cubic)
    mse_sinc = calculate_mse(img, big_sinc)
    
    print("\n=== WYNIKI MSE ===")
    print(f"Linear: {mse_linear:.6f}")
    print(f"Cubic:  {mse_cubic:.6f}")
    print(f"Sinc:   {mse_sinc:.6f}")
    print("==================\n")
    
    # 7. Wizualizacja
    print("Tworzenie wykresów...")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    images = [
        img, small_avg, big_linear, big_cubic,
        big_sinc, big_seq, big_non_int, small_max
    ]
    
    titles = [
        'Oryginał', 'Pomniejszony (avg)', f'Linear\nMSE: {mse_linear:.6f}',
        f'Cubic\nMSE: {mse_cubic:.6f}', f'Sinc\nMSE: {mse_sinc:.6f}',
        'Sekwencyjny 2×2', 'Niecałkowite 1.7×1.3', 'Max Pooling'
    ]
    
    for i, (ax, image, title) in enumerate(zip(axes.flat, images, titles)):
        ax.imshow(image, cmap='gray')
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("Gotowe! Wyniki pokazane na wykresie.")

if __name__ == "__main__":
    main()