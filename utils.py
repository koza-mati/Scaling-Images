import numpy as np
from PIL import Image

#Tworzymy przykładowy obraz szachownicy
def create_sample_image():
    print("Tworzę przykładowy obraz szachownicy...")
    img = np.zeros((200, 200), dtype=np.uint8)
    for i in range(0, 200, 40):
        for j in range(0, 200, 40):
            if (i // 40 + j // 40) % 2 == 0:
                img[i:i+40, j:j+40] = 255
    return img.astype(np.float32)

# Wczytywanie obrazu
def load_image_with_pil(image_path):
    try:
        img = Image.open(image_path)
        img_gray = img.convert('L')  # Konwersja do skali szarości
        return np.array(img_gray, dtype=np.float32)
    except:
        return None

# Obliczanie MSE między orginałem a przetworzonym obrazem
def calculate_mse(original, processed):
    from sklearn.metrics import mean_squared_error
    
    # Przycinamy do wspólnego rozmiaru
    min_h = min(original.shape[0], processed.shape[0])
    min_w = min(original.shape[1], processed.shape[1])
    
    original_crop = original[:min_h, :min_w]
    processed_crop = processed[:min_h, :min_w]
    
    return mean_squared_error(original_crop, processed_crop)