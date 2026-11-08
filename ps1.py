import cv2
import numpy as np
from PIL import Image

# 4x4 Standart PS1 / Retro Bayer Matrisi (Ordered Dithering için)
BAYER_MATRIX_4X4 = (np.array([
    [ 0,  8,  2, 10],
    [12,  4, 14,  6],
    [ 3, 11,  1,  9],
    [15,  7, 13,  5]
], dtype=np.float32) / 16.0) - 0.5


def apply_bayer_dither(image_np: np.ndarray, spread: float = 16.0) -> np.ndarray:
    """Görsele donanımsal PS1 Bayer dither deseni ekler."""
    h, w, c = image_np.shape
    # Bayer matrisini görsel boyutuna döşe
    tiled_bayer = np.tile(BAYER_MATRIX_4X4, (h // 4 + 1, w // 4 + 1))[:h, :w]
    tiled_bayer = np.repeat(tiled_bayer[:, :, np.newaxis], c, axis=2)

    # Matris gürültüsünü piksel değerlerine ekle
    dithered = image_np.astype(np.float32) + (tiled_bayer * spread)
    return np.clip(dithered, 0, 255)


def quantize_to_rgb555(image_np: np.ndarray) -> np.ndarray:
    """24-bit rengi (8-8-8) orijinal PS1 donanımı olan 15-bit renge (5-5-5) düşürür."""
    # 8-bit (0-255) değerleri 5-bit (0-31) aralığına ölçeklendirip geri çevir
    levels = 32.0  # 2^5 = 32 seviye
    quantized = np.round(image_np * (levels - 1) / 255.0) * (255.0 / (levels - 1))
    return quantized.astype(np.uint8)


def convert_image_to_ps1(
    input_path: str,
    output_path: str,
    target_width: int = 320,
    target_height: int = 240,
    dither_strength: float = 18.0,
    scanlines: bool = True
):
    # 1. Görseli yükle (RGB)
    img = Image.open(input_path).convert("RGB")
    original_size = img.size

    # 2. Poligon / Çizgisel Düzleştirme (Opsiyonel: 3D model hissi vermek için hafif filtre)
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    cv_img = cv2.bilateralFilter(cv_img, d=7, sigmaColor=50, sigmaSpace=50)
    img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

    # 3. PS1 Çözünürlüğüne İndir (240p Downscale)
    downscaled = img.resize((target_width, target_height), resample=Image.Resampling.BILINEAR)
    img_np = np.array(downscaled, dtype=np.float32)

    # 4. Donanımsal Dithering ve 15-bit Renk Kuantizasyonu
    dithered_np = apply_bayer_dither(img_np, spread=dither_strength)
    ps1_colored = quantize_to_rgb555(dithered_np)

    # 5. Orijinal Boyutuna "Nearest Neighbor" ile Büyüt (Piksel kenarlarını keskinleştirir)
    result_img = Image.fromarray(ps1_colored).resize(original_size, resample=Image.Resampling.NEAREST)

    # 6. CRT Ekran / Scanline Çizgileri (Opsiyonel nostalji efekti)
    if scanlines:
        res_np = np.array(result_img, dtype=np.float32)
        # Her çift satırda parlaklığı hafifçe kıs
        res_np[::2, :, :] *= 0.85
        result_img = Image.fromarray(np.clip(res_np, 0, 255).astype(np.uint8))

    result_img.save(output_path)
    print(f"Dönüştürme tamamlandı: {output_path}")


if __name__ == "__main__":
    # Test etmek için:
    convert_image_to_ps1(
        input_path="foto1.jpg",
        output_path="ps1_retro_output.png",
        target_width=320,
        target_height=240,
        dither_strength=18.0,
        scanlines=True
    )