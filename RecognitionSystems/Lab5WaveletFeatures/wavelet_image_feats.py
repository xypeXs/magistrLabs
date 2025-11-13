import matplotlib.pyplot as plt
import numpy
import numpy as np
from scipy.stats import entropy
from PIL import Image


def load_image(image_path):
    img = Image.open(image_path).convert('L')
    img_data = np.array(img, dtype=float)
    return img_data

def normalize(image):
    normalized_image = image.copy()
    # for i in range(normalized_image.shape[0]):
    #     for j in range(normalized_image.shape[1]):
    #         normalized_image[i, j] = 0 if normalized_image[i, j] == 0 else 1
    return normalized_image

def wavelet_2d(image):
    tempWaveletAprxLevels = [normalize(image)]
    tempWaveletDetailLevels = []

    current_level = 0
    while tempWaveletAprxLevels[current_level].shape[0] >= 2 and tempWaveletAprxLevels[current_level].shape[1] >= 2:
        current_img = tempWaveletAprxLevels[current_level]
        h, w = current_img.shape

        approx_rows = (current_img[0::2, :] + current_img[1::2, :]) / 2
        detail_rows = (current_img[0::2, :] - current_img[1::2, :]) / 2

        approx = (approx_rows[:, 0::2] + approx_rows[:, 1::2]) / 2
        horizontal = (approx_rows[:, 0::2] - approx_rows[:, 1::2]) / 2
        vertical = (detail_rows[:, 0::2] + detail_rows[:, 1::2]) / 2
        diagonal = (detail_rows[:, 0::2] - detail_rows[:, 1::2]) / 2

        tempWaveletAprxLevels.append(approx)

        if current_level == 0:
            tempWaveletDetailLevels = [{
                'horizontal': horizontal,
                'vertical': vertical,
                'diagonal': diagonal
            }]
        else:
            tempWaveletDetailLevels.append({
                'horizontal': horizontal,
                'vertical': vertical,
                'diagonal': diagonal
            })

        current_level += 1

    return tempWaveletAprxLevels[1:], tempWaveletDetailLevels


def show_wavelet_2d(approx_levels, detail_levels, original):
    levels = len(approx_levels)
    fig, axes = plt.subplots(levels + 1, 4, figsize=(15, 4 * (levels + 1)))

    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    for i in range(levels):
        axes[i + 1, 0].imshow(approx_levels[i], cmap='gray')
        axes[i + 1, 0].set_title(f'Approximation Level {i + 1}')
        axes[i + 1, 0].axis('off')

        axes[i + 1, 1].imshow(detail_levels[i]['horizontal'], cmap='gray')
        axes[i + 1, 1].set_title(f'Horizontal Details Level {i + 1}')
        axes[i + 1, 1].axis('off')

        axes[i + 1, 2].imshow(detail_levels[i]['vertical'], cmap='gray')
        axes[i + 1, 2].set_title(f'Vertical Details Level {i + 1}')
        axes[i + 1, 2].axis('off')

        axes[i + 1, 3].imshow(detail_levels[i]['diagonal'], cmap='gray')
        axes[i + 1, 3].set_title(f'Diagonal Details Level {i + 1}')
        axes[i + 1, 3].axis('off')

    plt.tight_layout()
    plt.show()


def compute_energy_2d(wavelet_data):
    # if isinstance(wavelet_data, list):
    #     energy = []
    #     for level in wavelet_data:
    #         if isinstance(level, dict):
    #             level_energy = {}
    #             for key, coeffs in level.items():
    #                 level_energy[key] = np.sum(coeffs ** 2)
    #             energy.append(level_energy)
    #         else:
    #             energy.append(np.sum(level ** 2))
    #     return energy
    # else:
    res = 0.0
    for i in range(len(wavelet_data)):
        res += wavelet_data[i]
    return res


def compute_wavelet_statistics(approx_levels, detail_levels):
    stats = {}

    stats['approx'] = []
    for i, approx in enumerate(approx_levels):
        stats['approx'].append({
            'level': i + 1,
            'mean': np.mean(approx),
            'std': np.std(approx),
            'energy': np.sum(approx ** 2),
            'entropy': compute_entropy(approx)
        })

    for coeff_type in ['horizontal', 'vertical', 'diagonal']:
        stats[coeff_type] = []
        for i, level in enumerate(detail_levels):
            coeffs = level[coeff_type]
            stats[coeff_type].append({
                'level': i + 1,
                'mean': np.mean(coeffs),
                'std': np.std(coeffs),
                'energy': np.sum(coeffs ** 2),
                'entropy': compute_entropy(coeffs)
            })

    return stats

def compute_entropy(data, bins=256):
    """
    Вычисляет энтропию массива данных
    """
    if data.size == 0:
        return 0

    # Нормализуем данные для построения гистограммы
    data_flat = data.flatten()
    data_min, data_max = np.min(data_flat), np.max(data_flat)

    if data_max - data_min == 0:
        return 0  # Все значения одинаковы - энтропия минимальна

    # Строим гистограмму и вычисляем вероятности
    hist, _ = np.histogram(data_flat, bins=bins, range=(data_min, data_max), density=True)

    # Убираем нулевые значения для избежания log(0)
    hist = hist[hist > 0]

    # Вычисляем энтропию: -sum(p * log2(p))
    entropy_val = -np.sum(hist * np.log2(hist))

    return entropy_val

def process_image(image_path):
    image_data = load_image(image_path)
    print(f"Image size: {image_data.shape}")

    approx_levels, detail_levels = wavelet_2d(image_data)

    show_wavelet_2d(approx_levels, detail_levels, image_data)

    stats = compute_wavelet_statistics(approx_levels, detail_levels)

    print("\n=== Wavelet Statistics ===")
    for coeff_type in ['approx', 'horizontal', 'vertical', 'diagonal']:
        print(f"\n{coeff_type.upper()} Coefficients:")
        for level_stats in stats[coeff_type]:
            print(f"  Level {level_stats['level']}: "
                  f"mean={level_stats['mean']:.4f}, "
                  f"std={level_stats['std']:.4f}, "
                  f"energy={level_stats['energy']:.4f}",
                  f"entropy={level_stats['entropy']:.4f}"
                  )

    # print("Entropy", st.entropy(image_data))
    return approx_levels, detail_levels, stats


if __name__ == "__main__":
    image_path = "SRLabImages/GZBMP24/zebra1.bmp"
    approx, details, statistics = process_image(image_path)
