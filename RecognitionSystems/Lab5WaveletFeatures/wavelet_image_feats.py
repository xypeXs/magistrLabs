import math
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_image(image_path):
    img = Image.open(image_path).convert('L')
    img_data = np.array(img, dtype=float)
    return img_data


def wavelet_2d(image):
    tempWaveletAprxLevels = [image]
    tempWaveletDetailLevels = []

    current_level = 0
    while tempWaveletAprxLevels[current_level].shape[0] >= 2 and tempWaveletAprxLevels[current_level].shape[1] >= 2:
        current_img = tempWaveletAprxLevels[current_level]
        h, w = current_img.shape

        h_even = h if h % 2 == 0 else h - 1
        w_even = w if w % 2 == 0 else w - 1

        if h_even < 2 or w_even < 2:
            break

        current_img = current_img[:h_even, :w_even]

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


def compute_energy(approx_levels, detail_levels):
    energy_stats = {}

    for i, (approx, details) in enumerate(zip(approx_levels, detail_levels)):
        level = i + 1

        std_approx = np.std(approx)

        energy_stats[level] = {
            'std_approx': std_approx,
            'details': {}
        }

        # Энергии деталей (пропорциональны std аппроксимаций)
        for coeff_type in ['horizontal', 'vertical', 'diagonal']:
            coeffs = details[coeff_type]
            # Энергия = сумма квадратов коэффициентов
            energy = np.sum(coeffs ** 2)
            normalized_energy = energy

            energy_stats[level]['details'][coeff_type] = {
                'raw_energy': energy,
                'normalized_energy': normalized_energy,
                'std': np.std(coeffs)
            }

    return energy_stats


def compute_entropy(approx_levels, detail_levels):
    entropy_stats = {}

    # Энтропия для приближающих коэффициентов
    entropy_stats['approx'] = []
    for i, approx in enumerate(approx_levels):
        level = i + 1
        n_coeffs = approx.size

        entropy_val = compute_shannon_entropy(approx)
        corrected_entropy = entropy_val / np.log2(n_coeffs + 1)

        entropy_stats['approx'].append({
            'level': level,
            'entropy': entropy_val,
            'corrected_entropy': corrected_entropy,
            'n_coeffs': n_coeffs,
            'size': approx.shape
        })

    for coeff_type in ['horizontal', 'vertical', 'diagonal']:
        entropy_stats[coeff_type] = []
        for i, level_details in enumerate(detail_levels):
            level = i + 1
            coeffs = level_details[coeff_type]
            n_coeffs = coeffs.size

            entropy_val = compute_shannon_entropy(coeffs)
            corrected_entropy = entropy_val / np.log2(n_coeffs + 1)

            entropy_stats[coeff_type].append({
                'level': level,
                'entropy': entropy_val,
                'corrected_entropy': corrected_entropy,
                'n_coeffs': n_coeffs,
                'size': coeffs.shape
            })

    return entropy_stats


def compute_shannon_entropy(data, bins=256):
    if data.size == 0:
        return 0

    data_flat = data.flatten()
    data_min, data_max = np.min(data_flat), np.max(data_flat)

    if data_max - data_min == 0:
        return 0

    hist, _ = np.histogram(data_flat, bins=bins, range=(data_min, data_max), density=True)
    hist = hist[hist > 0]

    entropy_val = -np.sum(hist * np.log2(hist))

    return entropy_val


def plot_corrected_analysis(energy_stats, entropy_stats):
    levels = list(energy_stats.keys())

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Скорректированный анализ вейвлет-коэффициентов', fontsize=16)

    # 1. STD аппроксимаций по уровням
    std_approx = [energy_stats[level]['std_approx'] for level in levels]
    axes[0, 0].plot(levels, std_approx, 'o-', linewidth=2, markersize=8, color='purple')
    axes[0, 0].set_title('STD аппроксимаций по уровням')
    axes[0, 0].set_xlabel('Уровень')
    axes[0, 0].set_ylabel('STD')
    axes[0, 0].grid(True)

    # 2. Нормализованная энергия деталей
    coeff_types = ['horizontal', 'vertical', 'diagonal']
    colors = ['red', 'green', 'blue']

    for i, coeff_type in enumerate(coeff_types):
        energies = [energy_stats[level]['details'][coeff_type]['normalized_energy'] for level in levels]
        axes[0, 1].plot(levels, energies, 'o-', linewidth=2, markersize=8,
                        color=colors[i], label=coeff_type)

    axes[0, 1].set_title('Нормализованная энергия деталей')
    axes[0, 1].set_xlabel('Уровень')
    axes[0, 1].set_ylabel('Энергия / STD_approx')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # 3. Скорректированная энтропия аппроксимаций
    approx_entropies = [e['corrected_entropy'] for e in entropy_stats['approx']]
    axes[0, 2].plot(levels, approx_entropies, 'o-', linewidth=2, markersize=8, color='orange')
    axes[0, 2].set_title('Скорректированная энтропия аппроксимаций')
    axes[0, 2].set_xlabel('Уровень')
    axes[0, 2].set_ylabel('Энтропия / log2(n_coeffs)')
    axes[0, 2].grid(True)

    # 4. Скорректированная энтропия деталей
    for i, coeff_type in enumerate(coeff_types):
        entropies = [e['corrected_entropy'] for e in entropy_stats[coeff_type]]
        axes[1, 0].plot(levels, entropies, 'o-', linewidth=2, markersize=8,
                        color=colors[i], label=coeff_type)

    axes[1, 0].set_title('Скорректированная энтропия деталей')
    axes[1, 0].set_xlabel('Уровень')
    axes[1, 0].set_ylabel('Энтропия / log2(n_coeffs)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # 5. Количество коэффициентов по уровням
    n_coeffs_approx = [e['n_coeffs'] for e in entropy_stats['approx']]
    axes[1, 1].plot(levels, n_coeffs_approx, 'o-', linewidth=2, markersize=8, color='brown', label='Approx')

    for i, coeff_type in enumerate(coeff_types):
        n_coeffs = [e['n_coeffs'] for e in entropy_stats[coeff_type]]
        axes[1, 1].plot(levels, n_coeffs, 'o-', linewidth=2, markersize=8,
                        color=colors[i], label=coeff_type)

    axes[1, 1].set_title('Количество коэффициентов по уровням')
    axes[1, 1].set_xlabel('Уровень')
    axes[1, 1].set_ylabel('Количество коэффициентов')
    axes[1, 1].legend()
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()


def process_image_corrected(image_path):
    image_data = load_image(image_path)
    approx_levels, detail_levels = wavelet_2d(image_data)

    print(f"Image size: {image_data.shape}")
    print(f"Number of decomposition levels: {len(approx_levels)}")

    energy_stats = compute_energy(approx_levels, detail_levels)
    entropy_stats = compute_entropy(approx_levels, detail_levels)

    print("\nSTD аппроксимаций и энергии деталей:")
    for level in energy_stats.keys():
        print(f"\nLevel {level}:")
        print(f"STD approx: {energy_stats[level]['std_approx']:.4f}")
        for coeff_type in ['horizontal', 'vertical', 'diagonal']:
            info = energy_stats[level]['details'][coeff_type]
            print(f"  {coeff_type}: energy={info['raw_energy']:.4f}, STD={info['std']:.4f}")

    for coeff_type in ['approx', 'horizontal', 'vertical', 'diagonal']:
        print(f"\n{coeff_type.upper()}:")
        for stats in entropy_stats[coeff_type]:
            print(f"Level {stats['level']}: entropy={stats['corrected_entropy']:.4f}")

    # print("\nСоотношение энергия/энтропия:")
    # for level in ratio_stats.keys():
    #     print(f"\nУровень {level}:")
    #     for coeff_type in ['horizontal', 'vertical', 'diagonal']:
    #         ratio_info = ratio_stats[level][coeff_type]
    #         print(f"  {coeff_type}: соотношение={ratio_info['energy_entropy_ratio']:.4f}")

    for level in energy_stats.keys():
        print(energy_stats[level]['std_approx'], 
              energy_stats[level]['details']['horizontal']['raw_energy'],
              energy_stats[level]['details']['horizontal']['std'], 
              energy_stats[level]['details']['vertical']['raw_energy'], 
              energy_stats[level]['details']['vertical']['std'], 
              energy_stats[level]['details']['diagonal']['raw_energy'], 
              energy_stats[level]['details']['diagonal']['std'], 
              entropy_stats['approx'][level - 1]['corrected_entropy'], 
              entropy_stats['horizontal'][level - 1]['corrected_entropy'], 
              entropy_stats['vertical'][level - 1]['corrected_entropy'], 
              entropy_stats['diagonal'][level - 1]['corrected_entropy']
              )

    plot_corrected_analysis(energy_stats, entropy_stats)

    return approx_levels, detail_levels, energy_stats, entropy_stats


if __name__ == "__main__":
    image_path = "SRLabImages/GZBMP24/giraffe5.bmp"
    approx, details, energy_stats, entropy_stats = process_image_corrected(image_path)
