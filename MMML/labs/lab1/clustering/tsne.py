#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
t-SNE 3D кластеризация по траекториям (каждый файл = одна 3D траектория)
Полный скрипт: загрузка всех файлов, ресемпл, масштабирование, (PCA), t-SNE 3D, KMeans для списка k,
метрики (silhouette, avg_intra_cluster_dist), визуализации:
 - 3D: истинные метки
 - 3D: кластеры (best by silhouette или первый)
 - 3D: все варианты кластеризации для каждого k из k_list
 - График метрик: silhouette (левая ось) и avg_intra_cluster_dist (правая ось)
"""

import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# -------------------- Вспомогательные функции --------------------

def find_activity_folder(dataset_path, activity_name):
    for n in os.listdir(dataset_path):
        if n.lower() == activity_name.lower():
            return os.path.join(dataset_path, n)
    return os.path.join(dataset_path, activity_name)


def load_trajectories_per_file_all(dataset_path, activities, convert_to_g=True):
    """
    Загружает ВСЕ .txt файлы из каждой папки активности (каждый файл = отдельная 3D траектория).
    Возвращает:
        trajs: list of np.array (n_points_i x 3)
        names: list of "activity/filename"
        labels: np.array class_id для каждого файла
    """
    trajs = []
    names = []
    labels = []

    for class_id, activity in enumerate(activities):
        folder = find_activity_folder(dataset_path, activity)
        if not os.path.exists(folder):
            print(f"Warning: папка активности не найдена: {folder}")
            continue
        files = sorted([f for f in os.listdir(folder) if f.endswith('.txt')])
        for fname in files:
            path = os.path.join(folder, fname)
            try:
                arr = np.loadtxt(path)
            except Exception as e:
                print(f"Ошибка чтения {path}: {e}")
                continue
            if arr.ndim == 1:
                # если одна строка — преобразуем в (n,3)
                arr = arr.reshape(-1, 3)
            # приведение к "реальным" значениям, как в вашем коде
            real = -1.5 + (arr / 63.0) * 3.0
            if convert_to_g:
                real = real * 9.81
            if real.shape[1] != 3:
                print(f"Пропускаю {path}: unexpected columns {real.shape}")
                continue
            trajs.append(real)
            names.append(f"{activity}/{fname}")
            labels.append(class_id)
    return trajs, names, np.array(labels, dtype=int)


def resample_trajectory(traj, length=100):
    """
    Линейная интерполяция каждой координаты, возвращает траекторию length x 3.
    Если траектория короче 2 точек — дополняет нулями (как fallback).
    """
    n = traj.shape[0]
    if n == length:
        return traj.copy()
    if n < 2:
        pad = np.zeros((length, 3))
        pad[:n, :] = traj
        return pad
    old = np.linspace(0, 1, n)
    new = np.linspace(0, 1, length)
    out = np.zeros((length, 3))
    for d in range(3):
        out[:, d] = np.interp(new, old, traj[:, d])
    return out


def build_flattened_traj_matrix(trajs, traj_len=100):
    """
    Преобразует список траекторий в матрицу (n_files, traj_len*3)
    """
    flat = []
    for t in trajs:
        r = resample_trajectory(t, traj_len)
        flat.append(r.flatten())
    if len(flat) == 0:
        return np.empty((0, traj_len * 3))
    return np.vstack(flat)


def choose_perplexity(n_samples, requested=30):
    """
    Подбор perplexity для t-SNE так, чтобы было < n_samples.
    Возвращаем float.
    """
    if n_samples <= 3:
        return 1.0
    p = min(requested, max(2, n_samples // 3))
    p = min(p, n_samples - 1)
    return float(p)


def run_tsne_with_pca(X, tsne_dims=3, tsne_perplexity=30, random_state=42,
                      pca_before_tsne=True, pca_components=50, verbose=1):
    """
    PCA (опционально) -> t-SNE (3D).
    Возвращает X_tsne (n_samples, tsne_dims), модель t-SNE и модель PCA (или None).
    """
    X_proc = X
    pca_model = None
    if pca_before_tsne and X.shape[1] > pca_components:
        pca_model = PCA(n_components=pca_components, random_state=random_state)
        X_proc = pca_model.fit_transform(X)
        print(f"PCA перед t-SNE: reduced features {X.shape[1]} -> {X_proc.shape[1]}")
    n = X_proc.shape[0]
    if n < 2:
        raise RuntimeError("Недостаточно образцов для t-SNE (нужно >=2 файлов).")
    per = choose_perplexity(n, requested=tsne_perplexity)
    print(f"Запускаем t-SNE: n_samples={n}, perplexity={per}, dims={tsne_dims}")
    ts = TSNE(n_components=tsne_dims, perplexity=per, random_state=random_state,
              init='pca', learning_rate='auto', verbose=verbose)
    X_tsne = ts.fit_transform(X_proc)
    return X_tsne, ts, pca_model


def run_kmeans(X, n_clusters=8, batch_size=1024, random_state=42):
    """
    MiniBatchKMeans, возвращает метки и модель.
    """
    m = MiniBatchKMeans(n_clusters=int(n_clusters), batch_size=batch_size, random_state=random_state)
    labels = m.fit_predict(X)
    return labels, m


def cluster_metrics_emb(X_emb, labels):
    """
    Вычисляет silhouette (если возможно) и среднее внутрикластерное расстояние до центроидов.
    Возвращает dict с ключами:
      'n_clusters', 'total_points', 'silhouette', 'avg_intra_cluster_dist'
    """
    unique = np.unique(labels)
    cluster_ids = [c for c in unique if c != -1]
    n_clusters = len(cluster_ids)
    n_total = X_emb.shape[0]

    # silhouette
    try:
        mask_valid = (labels != -1)
        if np.sum(mask_valid) > 1 and len(np.unique(labels[mask_valid])) > 1:
            sil = float(silhouette_score(X_emb[mask_valid], labels[mask_valid]))
        else:
            sil = float('nan')
    except Exception:
        sil = float('nan')

    sizes = {}
    intra_avgs = []
    for c in cluster_ids:
        pts = X_emb[labels == c]
        if pts.shape[0] == 0:
            sizes[c] = 0
            intra_avgs.append(0.0)
            continue
        centroid = pts.mean(axis=0)
        dists = np.linalg.norm(pts - centroid, axis=1)
        sizes[c] = len(pts)
        intra_avgs.append(dists.mean() if len(dists) > 0 else 0.0)

    if len(intra_avgs) > 0:
        weights = np.array([sizes[c] for c in cluster_ids], dtype=float)
        avg_intra = float(np.average(intra_avgs, weights=weights))
    else:
        avg_intra = float('nan')

    return {
        'n_clusters': n_clusters,
        'total_points': n_total,
        'silhouette': None if math.isnan(sil) else sil,
        'avg_intra_cluster_dist': avg_intra
    }


# -------------------- Основная функция: t-SNE 3D + кластеризация + визуализация --------------------

def tsne3d_cluster_all_trajectories_with_plots(dataset_path, activities,
                                               traj_len=100,
                                               convert_to_g=True,
                                               scaler_type='standard',
                                               pca_before_tsne=True,
                                               pca_components=50,
                                               tsne_dims=3,
                                               tsne_perplexity=30,
                                               random_state=42,
                                               k_list=[4, 8, 14],
                                               plot=True,
                                               save_results=False,
                                               out_prefix="tsne3d_cluster_results"):
    """
    Полная функция, возвращает словарь с результатами.
    Визуализация:
      - 3D: истинные метки
      - 3D: лучшие кластеры (по silhouette) или первый
      - 3D: все k из k_list (динамическая сетка)
      - график метрик silhouette (линия) и avg_intra_cluster_dist (столбцы, правая ось)
    """
    print("Загрузка всех траекторий...")
    trajs, names, true_labels = load_trajectories_per_file_all(dataset_path, activities, convert_to_g=convert_to_g)
    n = len(trajs)
    print(f"Всего файлов-траекторий: {n}")
    if n == 0:
        raise RuntimeError("Нет файлов для обработки. Проверьте dataset_path и activities.")

    # Построим матрицу признаков (каждый файл -> ресемпл * 3 -> flatten)
    X_flat = build_flattened_traj_matrix(trajs, traj_len=traj_len)
    print(f"Размер матрицы признаков: {X_flat.shape} (n_files x traj_len*3)")

    # Масштабирование
    scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
    X_scaled = scaler.fit_transform(X_flat)

    # PCA перед t-SNE и запуск t-SNE (3D)
    try:
        X_tsne, tsne_model, pca_model = run_tsne_with_pca(X_scaled, tsne_dims=tsne_dims,
                                                          tsne_perplexity=tsne_perplexity,
                                                          random_state=random_state,
                                                          pca_before_tsne=pca_before_tsne,
                                                          pca_components=pca_components,
                                                          verbose=1)
    except Exception as e:
        # попробовать уменьшить perplexity — часто помогает в случаях очень малого числа файлов
        print(f"Ошибка при запуске t-SNE с perplexity={tsne_perplexity}: {e}")
        per = choose_perplexity(n, requested=tsne_perplexity)
        print(f"Пробуем уменьшенный perplexity = {per}")
        X_tsne, tsne_model, pca_model = run_tsne_with_pca(X_scaled, tsne_dims=tsne_dims,
                                                          tsne_perplexity=per,
                                                          random_state=random_state,
                                                          pca_before_tsne=pca_before_tsne,
                                                          pca_components=pca_components,
                                                          verbose=1)

    results = []
    best_by_silhouette = None

    # Кластеризация по каждому k
    for k in k_list:
        if k <= 0 or k > n:
            print(f"Пропускаю k={k} (неподходящее для n={n}).")
            continue
        labels_k, kmodel = run_kmeans(X_tsne, n_clusters=k, random_state=random_state)
        met = cluster_metrics_emb(X_tsne, labels_k)
        met.update({'k': k})
        results.append({'k': k, 'metrics': met, 'labels': labels_k, 'kmodel': kmodel})
        print(
            f"[k={k}] silhouette={met['silhouette']}, avg_intra_cluster_dist={met['avg_intra_cluster_dist']}, n_clusters_reported={met['n_clusters']}")

        # обновить лучший вариант по silhouette, если есть значение
        if met['silhouette'] is not None:
            if best_by_silhouette is None or (met['silhouette'] > best_by_silhouette['metrics']['silhouette']):
                best_by_silhouette = {'k': k, 'metrics': met, 'labels': labels_k}

    # Если не было валидных результатов (например k_list пуст) — попробуем k=1 fallback
    if len(results) == 0:
        print("Не найдено валидных k в k_list — выполняется KMeans с k=1")
        labels_k = np.zeros(n, dtype=int)
        met = cluster_metrics_emb(X_tsne, labels_k)
        met.update({'k': 1})
        results.append({'k': 1, 'metrics': met, 'labels': labels_k, 'kmodel': None})
        best_by_silhouette = {'k': 1, 'metrics': met, 'labels': labels_k}

    # --------------- ВИЗУАЛИЗАЦИЯ ---------------

    if plot:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        # 1) Большая фигура: слева — истинные метки, справа — метки лучшего k (или первого)
        fig = plt.figure(figsize=(16, 7))
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        sc1 = ax1.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2],
                          c=true_labels, s=30, cmap='tab20')
        ax1.set_title("t-SNE 3D — истинные метки (activity classes)")
        ax1.set_xlabel("TSNE1");
        ax1.set_ylabel("TSNE2");
        ax1.set_zlabel("TSNE3")
        try:
            handles, labels_legend = sc1.legend_elements()
            ax1.legend(handles, labels_legend, title="classes", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        except Exception:
            pass

        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        if best_by_silhouette is not None:
            labels_plot = best_by_silhouette['labels']
            title = f"KMeans on t-SNE 3D (best k={best_by_silhouette['k']}, sil={best_by_silhouette['metrics']['silhouette']:.4f})"
        else:
            labels_plot = results[0]['labels']
            title = f"KMeans on t-SNE 3D (k={results[0]['k']})"
        sc2 = ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2],
                          c=labels_plot, s=30, cmap='tab20')
        ax2.set_title(title)
        ax2.set_xlabel("TSNE1");
        ax2.set_ylabel("TSNE2");
        ax2.set_zlabel("TSNE3")
        try:
            handles2, labels2 = sc2.legend_elements()
            ax2.legend(handles2, labels2, title="clusters", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        except Exception:
            pass

        plt.tight_layout()
        plt.show()

        # 2) Отдельная фигура: отображение меток кластеров для каждого k из k_list (динамическая сетка)
        # Определяем сколько подграфиков нужно
        valid_results = [r for r in results]
        n_plots = len(valid_results)
        if n_plots > 0:
            # Grid layout: columns up to 3 для читаемости
            cols = 3
            rows = (n_plots + cols - 1) // cols
            fig2 = plt.figure(figsize=(5 * cols, 4 * rows))
            for idx, r in enumerate(valid_results):
                ax = fig2.add_subplot(rows, cols, idx + 1, projection='3d')
                labels_k = r['labels']
                sc = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2],
                                c=labels_k, s=18, cmap='tab20')
                ax.set_title(
                    f"k={r['k']}  sil={r['metrics']['silhouette'] if r['metrics']['silhouette'] is not None else 'N/A'}")
                ax.set_xlabel("TSNE1");
                ax.set_ylabel("TSNE2");
                ax.set_zlabel("TSNE3")
                # Попытка поставить легенду (если не слишком много кластеров)
                try:
                    handles_k, labels_k_legend = sc.legend_elements()
                    ax.legend(handles_k, labels_k_legend, title="clusters", loc='upper left', fontsize=7)
                except Exception:
                    pass
            plt.tight_layout()
            plt.show()

        # 3) График метрик silhouette и avg_intra_cluster_dist по k
        ks = [r['k'] for r in results]
        silhouettes = [r['metrics']['silhouette'] if r['metrics']['silhouette'] is not None else float('nan') for r in
                       results]
        avgs = [r['metrics']['avg_intra_cluster_dist'] for r in results]

        fig3, ax_left = plt.subplots(figsize=(10, 5))
        ax_right = ax_left.twinx()

        # Bar for avg_intra, line for silhouette
        bar_positions = np.arange(len(ks))
        bar_width = 0.6
        ax_right.bar(bar_positions, avgs, width=bar_width, alpha=0.6, label='Avg intra-cluster dist', zorder=1)
        ax_left.plot(bar_positions, silhouettes, marker='o', linewidth=2, label='Silhouette', color='tab:blue',
                     zorder=2)

        # Labels and ticks
        ax_left.set_xticks(bar_positions)
        ax_left.set_xticklabels([str(k) for k in ks])
        ax_left.set_xlabel('k (number of clusters)')
        ax_left.set_ylabel('Silhouette score')
        ax_right.set_ylabel('Avg intra-cluster distance')

        ax_left.set_title('Metrics across different k')
        # Legends
        left_legend = ax_left.legend(loc='upper left')
        right_legend = ax_right.legend(loc='upper right')
        # grid
        ax_left.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()

    # --------------- Сохранение результатов (опционально) ---------------
    if save_results:
        import csv
        csv_path = f"{out_prefix}_labels.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'true_label'])
            for nm, tl in zip(names, true_labels):
                writer.writerow([nm, tl])
        np.savez(f"{out_prefix}_data.npz", X_tsne=X_tsne, names=np.array(names), true_labels=true_labels)
        # Сохранить метки кластеров по каждому k в отдельном CSV
        for r in results:
            k = r['k']
            labels_file = f"{out_prefix}_k{k}_labels.csv"
            with open(labels_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['name', 'true_label', 'cluster_label'])
                for nm, tl, cl in zip(names, true_labels, r['labels']):
                    writer.writerow([nm, tl, int(cl)])
        print(f"Saved names+true_labels -> {csv_path} and embedding -> {out_prefix}_data.npz and per-k label files.")

    # --------------- Возвращаем результаты ---------------
    return {
        'X_tsne': X_tsne,
        'tsne_model': tsne_model,
        'pca_model': pca_model,
        'results': results,
        'best_by_silhouette': best_by_silhouette,
        'names': names,
        'true_labels': true_labels,
        'X_flat': X_flat,
        'scaler': scaler,
    }


# -------------------- Пример использования --------------------

if __name__ == "__main__":
    # Фиксируем сиды для воспроизводимости
    random.seed(42)
    np.random.seed(42)

    # Укажите путь к датасету и список активностей (папок)
    dataset_path = r'../HMP_Dataset'  # <- поменяйте на ваш путь
    activities = ['brush_teeth', 'climb_stairs', 'comb_hair', 'descend_stairs', 'drink_glass',
                  'eat_meat', 'eat_soup', 'getup_bed', 'liedown_bed', 'pour_water',
                  'sitdown_chair', 'standup_chair', 'use_telephone', 'walk']

    # Параметры
    traj_len = 100  # длина ресемплированной траектории
    convert_to_g = True
    scaler_type = 'standard'  # 'standard' или 'minmax'
    pca_before_tsne = True
    pca_components = 50
    tsne_dims = 3
    tsne_perplexity = 30
    random_state = 42
    k_list = [5, 10, 15, 25]  # список k для KMeans
    plot = True
    save_results = False  # True для сохранения CSV / NPZ с результатами
    out_prefix = "tsne3d_allfiles"

    try:
        out = tsne3d_cluster_all_trajectories_with_plots(
            dataset_path=dataset_path,
            activities=activities,
            traj_len=traj_len,
            convert_to_g=convert_to_g,
            scaler_type=scaler_type,
            pca_before_tsne=pca_before_tsne,
            pca_components=pca_components,
            tsne_dims=tsne_dims,
            tsne_perplexity=tsne_perplexity,
            random_state=random_state,
            k_list=k_list,
            plot=plot,
            save_results=save_results,
            out_prefix=out_prefix
        )

        print("\n--- РЕЗУЛЬТАТЫ ---")
        for r in out['results']:
            m = r['metrics']
            print(
                f"k={r['k']}: silhouette={m['silhouette']}, avg_intra_cluster_dist={m['avg_intra_cluster_dist']}, reported_clusters={m['n_clusters']}")

        if out['best_by_silhouette'] is not None:
            b = out['best_by_silhouette']
            print(
                f"\nЛучший по silhouette: k={b['k']}, silhouette={b['metrics']['silhouette']}, avg_intra={b['metrics']['avg_intra_cluster_dist']}")
        else:
            print("\nНет валидного silhouette для выбора лучшего варианта.")

    except Exception as e:
        print(f"Ошибка: {e}")
