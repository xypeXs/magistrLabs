import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

try:
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean

    HAVE_FASTDTW = True
except Exception:
    HAVE_FASTDTW = False


def find_activity_folder(dataset_path, activity_name):
    for n in os.listdir(dataset_path):
        if n.lower() == activity_name.lower():
            return os.path.join(dataset_path, n)
    return os.path.join(dataset_path, activity_name)


def load_files_as_points(dataset_path, activity_folder, convert_to_g=True):
    activity_path = find_activity_folder(dataset_path, activity_folder)
    if not os.path.exists(activity_path):
        raise FileNotFoundError(f"{activity_path} не найден")
    files = sorted([f for f in os.listdir(activity_path) if f.endswith('.txt')])
    all_points = []
    file_index_map = []
    idx = 0
    for fname in files:
        arr = np.loadtxt(os.path.join(activity_path, fname))
        if arr.ndim == 1:
            arr = arr.reshape(-1, 3)
        real = -1.5 + (arr / 63.0) * 3.0
        if convert_to_g:
            real = real * 9.81
        n = real.shape[0]
        all_points.append(real)
        file_index_map.append((fname, idx, idx + n))
        idx += n
    if len(all_points) == 0:
        return np.empty((0, 3)), []
    return np.vstack(all_points), file_index_map


def build_trajectory_features(points, window_size=50):
    features = []
    n_points = points.shape[0]

    for i in range(0, n_points, window_size):
        end_idx = min(i + window_size, n_points)
        window = points[i:end_idx]

        if len(window) < 2:
            continue

        magnitude = np.linalg.norm(window, axis=1)

        window_features = []

        window_features.extend(np.mean(window, axis=0))
        window_features.extend(np.std(window, axis=0))
        window_features.extend(np.median(window, axis=0))

        window_features.append(np.mean(magnitude))
        window_features.append(np.std(magnitude))
        window_features.append(np.max(magnitude))
        window_features.append(np.min(magnitude))

        diff = np.diff(window, axis=0)
        if len(diff) > 0:
            window_features.extend(np.mean(diff, axis=0))
            window_features.extend(np.std(diff, axis=0))

            diff_magnitude = np.linalg.norm(diff, axis=1)
            window_features.append(np.mean(diff_magnitude))
            window_features.append(np.std(diff_magnitude))
        else:
            window_features.extend([0, 0, 0, 0, 0, 0, 0, 0])

        features.append(window_features)

    if len(features) == 0:
        return np.empty((0, 20))

    return np.array(features)


def sample_points(X, max_points):
    n = X.shape[0]
    # if n <= max_points:
    return X, np.arange(n)
    # idx = np.random.choice(n, size=max_points, replace=False)
    # idx.sort()
    # return X[idx], idx


def reduce_dim(X, n_components=10):
    if n_components >= X.shape[1]:
        return X, None
    pca = PCA(n_components=n_components, random_state=42)
    Xr = pca.fit_transform(X)
    return Xr, pca


def run_kmeans(X, n_clusters=8, batch_size=1024, random_state=42):
    m = MiniBatchKMeans(n_clusters=int(n_clusters), batch_size=batch_size, random_state=random_state)
    labels = m.fit_predict(X)
    return labels, m


def cluster_metrics(X, labels):
    unique = np.unique(labels)
    n_total = X.shape[0]
    cluster_ids = [c for c in unique if c != -1]
    n_clusters = len(cluster_ids)

    try:
        mask = labels != -1
        if np.sum(mask) > 1 and len(np.unique(labels[mask])) > 1:
            sil = silhouette_score(X[mask], labels[mask])
        else:
            sil = float('nan')
    except Exception:
        sil = float('nan')

    centroids = {}
    sizes = {}
    intra_avgs = []
    for c in cluster_ids:
        pts = X[labels == c]
        centroid = pts.mean(axis=0)
        dists = np.linalg.norm(pts - centroid, axis=1)
        centroids[c] = centroid
        sizes[c] = len(pts)
        intra_avgs.append(dists.mean() if len(dists) > 0 else 0.0)
    avg_intra = float(np.average(intra_avgs, weights=[sizes[c] for c in cluster_ids])) if len(
        intra_avgs) > 0 else float('nan')

    return {
        'n_clusters': n_clusters,
        'total_points': n_total,
        'silhouette': None if math.isnan(sil) else float(sil),
        'avg_intra_cluster_dist': avg_intra
    }


def resample_trajectory(traj, length=100):
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


def compute_traj_distance_matrix(dataset_path, activities, max_files_per_activity=5, traj_len=100):
    feats = []
    names = []
    for activity in activities:
        folder = find_activity_folder(dataset_path, activity)
        file_names = sorted([f for f in os.listdir(folder) if f.endswith('.txt')])[:max_files_per_activity]
        for fname in file_names:
            arr = np.loadtxt(os.path.join(folder, fname))
            if arr.ndim == 1:
                arr = arr.reshape(-1, 3)
            real = -1.5 + (arr / 63.0) * 3.0
            traj = resample_trajectory(real, length=traj_len)
            feats.append(traj)
            names.append(f"{activity}/{fname}")
    n = len(feats)
    D = np.zeros((n, n))
    if HAVE_FASTDTW:
        for i in range(n):
            for j in range(i + 1, n):
                d, _ = fastdtw(feats[i], feats[j], dist=euclidean)
                D[i, j] = D[j, i] = d
    else:
        flat = np.array([f.flatten() for f in feats])
        D = pairwise_distances(flat)
    return D, names


def load_all_activities(dataset_path, activities, use_trajectory_features=True, window_size=50):
    all_features = []
    true_labels = []
    activity_map = {}
    global_idx = 0

    for class_id, activity in enumerate(activities):
        points, file_map = load_files_as_points(dataset_path, activity)
        if points.size == 0:
            continue

        if use_trajectory_features:
            features = build_trajectory_features(points, window_size=window_size)
        else:
            features = points

        n_features = features.shape[0]
        activity_map[activity] = (global_idx, global_idx + n_features, class_id)
        all_features.append(features)
        true_labels.extend([class_id] * n_features)
        global_idx += n_features

    if len(all_features) == 0:
        return np.empty((0, 20)), np.empty(0), {}

    X = np.vstack(all_features)
    true_labels = np.array(true_labels)
    return X, true_labels, activity_map


def analyze_dataset_and_plot_kmeans_expanded(dataset_path, activities,
                                             use_trajectory_features=True,
                                             scaler_type='standard',
                                             pca_components_for_clustering=10,
                                             pca_components_vis=3,
                                             max_points=50000,
                                             kmeans_list=[4, 8, 14],
                                             window_size=50,
                                             random_state=42,
                                             plot=True,
                                             save_results=False,
                                             out_prefix="kmeans_expanded"):
    print("\n=== KMeans expanded analysis ===")
    X, true_labels, activity_map = load_all_activities(dataset_path, activities,
                                                       use_trajectory_features, window_size)
    if X.size == 0:
        print("No data")
        return

    # print(f"Total points: {X.shape[0]} (will sample up to {max_points})")
    X_sampled, sample_idx = sample_points(X, max_points)
    true_labels_sampled = true_labels[sample_idx]

    scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
    X_scaled = scaler.fit_transform(X_sampled)

    X_red, pca_clust = reduce_dim(X_scaled, n_components=pca_components_for_clustering)

    alg_results = {'KMeans': []}
    print("Running KMeans for k values:", kmeans_list)
    for k in kmeans_list:
        try:
            labels, kmodel = run_kmeans(X_red, n_clusters=k, batch_size=1024, random_state=random_state)
            met = cluster_metrics(X_red, labels)
            met.update({'k': k})
            alg_results['KMeans'].append({'met': met, 'labels': labels, 'model': kmodel})
            print(f"Completed KMeans k={k}: silhouette={met['silhouette']}, avg_intra={met['avg_intra_cluster_dist']}")
        except Exception as e:
            print(f"Error running KMeans for k={k}: {e}")

    best_per_alg = {}
    for alg, lst in alg_results.items():
        if len(lst) == 0:
            best_per_alg[alg] = None
            continue
        valid = [item for item in lst if item['met'].get('silhouette') is not None and not (
                    isinstance(item['met']['silhouette'], float) and math.isnan(item['met']['silhouette']))]
        if valid:
            best = min(valid, key=lambda it: it['met']['silhouette'])
        else:
            # если silhouette недоступен для всех — выбрать по числу кластеров (максимальному)
            best = max(lst, key=lambda it: it['met']['n_clusters'])
        best_per_alg[alg] = best

    pca_vis = PCA(n_components=3, random_state=random_state)
    X_vis = pca_vis.fit_transform(X_scaled)

    if plot:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig1 = plt.figure(figsize=(14, 6), constrained_layout=True)
        ax_true = fig1.add_subplot(1, 2, 1, projection='3d')
        ax_best = fig1.add_subplot(1, 2, 2, projection='3d')

        sc_true = ax_true.scatter(X_vis[:, 0], X_vis[:, 1], X_vis[:, 2],
                                  c=true_labels_sampled, s=10, cmap='tab20')
        ax_true.set_title("True class labels (PCA 3D)")
        ax_true.set_xlabel("PC1");
        ax_true.set_ylabel("PC2");
        ax_true.set_zlabel("PC3")
        try:
            handles_t, labels_t = sc_true.legend_elements()
            ax_true.legend(handles_t, labels_t, title="classes", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        except Exception:
            pass

        best_kmeans_entry = best_per_alg['KMeans']
        if best_kmeans_entry is not None:
            met_km = best_kmeans_entry['met']
            labels_km = best_kmeans_entry['labels']
            ax_best.scatter(X_vis[:, 0], X_vis[:, 1], X_vis[:, 2], c=labels_km, s=10, cmap='tab20')
            ax_best.set_title(f"KMeans (best k={met_km.get('k')}) clusters={met_km['n_clusters']}\n"
                              f"sil={met_km['silhouette']}, avg_intra={met_km['avg_intra_cluster_dist']:.4f}")
            ax_best.set_xlabel("PC1");
            ax_best.set_ylabel("PC2");
            ax_best.set_zlabel("PC3")
        else:
            ax_best.text(0.5, 0.5, 0.5, "KMeans results absent", horizontalalignment='center')

        plt.show()
        plt.pause(0.001)  # гарантируем отрисовку

        results_list = alg_results['KMeans']
        n_plots = len(results_list)
        if n_plots > 0:
            cols = 3
            rows = (n_plots + cols - 1) // cols
            fig2 = plt.figure(figsize=(5 * cols, 4 * rows), constrained_layout=True)
            for idx, entry in enumerate(results_list):
                met = entry['met']
                labels_k = entry['labels']
                ax = fig2.add_subplot(rows, cols, idx + 1, projection='3d')
                ax.scatter(X_vis[:, 0], X_vis[:, 1], X_vis[:, 2], c=labels_k, s=10, cmap='tab20')
                sil_val = met.get('silhouette')
                sil_str = f"{sil_val:.4f}" if (
                            sil_val is not None and not (isinstance(sil_val, float) and math.isnan(sil_val))) else "N/A"
                ax.set_title(f"k={met.get('k')}  sil={sil_str}")
                ax.set_xlabel("PC1");
                ax.set_ylabel("PC2");
                ax.set_zlabel("PC3")
            plt.show()
            plt.pause(0.001)
        else:
            print("No KMeans results to build per-k grid.")

        if n_plots > 0:
            ks = [entry['met']['k'] for entry in results_list]
            silhouettes = np.array(
                [entry['met']['silhouette'] if entry['met']['silhouette'] is not None else np.nan for entry in
                 results_list], dtype=float)
            avgs = np.array([entry['met']['avg_intra_cluster_dist'] for entry in results_list], dtype=float)

            fig3, ax_left = plt.subplots(figsize=(10, 5), constrained_layout=True)
            ax_right = ax_left.twinx()

            x = np.arange(len(ks))
            bar_width = 0.6
            ax_right.bar(x, avgs, width=bar_width, alpha=0.6, label='Avg intra-cluster dist', zorder=1)
            silhouettes_masked = np.ma.masked_invalid(silhouettes)
            ax_left.plot(x, silhouettes_masked, marker='o', linewidth=2, label='Silhouette', color='tab:blue', zorder=2)

            ax_left.set_xticks(x)
            ax_left.set_xticklabels([str(k) for k in ks])
            ax_left.set_xlabel('k (number of clusters)')
            ax_left.set_ylabel('Silhouette score')
            ax_right.set_ylabel('Avg intra-cluster distance')
            ax_left.set_title('KMeans metrics across k')

            h1, l1 = ax_left.get_legend_handles_labels()
            h2, l2 = ax_right.get_legend_handles_labels()
            handles = h1 + h2
            labels = l1 + l2
            if handles:
                ax_left.legend(handles, labels, loc='upper right')

            ax_left.grid(True, linestyle='--', alpha=0.3)
            plt.show()
            plt.pause(0.001)
        else:
            print("No KMeans metrics to plot.")

    if save_results:
        import csv
        csv_path = f"{out_prefix}_kmeans_summary.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['algorithm', 'k', 'n_clusters', 'total_points', 'silhouette', 'avg_intra_cluster_dist'])
            for alg, lst in alg_results.items():
                for entry in lst:
                    met = entry['met']
                    writer.writerow([alg, met.get('k'), met.get('n_clusters'), met.get('total_points'),
                                     met.get('silhouette'), met.get('avg_intra_cluster_dist')])
        print(f"Saved summary -> {csv_path}")

    print("\nBest configurations (by silhouette if computed):")
    for alg, best in best_per_alg.items():
        if best is None:
            print(f"  {alg}: none")
        else:
            print(f"  {alg}: {best['met']}")

    return {
        'X_sampled': X_sampled,
        'sample_idx': sample_idx,
        'true_labels_sampled': true_labels_sampled,
        'pca_clust': pca_clust,
        'pca_vis': pca_vis,
        'alg_results': alg_results,
        'best_per_alg': best_per_alg,
        'activity_map': activity_map
    }


if __name__ == "__main__":
    dataset_path = r'../HMP_Dataset'
    activities = ['brush_teeth', 'climb_stairs', 'comb_hair', 'descend_stairs', 'drink_glass',
                  'eat_meat', 'eat_soup', 'getup_bed', 'liedown_bed', 'pour_water',
                  'sitdown_chair', 'standup_chair', 'use_telephone', 'walk']
    random.seed(42)
    np.random.seed(42)

    try:
        analyze_dataset_and_plot_kmeans_expanded(
            dataset_path=dataset_path,
            activities=activities,
            use_trajectory_features=True,
            scaler_type='standard',
            pca_components_for_clustering=25,
            pca_components_vis=3,
            # max_points=10000,
            kmeans_list=[5, 10, 15, 25],
            window_size=50,
            random_state=104,
            plot=True,
            save_results=False,
            out_prefix="kmeans_expanded"
        )
    except Exception as e:
        print(f"Error processing dataset: {e}")

    print("All done.")
