import csv
import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

try:
    from sklearn_som.som import SOM
except Exception:
    SOM = None

try:
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean

    HAVE_FASTDTW = True
except Exception:
    HAVE_FASTDTW = False



def find_activity_folder(dataset_path, activity_name):
    for n in os.listdir(dataset_path):
        if n.lower() == activity_name.lower() and os.path.isdir(os.path.join(dataset_path, n)):
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



def run_som_sklearn_som(X, m, n, epochs=20, lr=0.5, sigma=None):
    if SOM is None:
        raise ImportError("sklearn-som not found. Install: pip install sklearn-som")
    if sigma is None:
        sigma = max(m, n) / 2.0
    som = SOM(m=m, n=n, dim=X.shape[1], lr=lr, sigma=sigma, max_iter=1000)
    som.fit(X, epochs=int(epochs), shuffle=True)
    labels_raw = som.predict(X)

    labels_raw = np.asarray(labels_raw)
    if labels_raw.ndim == 1:
        labels_flat = labels_raw.astype(int)
    elif labels_raw.ndim == 2:
        coords = [tuple(map(int, row)) for row in labels_raw]
        unique_coords = {}
        idx = 0
        labels_flat = np.zeros(len(coords), dtype=int)
        for i, c in enumerate(coords):
            if c not in unique_coords:
                unique_coords[c] = idx
                idx += 1
            labels_flat[i] = unique_coords[c]
    else:
        labels_flat = np.arange(len(labels_raw), dtype=int)

    return labels_flat, som



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
    if n == 0:
        return np.empty((0, 0)), []
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



def load_all_activities(dataset_path, activities=None, use_trajectory_features=True, window_size=50):
    all_features = []
    true_labels = []
    activity_map = {}
    global_idx = 0
    if activities is None:
        activities = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])

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
        return np.empty((0, 3)), np.empty(0), {}

    X = np.vstack(all_features)
    true_labels = np.array(true_labels)
    return X, true_labels, activity_map



def analyze_dataset_with_som_expanded(dataset_path,
                                      activities=None,
                                      use_trajectory_features=True,
                                      scaler_type='standard',
                                      pca_components_for_clustering=10,
                                      pca_components_vis=3,
                                      max_points=50000,
                                      som_grid=[(6, 6), (8, 8)],
                                      som_epochs=[10, 30],
                                      window_size=50,
                                      random_state=42,
                                      plot=True,
                                      save_results=False,
                                      out_prefix="som_expanded"):
    print("\n=== SOM expanded analysis ===")
    X, true_labels, activity_map = load_all_activities(dataset_path, activities, use_trajectory_features, window_size)
    if X.size == 0:
        print("No data")
        return

    # print(f"Total points: {X.shape[0]} (will sample up to {max_points})")
    X_sampled, sample_idx = sample_points(X, max_points)
    true_labels_sampled = true_labels[sample_idx]

    scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
    X_scaled = scaler.fit_transform(X_sampled)

    X_red, pca_clust = reduce_dim(X_scaled, n_components=pca_components_for_clustering)

    som_results = []

    if SOM is None:
        print("SOM unavailable (sklearn-som not installed); skipping SOM experiments.")
    else:
        for (m, n) in som_grid:
            for e in som_epochs:
                try:
                    print(f"Running SOM {m}x{n}, epochs={e} ...")
                    labels_raw, som_model = run_som_sklearn_som(X_red, m=m, n=n, epochs=e, lr=0.5, sigma=None)
                    labels = np.asarray(labels_raw, dtype=int)
                    met = cluster_metrics(X_red, labels)
                    met.update({'map_m': m, 'map_n': n, 'epochs': e})
                    som_results.append(
                        {'m': m, 'n': n, 'epochs': e, 'labels': labels, 'model': som_model, 'metrics': met})
                    print(
                        f"  -> silhouette={met['silhouette']}, avg_intra={met['avg_intra_cluster_dist']}, clusters={met['n_clusters']}")
                except Exception as ex:
                    print(f"SOM error {m}x{n}, epochs={e}: {ex}")

    valid = [r for r in som_results if r['metrics']['silhouette'] is not None]
    if len(valid) > 0:
        best = max(valid, key=lambda it: it['metrics']['silhouette'])
        best_by_silhouette = best
    elif len(som_results) > 0:
        best = max(som_results, key=lambda it: it['metrics']['n_clusters'])
        best_by_silhouette = best
    else:
        best_by_silhouette = None

    if pca_components_vis is None or pca_components_vis < 3:
        pca_vis = PCA(n_components=3, random_state=random_state)
    else:
        pca_vis = PCA(n_components=pca_components_vis, random_state=random_state)
    X_vis = pca_vis.fit_transform(X_scaled)

    if plot:
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(18, 12))
        ax_true = fig.add_subplot(2, 2, 1, projection='3d')
        ax_best = fig.add_subplot(2, 2, 3, projection='3d')
        ax_metrics = fig.add_subplot(2, 2, 4)

        sc_true = ax_true.scatter(X_vis[:, 0], X_vis[:, 1], X_vis[:, 2],
                                  c=true_labels_sampled, s=6, cmap='tab20')
        ax_true.set_title("True class labels (PCA 3D)")
        ax_true.set_xlabel("PC1");
        ax_true.set_ylabel("PC2");
        ax_true.set_zlabel("PC3")
        try:
            handles_t, labels_t = sc_true.legend_elements()
            ax_true.legend(handles_t, labels_t, title="classes", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        except Exception:
            pass

        if best_by_silhouette is not None:
            lbls_best = best_by_silhouette['labels']
            ax_best.scatter(X_vis[:, 0], X_vis[:, 1], X_vis[:, 2], c=lbls_best, s=6, cmap='tab20')
            m = best_by_silhouette['m'];
            n = best_by_silhouette['n'];
            e = best_by_silhouette['epochs']
            met = best_by_silhouette['metrics']
            ax_best.set_title(
                f"SOM best ({m}x{n}, epochs={e})\nclusters={met['n_clusters']}, sil={met['silhouette']}, avg_intra={met['avg_intra_cluster_dist']:.4f}")
            ax_best.set_xlabel("PC1");
            ax_best.set_ylabel("PC2");
            ax_best.set_zlabel("PC3")
        else:
            ax_best.text(0.5, 0.5, 0.5, "SOM not available/failed", horizontalalignment='center')

        n_plots = len(som_results)
        if n_plots > 0:
            cols = 3
            rows = (n_plots + cols - 1) // cols
            fig2 = plt.figure(figsize=(5 * cols, 4 * rows))
            for idx, r in enumerate(som_results):
                ax = fig2.add_subplot(rows, cols, idx + 1, projection='3d')
                labels_k = r['labels']
                ax.scatter(X_vis[:, 0], X_vis[:, 1], X_vis[:, 2], c=labels_k, s=10, cmap='tab20')
                ax.set_title(
                    f"{r['m']}x{r['n']} e={r['epochs']} sil={r['metrics']['silhouette'] if r['metrics']['silhouette'] is not None else 'N/A'}")
                ax.set_xlabel("PC1");
                ax.set_ylabel("PC2");
                ax.set_zlabel("PC3")
            plt.tight_layout()
            plt.show()

        if n_plots > 0:
            labels_for_x = [f"{r['m']}x{r['n']}_e{r['epochs']}" for r in som_results]
            silhouettes = [r['metrics']['silhouette'] if r['metrics']['silhouette'] is not None else float('nan') for r
                           in som_results]
            avgs = [r['metrics']['avg_intra_cluster_dist'] for r in som_results]

            fig3, ax_left = plt.subplots(figsize=(12, 5))
            ax_right = ax_left.twinx()

            x = np.arange(len(labels_for_x))
            width = 0.6
            ax_right.bar(x, avgs, width=width, alpha=0.6, label='Avg intra-cluster dist', zorder=1)
            ax_left.plot(x, silhouettes, marker='o', linewidth=2, label='Silhouette', color='tab:blue', zorder=2)

            ax_left.set_xticks(x)
            ax_left.set_xticklabels(labels_for_x, rotation=45, ha='right', fontsize=8)
            ax_left.set_xlabel('SOM configuration')
            ax_left.set_ylabel('Silhouette score')
            ax_right.set_ylabel('Avg intra-cluster distance')
            ax_left.set_title('SOM metrics across configurations')

            l1 = ax_left.get_legend_handles_labels()
            l2 = ax_right.get_legend_handles_labels()
            ax_left.legend(l1[0] + l2[0], l1[1] + l2[1], loc='upper right')

            ax_left.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.show()

        else:
            print("No SOM results to plot metrics for.")

    if save_results and len(som_results) > 0:
        summary_csv = f"{out_prefix}_summary.csv"
        with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(
                ['map_m', 'map_n', 'epochs', 'n_clusters', 'total_points', 'silhouette', 'avg_intra_cluster_dist'])
            for r in som_results:
                m = r['m'];
                n = r['n'];
                e = r['epochs']
                met = r['metrics']
                writer.writerow([m, n, e, met.get('n_clusters'), met.get('total_points'), met.get('silhouette'),
                                 met.get('avg_intra_cluster_dist')])
        print(f"Saved SOM summary -> {summary_csv}")

        for r in som_results:
            m = r['m'];
            n = r['n'];
            e = r['epochs']
            fname = f"{out_prefix}_{m}x{n}_e{e}_labels.csv"
            with open(fname, 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(['index', 'label'])
                for idx, lab in enumerate(r['labels']):
                    w.writerow([idx, int(lab)])
        np.savez(f"{out_prefix}_data.npz", X_sampled=X_sampled, X_vis=X_vis, true_labels=true_labels_sampled)
        print(f"Saved labels files and embedding -> {out_prefix}_data.npz")

    print("\n--- SOM experiments summary ---")
    if len(som_results) == 0:
        print("No SOM results (SOM not installed or all runs failed).")
    else:
        for r in som_results:
            m = r['m'];
            n = r['n'];
            e = r['epochs']
            met = r['metrics']
            print(
                f"{m}x{n}, e={e} -> clusters={met['n_clusters']}, silhouette={met['silhouette']}, avg_intra={met['avg_intra_cluster_dist']}")

    if best_by_silhouette is None:
        print("\nBest configuration: none")
    else:
        m = best_by_silhouette['m'];
        n = best_by_silhouette['n'];
        e = best_by_silhouette['epochs']
        met = best_by_silhouette['metrics']
        print(
            f"\nBest by silhouette: {m}x{n}, epochs={e} -> silhouette={met['silhouette']}, avg_intra={met['avg_intra_cluster_dist']}, clusters={met['n_clusters']}")

    return {
        'X_sampled': X_sampled,
        'sample_idx': sample_idx,
        'true_labels_sampled': true_labels_sampled,
        'pca_clust': pca_clust,
        'pca_vis': pca_vis,
        'som_results': som_results,
        'best_by_silhouette': best_by_silhouette,
        'activity_map': activity_map,
        'scaler': scaler
    }


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    dataset_path = r'../HMP_Dataset'
    activities = None
    try:
        out = analyze_dataset_with_som_expanded(
            dataset_path=dataset_path,
            activities=activities,
            use_trajectory_features=True,
            scaler_type='mixmax',
            pca_components_for_clustering=25,
            pca_components_vis=3,
            # max_points=10000,
            som_grid=[(5, 5), (10, 10), (25, 25)],
            som_epochs=[25, 50, 100],
            window_size=100,
            random_state=104,
            plot=True,
            save_results=False,
            out_prefix="som_expanded"
        )
    except Exception as e:
        print(f"Error processing dataset: {e}")

    print("All done.")
