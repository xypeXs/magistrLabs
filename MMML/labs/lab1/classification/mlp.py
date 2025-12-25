import os
import random

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import visualize
from utils import get_dirs


def find_activity_folder(dataset_path, activity_name):
    for n in os.listdir(dataset_path):
        if n.lower() == activity_name.lower():
            return os.path.join(dataset_path, n)
    return os.path.join(dataset_path, activity_name)


def build_trajectory_features(points, window_size):
    features = []
    n_points = points.shape[0]
    for i in range(0, n_points, window_size):
        end_idx = min(i + window_size, n_points)
        window = points[i:end_idx]
        if len(window) < 2:
            continue
        magnitude = np.linalg.norm(window, axis=1)
        window_features = []
        # заполнение статистич. признаками
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


def build_feature_windows_from_files(dataset_path, activities, window_size, convert_to_g):
    all_features = []
    all_labels = []
    for class_id, activity in enumerate(activities):
        folder = find_activity_folder(dataset_path, activity)
        if not os.path.exists(folder):
            print(f"warn: folder not found {folder}!")
            continue
        files = sorted([f for f in os.listdir(folder) if f.endswith('.txt')])
        for fname in files:
            arr = np.loadtxt(os.path.join(folder, fname))
            if arr.ndim == 1:
                arr = arr.reshape(-1, 3)
            real = -1.5 + (arr / 63.0) * 3.0
            if convert_to_g:
                real = real * 9.81
            feats = build_trajectory_features(real, window_size=window_size)
            if feats.size == 0:
                continue
            all_features.append(feats)
            all_labels.extend([class_id] * feats.shape[0])
    if len(all_features) == 0:
        return np.empty((0, 20)), np.array([], dtype=int)
    X = np.vstack(all_features)
    y = np.array(all_labels, dtype=int)
    return X, y


class MLP:
    def __init__(self, layer_sizes, dropout=0.0, seed=42, momentum=0.9):
        self.layer_sizes = layer_sizes
        self.dropout = dropout
        self.rng = np.random.RandomState(seed)
        self.n_layers = len(layer_sizes) - 1
        self.W = []
        self.b = []
        self.vW = []
        self.vb = []
        self.momentum = momentum
        for i in range(self.n_layers):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i + 1]
            limit = np.sqrt(6.0 / (in_dim + out_dim))
            W = self.rng.uniform(-limit, limit, size=(in_dim, out_dim)).astype(np.float32)
            b = np.zeros((out_dim,), dtype=np.float32)
            self.W.append(W)
            self.b.append(b)
            self.vW.append(np.zeros_like(W))
            self.vb.append(np.zeros_like(b))

    @staticmethod
    def _relu(x):
        return np.maximum(0, x)

    @staticmethod
    def _relu_grad(x):
        return (x > 0).astype(np.float32)

    @staticmethod
    def _softmax(x):
        x = x - np.max(x, axis=1, keepdims=True)
        e = np.exp(x)
        return e / np.sum(e, axis=1, keepdims=True)

    def forward(self, X, train=True):
        activations = [X]
        pre_acts = []
        dropout_masks = []
        out = X
        for i in range(self.n_layers - 1):
            z = out.dot(self.W[i]) + self.b[i]
            pre_acts.append(z)
            a = self._relu(z)
            if train and self.dropout > 0.0:
                mask = (self.rng.rand(*a.shape) >= self.dropout).astype(np.float32) / (1.0 - self.dropout)
                a *= mask
                dropout_masks.append(mask)
            else:
                dropout_masks.append(None)
            activations.append(a)
            out = a
        z = out.dot(self.W[-1]) + self.b[-1]
        pre_acts.append(z)
        probs = self._softmax(z)
        activations.append(probs)
        return probs, {'activations': activations, 'pre_acts': pre_acts, 'dropout_masks': dropout_masks}

    def backward_weighted(self, cache, y_onehot, sample_weights):
        activations = cache['activations']
        pre_acts = cache['pre_acts']
        dropout_masks = cache['dropout_masks']
        grads_W = [None] * self.n_layers
        grads_b = [None] * self.n_layers
        probs = activations[-1]
        batch_w = sample_weights
        sum_w = np.sum(batch_w) + 1e-12
        delta = (probs - y_onehot) * (batch_w[:, None] / sum_w)
        a_prev = activations[-2]
        grads_W[-1] = a_prev.T.dot(delta)
        grads_b[-1] = np.sum(delta, axis=0)
        delta_prev = delta
        for i in range(self.n_layers - 2, -1, -1):
            z = pre_acts[i]
            a_prev = activations[i]
            grad_a = delta_prev.dot(self.W[i + 1].T)
            rel = self._relu_grad(z)
            delta_i = grad_a * rel
            mask = dropout_masks[i]
            if mask is not None:
                delta_i *= mask
            grads_W[i] = a_prev.T.dot(delta_i)
            grads_b[i] = np.sum(delta_i, axis=0)
            delta_prev = delta_i
        return grads_W, grads_b

    def update_params_momentum(self, grads_W, grads_b, lr=1e-3):
        for i in range(self.n_layers):
            self.vW[i] = self.momentum * self.vW[i] - lr * grads_W[i]
            self.vb[i] = self.momentum * self.vb[i] - lr * grads_b[i]
            self.W[i] += self.vW[i]
            self.b[i] += self.vb[i]

    def predict(self, X):
        probs, _ = self.forward(X, train=False)
        return np.argmax(probs, axis=1)


def run_pipeline(dataset_path, activities,
                 window_size=50,
                 n_splits=5,
                 random_state=42,
                 hidden=[128, 64],
                 dropout=0.2,
                 epochs=40,
                 batch=128,
                 lr=1e-2,
                 val_fraction=0.1):
    np.random.seed(random_state)
    random.seed(random_state)
    X_feat, y = build_feature_windows_from_files(dataset_path, activities, window_size=window_size, convert_to_g=True)
    if X_feat.shape[0] == 0:
        raise RuntimeError("no samples")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_feat)
    n_classes = int(np.max(y) + 1)
    # разбиение датасета на фолды
    skf = StratifiedKFold(n_splits=min(n_splits, max(2, np.min(np.bincount(y)))), shuffle=True,
                          random_state=random_state)

    fold = 0
    fold_accs = []
    histories = []
    class_counts = np.bincount(y)
    class_weights = (1.0 / (class_counts + 1e-12)).astype(np.float32)
    class_weights = class_weights / np.mean(class_weights)

    for train_idx, test_idx in skf.split(Xs, y):
        fold += 1
        print(f"Fold {fold}: train {len(train_idx)} test {len(test_idx)}")
        X_train_full = Xs[train_idx]
        y_train_full = y[train_idx]
        X_test = Xs[test_idx]
        y_test = y[test_idx]

        sample_weights_full = class_weights[y_train_full]

        # classes output
        layer_sizes = [X_train_full.shape[1]] + hidden + [n_classes]
        model = MLP(layer_sizes=layer_sizes, dropout=dropout, seed=random_state + fold, momentum=0.9)

        history = {'losses': [], 'val_accs': []}
        rng = np.random.RandomState(random_state + fold)

        for ep in range(1, epochs + 1):
            # выборка val-подмножества
            n_val = max(1, int(val_fraction * len(X_train_full)))
            val_idx_in_train = rng.choice(len(X_train_full), size=n_val, replace=False)
            tr_mask = np.ones(len(X_train_full), dtype=bool)
            tr_mask[val_idx_in_train] = False
            X_tr = X_train_full[tr_mask]
            y_tr = y_train_full[tr_mask]
            sw_tr = sample_weights_full[tr_mask]
            X_val = X_train_full[val_idx_in_train]
            y_val = y_train_full[val_idx_in_train]

            perm = rng.permutation(len(X_tr))
            X_tr_sh = X_tr[perm]
            y_tr_sh = y_tr[perm]
            sw_tr_sh = sw_tr[perm]

            # минибатчи
            epoch_loss = 0.0
            for i in range(0, len(X_tr_sh), batch):
                xb = X_tr_sh[i:i + batch]
                yb = y_tr_sh[i:i + batch]
                swb = sw_tr_sh[i:i + batch]
                # добавление гауссова шума
                xb_aug = xb + rng.normal(scale=0.005, size=xb.shape).astype(np.float32)
                probs, cache = model.forward(xb_aug, train=True)
                yb_onehot = np.zeros((len(yb), n_classes), dtype=np.float32)
                yb_onehot[np.arange(len(yb)), yb] = 1.0
                batch_w = swb
                sum_w = np.sum(batch_w) + 1e-12
                loss_batch = -np.sum(batch_w * np.sum(yb_onehot * np.log(probs + 1e-12), axis=1)) / sum_w
                epoch_loss += loss_batch * len(xb)
                grads_W, grads_b = model.backward_weighted(cache, yb_onehot, batch_w)
                model.update_params_momentum(grads_W, grads_b, lr=lr)
            epoch_loss /= max(1, len(X_tr_sh))

            train_preds = model.predict(X_tr)
            val_preds = model.predict(X_val)
            val_acc = np.mean(val_preds == y_val) if len(y_val) > 0 else 0.0

            history['losses'].append(epoch_loss)
            history['val_accs'].append(val_acc)

            print(
                f"Fold {fold} Epoch {ep}/{epochs} loss={epoch_loss:.4f} val_acc={val_acc:.4f}")

        # оценка на тестовых данных [в train не испол-ся]
        test_preds = model.predict(X_test)
        test_acc = np.mean(test_preds == y_test)
        print(f"Fold {fold} test accuracy: {test_acc:.4f}")
        fold_accs.append(test_acc)
        histories.append(history)

    visualize.plot_mlp_history(histories, out_dir="plots", prefix="mlp")
    visualize.plot_summary(fold_accs, [], out_dir="plots", prefix="mlp_sum")

    print("MLP mean acc: {:.4f} std: {:.4f}".format(np.mean(fold_accs), np.std(fold_accs)))
    return {'fold_acc': fold_accs, 'histories': histories}


if __name__ == "__main__":
    ds_path = r'../HMP_Dataset'
    activities = get_dirs(ds_path)
    out = run_pipeline(ds_path, activities,
                       window_size=60, n_splits=5, random_state=42,
                       hidden=[512, 256, 128, 64, 32], dropout=0.1,
                       epochs=300, batch=128, lr=1e-2, val_fraction=0.1)
