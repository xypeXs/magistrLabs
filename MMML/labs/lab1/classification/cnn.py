import os
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold, KFold
from torch.utils.data import DataLoader, TensorDataset

import visualize
from utils import get_dirs


def find_activity_folder(dataset_path, activity_name):
    for n in os.listdir(dataset_path):
        if n.lower() == activity_name.lower():
            return os.path.join(dataset_path, n)
    return os.path.join(dataset_path, activity_name)


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


def augment_sequence(seq, noise_std=0.05):
    noise = np.random.normal(0, noise_std, seq.shape)
    return seq + noise.astype(np.float32)


def load_sequence_windows(dataset_path, activities, window_size=50, step_size=25, traj_len=50, convert_to_g=True,
                          augment=False):
    X_seq = []
    y = []
    names = []
    for class_id, activity in enumerate(activities):
        folder = find_activity_folder(dataset_path, activity)
        if not os.path.exists(folder):
            print(f"warn: folder not found {folder}!")
            continue
        file_names = sorted([f for f in os.listdir(folder) if f.endswith('.txt')])
        for fname in file_names:
            arr = np.loadtxt(os.path.join(folder, fname))
            if arr.ndim == 1:
                arr = arr.reshape(-1, 3)
            real = -1.5 + (arr / 63.0) * 3.0
            if convert_to_g:
                real = real * 9.81
            n_points = real.shape[0]
            for i in range(0, n_points - window_size + 1, step_size):
                end_idx = i + window_size
                window = real[i:end_idx]
                if window.shape[0] < 2:
                    continue
                seq = resample_trajectory(window, length=traj_len)
                X_seq.append(seq.astype(np.float32))
                y.append(class_id)
                names.append(f"{activity}/{fname}")
                if augment:
                    aug_seq = augment_sequence(seq)
                    X_seq.append(aug_seq)
                    y.append(class_id)
                    names.append(f"{activity}/{fname}_aug")
    if len(X_seq) == 0:
        return np.empty((0, traj_len, 3), dtype=np.float32), np.array([], dtype=int), []
    return np.stack(X_seq, axis=0), np.array(y, dtype=int), names


class ConvNet1D(nn.Module):
    def __init__(self, n_classes, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),

            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),

            nn.Conv1d(512, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        return self.net(x)


def train(model, train_loader, val_loader, criterion, optimizer, device, scheduler=None):
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
    running_loss /= len(train_loader.dataset) if len(train_loader.dataset) > 0 else 1.0

    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            out = model(xb)
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_true.append(yb.cpu().numpy())
    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds)
        all_true = np.concatenate(all_true)
        val_acc = np.mean(all_preds == all_true)
    else:
        val_acc = 0.0

    if scheduler is not None:
        scheduler.step(val_acc)

    return running_loss, val_acc


def evaluate_pytorch_model(model, X_seq, y, device='cpu', batch_size=256):
    model.eval()
    Xt = torch.tensor(np.transpose(X_seq, (0, 2, 1)), dtype=torch.float32).to(device)
    ds = TensorDataset(Xt, torch.tensor(y, dtype=torch.long).to(device))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds = []
    trues = []
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb)
            p = out.argmax(dim=1).cpu().numpy()
            preds.append(p)
            trues.append(yb.cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    return np.mean(preds == trues)


def group_stratified_kfold_indices(names, y, n_splits=5, random_state=42):
    group_keys = [os.path.basename(n) if isinstance(n, str) else str(n) for n in names]
    unique_groups, inverse = np.unique(group_keys, return_inverse=True)
    group_labels = []
    for gi in range(len(unique_groups)):
        idxs = np.where(inverse == gi)[0]
        labs = y[idxs]
        most_common = Counter(labs).most_common(1)[0][0]
        group_labels.append(most_common)
    group_labels = np.array(group_labels, dtype=int)

    class_group_counts = np.bincount(group_labels)
    nonzero = class_group_counts[class_group_counts > 0]
    if len(nonzero) == 0:
        raise RuntimeError("No groups found")
    min_per_class = int(nonzero.min())

    actual_splits = min(n_splits, len(unique_groups))
    if min_per_class >= 2 and actual_splits >= 2:
        skf = StratifiedKFold(n_splits=min(n_splits, min_per_class), shuffle=True, random_state=random_state)
        splits = skf.split(np.arange(len(unique_groups)), group_labels)
    else:
        kf = KFold(n_splits=actual_splits, shuffle=True, random_state=random_state)
        splits = kf.split(np.arange(len(unique_groups)))

    for group_train_idx, group_test_idx in splits:
        train_groups = set(unique_groups[group_train_idx])
        test_groups = set(unique_groups[group_test_idx])
        train_mask = np.array([g in train_groups for g in group_keys])
        test_mask = np.array([g in test_groups for g in group_keys])
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        yield train_idx, test_idx


def run_pipeline(dataset_path, activities, window_size=70, step_size=35, traj_len=60,
                 epochs=1000, batch=64, lr=1e-3, dropout=0.1,
                 n_splits=5, random_state=42, device=None):
    np.random.seed(random_state)
    random.seed(random_state)
    torch.manual_seed(random_state)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device:", device)

    X_seq, y_seq, names = load_sequence_windows(dataset_path, activities, window_size=window_size, step_size=step_size,
                                                traj_len=traj_len)
    print("sequences shape:", X_seq.shape, "labels:", np.unique(y_seq))
    if X_seq.shape[0] == 0:
        raise RuntimeError("no samples")

    n_classes = int(np.max(y_seq) + 1)

    class_counts = np.bincount(y_seq)
    class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float32).to(device)

    fold = 0
    fold_acc = []
    histories = []
    for train_idx, test_idx in group_stratified_kfold_indices(names, y_seq, n_splits=n_splits,
                                                              random_state=random_state):
        fold += 1
        train_names = [names[i] for i in train_idx]
        train_y = y_seq[train_idx]
        Xtr = X_seq[train_idx].copy()
        ytr = y_seq[train_idx].copy()
        Xte = X_seq[test_idx]
        yte = y_seq[test_idx]
        aug_Xtr = augment_sequence(Xtr)
        Xtr = np.concatenate([Xtr, aug_Xtr], axis=0)
        ytr = np.concatenate([ytr, ytr], axis=0)
        print(f"Fold {fold}: train {len(train_idx)} (aug to {Xtr.shape[0]}) test {len(test_idx)} (group-aware)")

        # нормализация
        mean_ch = Xtr.mean(axis=(0, 1), keepdims=False)
        std_ch = Xtr.std(axis=(0, 1), keepdims=False) + 1e-8
        Xtr_n = (Xtr - mean_ch[None, None, :]) / std_ch[None, None, :]
        Xte_n = (Xte - mean_ch[None, None, :]) / std_ch[None, None, :]

        Xtr_t = torch.tensor(np.transpose(Xtr_n, (0, 2, 1)), dtype=torch.float32)
        ytr_t = torch.tensor(ytr, dtype=torch.long)
        Xte_t = torch.tensor(np.transpose(Xte_n, (0, 2, 1)), dtype=torch.float32)
        yte_t = torch.tensor(yte, dtype=torch.long)

        train_ds = TensorDataset(Xtr_t, ytr_t)
        val_ds = TensorDataset(Xte_t, yte_t)
        train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False)

        model = ConvNet1D(n_classes=n_classes, dropout=dropout).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20)

        history = {'train_losses': [], 'val_accs': []}
        best_val = 0.0
        best_state = None
        for ep in range(1, epochs + 1):
            train_loss, val_acc = train(model, train_loader, val_loader, criterion, optimizer, device,
                                        scheduler)
            history['train_losses'].append(train_loss)
            history['val_accs'].append(val_acc)
            print(f"Fold {fold} Epoch {ep}/{epochs} train_loss={train_loss:.4f} val_acc={val_acc:.4f}")
            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)
        test_acc = evaluate_pytorch_model(model, Xte_n, yte, device=device)
        print(f"Fold {fold} test accuracy: {test_acc:.4f}")
        fold_acc.append(test_acc)
        histories.append(history)

    visualize.plot_cnn_history(histories, out_dir="plots", prefix="cnn")
    visualize.plot_summary([], fold_acc, out_dir="plots", prefix="cnn_vs_mlp")

    print("CNN CV mean acc: {:.4f} std: {:.4f}".format(np.mean(fold_acc), np.std(fold_acc)))
    return {'fold_acc': fold_acc, 'histories': histories}


if __name__ == "__main__":
    ds_path = r'../HMP_Dataset'
    activities = get_dirs(ds_path)
    out = run_pipeline(ds_path, activities,
                       window_size=70, step_size=35, traj_len=60,
                       epochs=100, batch=128, lr=1e-3, dropout=0.1,
                       n_splits=5, random_state=104, device=None)
