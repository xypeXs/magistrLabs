import os

import matplotlib.pyplot as plt
import numpy as np


def plot_mlp_history(history, out_dir="plots", prefix="mlp_fold"):
    os.makedirs(out_dir, exist_ok=True)
    n_folds = len(history)
    for i, h in enumerate(history, start=1):
        epochs = np.arange(1, len(h['losses']) + 1)
        fig, ax1 = plt.subplots(figsize=(8, 4), constrained_layout=True)
        ax1.plot(epochs, h['losses'], label='train_loss', marker='o')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax2 = ax1.twinx()
        ax2.plot(epochs, h['train_accs'], label='train_acc', marker='s')
        ax2.plot(epochs, h['val_accs'], label='val_acc', marker='^')
        ax2.set_ylabel("Accuracy")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='best')
        ax1.set_title(f"MLP Training (Fold {i})")
        fname = os.path.join(out_dir, f"{prefix}_fold{i}.png")
        plt.savefig(fname)
        plt.show()
        plt.pause(0.001)

    plt.figure(figsize=(8, 4), constrained_layout=True)
    for i, h in enumerate(history, start=1):
        plt.plot(np.arange(1, len(h['val_accs']) + 1), h['val_accs'], label=f'fold{i} val_acc')
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("MLP: Validation accuracy per fold")
    plt.legend()
    fname = os.path.join(out_dir, f"{prefix}_val_acc_all_folds.png")
    plt.savefig(fname)
    plt.show()
    plt.pause(0.001)


def plot_cnn_history(history, out_dir="plots", prefix="cnn_fold"):
    os.makedirs(out_dir, exist_ok=True)
    for i, h in enumerate(history, start=1):
        epochs = np.arange(1, len(h['train_losses']) + 1)
        fig, ax1 = plt.subplots(figsize=(8, 4), constrained_layout=True)
        ax1.plot(epochs, h['train_losses'], label='train_loss', marker='o')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax2 = ax1.twinx()
        ax2.plot(epochs, h['val_accs'], label='val_acc', color='tab:orange', marker='^')
        ax2.set_ylabel("Validation Accuracy")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='best')
        ax1.set_title(f"CNN Training (Fold {i})")
        fname = os.path.join(out_dir, f"{prefix}_fold{i}.png")
        plt.savefig(fname)
        plt.show()
        plt.pause(0.001)

    plt.figure(figsize=(8, 4), constrained_layout=True)
    for i, h in enumerate(history, start=1):
        plt.plot(np.arange(1, len(h['val_accs']) + 1), h['val_accs'], label=f'fold{i} val_acc')
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("CNN: Validation accuracy per fold")
    plt.legend()
    fname = os.path.join(out_dir, f"{prefix}_val_acc_all_folds.png")
    plt.savefig(fname)
    plt.show()
    plt.pause(0.001)


def plot_summary(mlp_fold_acc, cnn_fold_acc, out_dir="plots", prefix="cv_summary"):
    os.makedirs(out_dir, exist_ok=True)
    folds = np.arange(1, len(mlp_fold_acc) + 1)
    width = 0.35
    plt.figure(figsize=(8, 4), constrained_layout=True)
    plt.bar(folds - width / 2, mlp_fold_acc, width=width, label='MLP')
    if len(cnn_fold_acc) == len(mlp_fold_acc):
        plt.bar(folds + width / 2, cnn_fold_acc, width=width, label='CNN')
    else:
        plt.bar(folds + width / 2, [np.nan] * len(folds), width=width, label='CNN (diff len)')
    plt.xlabel("Fold")
    plt.ylabel("Test Accuracy")
    plt.title("Per-fold test accuracies")
    plt.legend()
    fname = os.path.join(out_dir, f"{prefix}_per_fold.png")
    plt.savefig(fname)
    plt.show()
    plt.pause(0.001)

    print("MLP mean acc: {:.4f} std: {:.4f}".format(np.mean(mlp_fold_acc), np.std(mlp_fold_acc)))
    if len(cnn_fold_acc) > 0:
        print("CNN mean acc: {:.4f} std: {:.4f}".format(np.mean(cnn_fold_acc), np.std(cnn_fold_acc)))
