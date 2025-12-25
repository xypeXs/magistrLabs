import json
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from PIL import Image
from torchvision import models
from tqdm import tqdm


class DogDataset(data.Dataset):
    def __init__(self, path, train=True, transform=None):
        self.path = os.path.join(path, "train" if train else "test")
        self.transform = transform

        with open(os.path.join(self.path, "format.json"), "r") as fp:
            self.format = json.load(fp)

        self.length = 0
        self.files = []
        self.targets = torch.eye(10)

        for _dir, _target in self.format.items():
            path = os.path.join(self.path, _dir)
            list_files = os.listdir(path)
            self.length += len(list_files)
            self.files.extend(map(lambda _x: (os.path.join(path, _x), _target), list_files))

    def __getitem__(self, item):
        path_file, target = self.files[item]
        t = self.targets[target]
        img = Image.open(path_file)

        if self.transform:
            img = self.transform(img)

        return img, t

    def __len__(self):
        return self.length


# классификация на основе предобученной модели ResNet
def create_model_scenario1():
    resnet_weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=resnet_weights)
    model.fc = nn.Linear(2048, 10)
    return model, resnet_weights


# заморозка слоев, кроме layer4 и full-connected
# fine-tuning layer4, fc
def create_model_scenario2():
    resnet_weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=resnet_weights)

    for name, param in model.named_parameters():
        if 'layer4' not in name and 'fc' not in name:
            param.requires_grad = False

    model.fc = nn.Linear(2048, 10)
    return model, resnet_weights


# расширение архитектуры
def create_model_scenario3():
    resnet_weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=resnet_weights)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 10)
    )
    return model, resnet_weights


def train_model(model, train_loader, val_loader, scenario_name, epochs=10):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=0.001, weight_decay=0.01)
    loss_function = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        train_correct = 0
        train_total = 0

        train_tqdm = tqdm(train_loader, desc=f"{scenario_name} Epoch [{epoch + 1}/{epochs}]")
        for x_train, y_train in train_tqdm:
            optimizer.zero_grad()
            predict = model(x_train)
            loss = loss_function(predict, torch.argmax(y_train, dim=1))
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            _, predicted = torch.max(predict.data, 1)
            train_total += y_train.size(0)
            train_correct += (predicted == torch.argmax(y_train, dim=1)).sum().item()

            train_tqdm.set_postfix(loss=loss.item())

        scheduler.step()

        train_loss = epoch_train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total

        val_loss, val_accuracy = evaluate_model(model, val_loader, loss_function)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        print(
            f"{scenario_name} Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    return train_losses, val_losses, train_accuracies, val_accuracies


def evaluate_model(model, data_loader, loss_function):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in data_loader:
            outputs = model(x)
            loss = loss_function(outputs, torch.argmax(y, dim=1))
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == torch.argmax(y, dim=1)).sum().item()

    avg_loss = total_loss / len(data_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def plot_comparison(results, metric_name, title):
    plt.figure(figsize=(10, 6))
    for scenario_name, metrics in results.items():
        plt.plot(metrics[metric_name], label=scenario_name, linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel(metric_name.replace('_', ' ').title())
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


resnet_weights = models.ResNet50_Weights.DEFAULT
transforms = resnet_weights.transforms()

dataset_path = r"./dogs"
dataset_train = DogDataset(dataset_path, transform=transforms)
dataset_val = DogDataset(dataset_path, train=False, transform=transforms)

train_loader = data.DataLoader(dataset_train, batch_size=32, shuffle=True)
val_loader = data.DataLoader(dataset_val, batch_size=32, shuffle=False)

scenarios = {
    "1_Full_Training": create_model_scenario1(),
    "2_Partial_Freeze": create_model_scenario2(),
    "3_Extended_FC": create_model_scenario3()
}

results = {}

for scenario_name, (model, _) in scenarios.items():
    print(f"\n=== Training {scenario_name} ===")
    train_loss, val_loss, train_acc, val_acc = train_model(
        model, train_loader, val_loader, scenario_name, epochs=3
    )

    results[scenario_name] = {
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_accuracy': train_acc,
        'val_accuracy': val_acc
    }

    final_val_loss, final_val_acc = evaluate_model(model, val_loader, nn.CrossEntropyLoss())
    print(f"{scenario_name} Final - Val Loss: {final_val_loss:.4f}, Val Acc: {final_val_acc:.2f}%")

plot_comparison(results, 'train_loss', 'Training Loss Comparison')
plot_comparison(results, 'val_loss', 'Validation Loss Comparison')
plot_comparison(results, 'train_accuracy', 'Training Accuracy Comparison')
plot_comparison(results, 'val_accuracy', 'Validation Accuracy Comparison')

print("\n=== Final Test Results ===")
test_loader = data.DataLoader(dataset_val, batch_size=50, shuffle=False)

for scenario_name, (model, _) in scenarios.items():
    model.eval()
    test_correct = 0
    test_total = 0
    test_loss = 0
    loss_function = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x_test, y_test in test_loader:
            outputs = model(x_test)
            test_loss += loss_function(outputs, torch.argmax(y_test, dim=1)).item()
            _, predicted = torch.max(outputs.data, 1)
            test_total += y_test.size(0)
            test_correct += (predicted == torch.argmax(y_test, dim=1)).sum().item()

    test_accuracy = 100 * test_correct / test_total
    avg_test_loss = test_loss / len(test_loader)
    print(f"{scenario_name}: Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
