import os


def get_dirs(dataset_path):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"dataset path not found: {dataset_path}")

    activities = []
    for name in sorted(os.listdir(dataset_path)):
        full = os.path.join(dataset_path, name)
        if name.startswith('.'):
            continue
        if os.path.isdir(full):
            activities.append(name)
    return activities
