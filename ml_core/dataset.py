import copy
import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image

def get_age_group(age):
    """Map continuous age to 6 discrete age groups."""
    if age <= 10: return 0
    elif age <= 20: return 1
    elif age <= 30: return 2
    elif age <= 40: return 3
    elif age <= 50: return 4
    else: return 5

class UTKFaceDataset(Dataset):
    """
    Dataset loader for UTKFace dataset.
    Assumes filenames are in format: [age]_[gender]_[race]_[date].jpg
    """
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        if os.path.exists(image_dir):
            # Sorted so the file order (and any index-based split) is
            # reproducible across runs and platforms
            for filename in sorted(os.listdir(image_dir)):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    parts = filename.split('_')
                    if len(parts) >= 1:
                        try:
                            age = int(parts[0])
                            age_group = get_age_group(age)
                            self.image_paths.append(os.path.join(image_dir, filename))
                            self.labels.append(age_group)
                        except ValueError:
                            continue

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        # Convert label to one-hot encoding representation not needed here if using CrossEntropyLoss
        # But StarGAN typically uses one-hot encoded domain labels for the generator
        return image, label

def get_loader(image_dir, image_size=128, batch_size=16, num_workers=4):
    """Builds and returns Dataloader."""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = UTKFaceDataset(image_dir, transform)
    
    if len(dataset) == 0:
        print(f"Warning: No valid images found in {image_dir}")

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=num_workers)
    return data_loader

def get_loaders(image_dir, image_size=128, batch_size=16, num_workers=4, val_frac=0.2, seed=42):
    """Builds train/val DataLoaders with a deterministic 80/20 split.

    The split is a seeded permutation over the sorted file list, so it is
    reproducible across runs (training resume keeps the same val set).
    The val dataset omits RandomHorizontalFlip. Returns (train_loader,
    val_loader); val_loader is None when the dataset is too small to
    yield a val sample.
    """
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Single directory listing shared by both splits: two separate
    # UTKFaceDataset instances could observe different directory contents
    # between calls and desync the index permutation from dataset length.
    train_dataset = UTKFaceDataset(image_dir, train_transform)
    val_dataset = copy.copy(train_dataset)
    val_dataset.transform = val_transform

    if len(train_dataset) == 0:
        print(f"Warning: No valid images found in {image_dir}")

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(train_dataset), generator=generator).tolist()
    n_val = int(len(perm) * val_frac)
    val_indices, train_indices = perm[:n_val], perm[n_val:]

    train_loader = DataLoader(dataset=Subset(train_dataset, train_indices),
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)
    val_loader = None
    if n_val > 0:
        val_loader = DataLoader(dataset=Subset(val_dataset, val_indices),
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers)
    return train_loader, val_loader
