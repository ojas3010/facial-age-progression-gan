import os
import torch
from torch.utils.data import Dataset, DataLoader
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
            for filename in os.listdir(image_dir):
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
