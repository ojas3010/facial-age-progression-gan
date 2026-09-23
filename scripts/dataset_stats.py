"""Image counts per age group for the train/val split used in training.

Usage: python scripts/dataset_stats.py [image_dir]
"""
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from ml_core.dataset import get_loaders

AGE_RANGES = ['0-10', '11-20', '21-30', '31-40', '41-50', '51+']  # get_age_group()

# Same default and same get_loaders() split as ml_core/train.py
image_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'ml_core', 'data', 'utkface')
train_loader, val_loader = get_loaders(image_dir, num_workers=0)

# Each loader wraps a Subset: count its indices' labels without loading images
train, val = (Counter(l.dataset.dataset.labels[i] for i in l.dataset.indices)
              for l in (train_loader, val_loader))

print(f"Total images: {train.total() + val.total()} (train {train.total()}, val {val.total()})")
print(f"{'Group':<7}{'Ages':<7}{'Train':>7}{'Val':>7}{'Total':>7}")
for g, ages in enumerate(AGE_RANGES):
    print(f"{g:<7}{ages:<7}{train[g]:>7}{val[g]:>7}{train[g] + val[g]:>7}")
