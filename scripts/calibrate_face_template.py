"""Measure where UTKFace's aligned & cropped images place the face, so
uploads can be aligned to the same framing (UTKFACE_TEMPLATE in
ml_core/inference.py).

Runs the same YuNet detector the backend uses over train-split images and
averages the five landmarks, as fractions of the image side. The mean is then
mirror-symmetrized: UTKFace is a left/right-balanced dataset, so any residual
asymmetry is detector bias, and averaging it out makes the eyes exactly level.

    python scripts/calibrate_face_template.py [--n 2000]
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ml_core.dataset import UTKFaceDataset
from ml_core.inference import FaceAligner


def train_paths(image_dir, seed=42, val_frac=0.2):
    """Train-split files, using the permutation in get_loaders(); the val
    split stays unseen for the alignment check."""
    ds = UTKFaceDataset(image_dir)
    perm = torch.randperm(len(ds), generator=torch.Generator().manual_seed(seed)).tolist()
    return [ds.image_paths[i] for i in perm[int(len(perm) * val_frac):]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", default=os.path.join(ROOT, "ml_core", "data", "utkface"))
    parser.add_argument("--n", type=int, default=2000)
    args = parser.parse_args()

    aligner = FaceAligner()
    points, sides, misses = [], set(), 0
    for path in train_paths(args.image_dir)[: args.n]:
        img = cv2.imread(path)
        sides.add(img.shape[:2])
        faces = aligner.detect_faces(img)
        if len(faces) == 0:
            misses += 1
            continue
        largest = faces[np.argmax(faces[:, 2] * faces[:, 3])]
        points.append(largest[4:14].reshape(5, 2) / img.shape[1])

    pts = np.stack(points)
    print(f"image sizes: {sides}; detected {len(pts)}, missed {misses}")
    eyes = pts[:, 1] - pts[:, 0]
    roll = np.degrees(np.arctan2(eyes[:, 1], eyes[:, 0]))
    iod = np.linalg.norm(eyes, axis=1)
    print(f"eye-line roll: mean {roll.mean():+.2f} deg, std {roll.std():.2f} deg")
    print(f"inter-ocular distance: mean {iod.mean():.4f}, std {iod.std():.4f} (fraction of side)")

    mean = pts.mean(axis=0)
    # Mirror: x -> 1 - x, and swap the left/right eye and mouth-corner pairs
    mirrored = mean[[1, 0, 2, 4, 3]] * [-1, 1] + [1, 0]
    template = (mean + mirrored) / 2
    print("raw mean:\n", np.round(mean, 4))
    print("UTKFACE_TEMPLATE = np.array([")
    for (x, y), name in zip(template, ("eye (image left)", "eye (image right)", "nose tip",
                                       "mouth corner (image left)", "mouth corner (image right)")):
        print(f"    [{x:.4f}, {y:.4f}],  # {name}")
    print("], dtype=np.float32)")


if __name__ == "__main__":
    main()
