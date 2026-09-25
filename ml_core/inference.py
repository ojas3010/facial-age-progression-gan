import hashlib
import math
import threading

import cv2
import torch
from torchvision import transforms
from PIL import Image
import os
import sys

# Ensure ml_core is in path for imports to work if run from backend
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_core.model import Generator, generator_state_dict, label2onehot
import numpy as np

# YuNet face detector (OpenCV Zoo, MIT - see assets/YUNET_LICENSE). The
# 2026may export has dynamic input dims, which OpenCV 5's ONNX engine needs to
# run at arbitrary image sizes; the older 2023mar file is fixed-shape.
YUNET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets",
                          "face_detection_yunet_2026may.onnx")
# Upstream git-LFS oid. A mismatch usually means the 131-byte LFS pointer was
# saved instead of the model.
YUNET_SHA256 = "ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0"

# Where UTKFace's aligned & cropped images put the five YuNet landmarks, as
# fractions of the image side (eyes, nose tip, mouth corners; image-left
# first). Measured by scripts/calibrate_face_template.py on 2000 train images:
# UTKFace eye lines are level to within 2.9 deg (std) and the inter-ocular
# distance varies by 7.5%, so a single template describes the dataset well.
UTKFACE_TEMPLATE = np.array([
    [0.3017, 0.2831],  # eye (image left)
    [0.6983, 0.2831],  # eye (image right)
    [0.5000, 0.5170],  # nose tip
    [0.3294, 0.6765],  # mouth corner (image left)
    [0.6706, 0.6765],  # mouth corner (image right)
], dtype=np.float32)


class NoFaceDetectedError(ValueError):
    """The image contains no face the detector is confident about."""


class FaceAligner:
    """Detects the largest face in a photo and crops it the way UTKFace's
    aligned & cropped images are framed: eyes level, landmarks fitted to
    UTKFACE_TEMPLATE, output_size x output_size."""

    # YuNet is trained on faces of roughly 10-300 px. Larger photos are
    # detected on a downscaled copy; if that finds nothing, a smaller copy
    # catches close-ups whose face is still over the limit.
    DETECT_MAX_SIDES = (640, 256)
    # A face that fills the frame (a tight crop, a UTKFace image) has no
    # context around it; a border makes it detectable
    DETECT_PAD_FRAC = 0.25

    def __init__(self, model_path=YUNET_PATH, output_size=128, score_threshold=0.9):
        with open(model_path, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        if digest != YUNET_SHA256:
            raise RuntimeError(f"Face detector {model_path} has SHA-256 {digest}, expected {YUNET_SHA256}.")
        self.output_size = output_size
        self._detector = cv2.FaceDetectorYN.create(model_path, "", (320, 320), score_threshold, 0.3, 5000)
        # The detector is stateful (setInputSize) and FastAPI serves sync
        # endpoints from a thread pool
        self._lock = threading.Lock()

    def detect_faces(self, image_bgr):
        """All faces as YuNet rows in image_bgr's coordinates: x, y, w, h,
        five landmark (x, y) pairs, score. Empty (0, 15) if none."""
        h, w = image_bgr.shape[:2]
        tried = set()
        for max_side in self.DETECT_MAX_SIDES:
            scale = min(1.0, max_side / max(h, w))
            if scale in tried:
                continue  # small image: both passes would run at full size
            tried.add(scale)
            small = image_bgr
            if scale < 1:
                small = cv2.resize(image_bgr, (max(1, round(w * scale)), max(1, round(h * scale))),
                                   interpolation=cv2.INTER_AREA)
            pad = round(self.DETECT_PAD_FRAC * max(small.shape[:2]))
            padded = cv2.copyMakeBorder(small, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
            with self._lock:
                self._detector.setInputSize((padded.shape[1], padded.shape[0]))
                _, faces = self._detector.detect(padded)
            if faces is not None and len(faces):
                faces = faces.copy()
                faces[:, [0, 1, *range(4, 14)]] -= pad  # positions, not w/h
                faces[:, :14] *= [w / small.shape[1], h / small.shape[0]] * 7
                return faces
        return np.empty((0, 15), dtype=np.float32)

    def alignment_matrix(self, landmarks):
        """2x3 affine taking image coordinates to the output crop: rotation
        that levels the eyes, then the scale and shift that best fit all five
        landmarks to the template (least squares, rotation held fixed)."""
        dx, dy = landmarks[1] - landmarks[0]
        angle = math.atan2(dy, dx)
        c, s = math.cos(angle), math.sin(angle)
        rot = np.array([[c, s], [-s, c]])  # rotates by -angle
        src = landmarks @ rot.T
        dst = UTKFACE_TEMPLATE * self.output_size
        src_c, dst_c = src - src.mean(axis=0), dst - dst.mean(axis=0)
        scale = (src_c * dst_c).sum() / (src_c ** 2).sum()
        if not scale > 0:
            # Degenerate or mirrored landmark set: not a usable face
            raise NoFaceDetectedError("Detected landmarks do not form a face.")
        shift = dst.mean(axis=0) - scale * src.mean(axis=0)
        return np.hstack([scale * rot, shift[:, None]])

    def align(self, image_rgb, landmarks):
        """Crop image_rgb (H x W x 3 uint8) to the aligned output face."""
        matrix = self.alignment_matrix(np.asarray(landmarks, dtype=np.float64))
        scale = math.hypot(*matrix[0, :2])
        if scale < 1:
            # warpAffine samples without antialiasing: shrinking a 1000 px
            # face straight to 128 px aliases. Area-downsample first so the
            # warp itself runs at ~1:1.
            h, w = image_rgb.shape[:2]
            new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
            image_rgb = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            matrix = matrix.copy()
            matrix[:, 0] *= w / new_w
            matrix[:, 1] *= h / new_h
        # Black outside the photo, as in UTKFace's own crops of faces near
        # the frame edge (~13% of the dataset has black wedges)
        return cv2.warpAffine(image_rgb, matrix, (self.output_size, self.output_size),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def __call__(self, image_pil):
        """Aligned crop of the largest face in image_pil, as a PIL image.
        Raises NoFaceDetectedError if there is none."""
        rgb = np.asarray(image_pil.convert("RGB"))
        faces = self.detect_faces(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if len(faces) == 0:
            raise NoFaceDetectedError("No face detected.")
        largest = faces[np.argmax(faces[:, 2] * faces[:, 3])]
        return Image.fromarray(self.align(rgb, largest[4:14].reshape(5, 2)))


class AgeProgressor:
    def __init__(self, model_path=None, c_dim=6, image_size=128):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.c_dim = c_dim
        self.image_size = image_size
        # False means the generator is running on random init: output is noise
        self.weights_loaded = False

        self.G = Generator(c_dim=c_dim, repeat_num=6)

        if model_path and os.path.exists(model_path):
            try:
                # weights_only: never unpickle arbitrary objects from a checkpoint file
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
                # Strict: a checkpoint whose keys don't match (wrong network,
                # wrong architecture) must fail here, not load nothing and
                # silently serve random weights
                self.G.load_state_dict(generator_state_dict(checkpoint))
                self.weights_loaded = True
                print(f"Loaded model from {model_path}")
            except Exception as e:
                print(f"Warning: Failed to load model weights: {e}")
                print("Using uninitialized weights (results will be noise).")
        else:
            print("Warning: Model file not found. Using uninitialized weights (results will be noise).")

        self.G.to(self.device)
        self.G.eval()

        # Uploads are not framed like the training data; this crops them to
        # UTKFace's aligned framing before the generator sees them
        self.aligner = FaceAligner(output_size=image_size)

        # Training images are square face crops. Scale the short side and
        # center-crop instead of squashing to a square, which would distort
        # the face geometry of any non-square upload.
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def align_and_progress(self, image_pil, target_age_group):
        """
        Full pipeline for an arbitrary photo: align the largest face, then
        age it. Raises NoFaceDetectedError if the photo has no face.
        Returns: (aligned input crop, aged crop), both PIL images
        """
        aligned = self.aligner(image_pil)
        return aligned, self.progress_age(aligned, target_age_group)

    def progress_age(self, image_pil, target_age_group):
        """
        Runs the generator on an image that is already a face crop (UTKFace
        framing); use align_and_progress() for arbitrary photos.
        image_pil: PIL Image
        target_age_group: int (0 to 5)
        Returns: PIL Image
        """
        # Prepare input
        x = self.transform(image_pil).unsqueeze(0).to(self.device)

        # Prepare target domain label
        target_label = torch.tensor([target_age_group])
        c_trg = label2onehot(target_label, self.c_dim).to(self.device)

        # Inference
        with torch.no_grad():
            x_fake = self.G(x, c_trg)

        # Denormalize and convert back to PIL
        x_fake = (x_fake.squeeze(0).cpu() + 1) / 2.0
        x_fake = x_fake.clamp(0, 1)
        x_fake = transforms.ToPILImage()(x_fake)

        return x_fake

# For testing independently
if __name__ == '__main__':
    # Initialize without weights
    progressor = AgeProgressor()
    # Create a dummy image
    dummy_img = Image.fromarray(np.uint8(np.random.rand(256, 256, 3) * 255))
    # Target age group 5 (51+)
    out_img = progressor.progress_age(dummy_img, 5)
    print(f"Generated image size: {out_img.size}")
