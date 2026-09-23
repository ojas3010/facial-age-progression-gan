"""Per-pair ArcFace similarity between each saved source image and its translations.

Usage: python scripts/arcface_pairs.py [eval_dir]

Recomputes, from the PNGs scripts/evaluate.py saved, the similarities behind
its ArcFace column, one row per (source, target group) pair, so that
make_tables.py can split them by gender and race. Same detector and alignment
as evaluate.py. Writes <eval_dir>/arcface_pairs.csv. CPU only.
"""
import os
import sys

import cv2
import numpy as np
import onnxruntime
from insightface.app import FaceAnalysis
from insightface.utils.face_align import norm_crop

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
eval_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'results', 'eval', '100000-G')

so = onnxruntime.SessionOptions()
so.intra_op_num_threads = 4  # leaves cores for concurrent CPU jobs; affects speed only
app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'recognition'],
                   providers=['CPUExecutionProvider'], sess_options=so)
app.prepare(ctx_id=-1, det_size=(128, 128))  # as in evaluate.py

rows, no_face = ['file,src_group,trg_group,arcface'], 0
for g in range(6):
    for name in sorted(os.listdir(os.path.join(eval_dir, 'real', str(g)))):
        src = cv2.imread(os.path.join(eval_dir, 'real', str(g), name))  # BGR, as insightface expects
        _, kps = app.det_model.detect(src, max_num=1)
        if len(kps) == 0:
            no_face += 1
            continue
        # Landmarks from the source, reused for its outputs (as in evaluate.py)
        targets = [t for t in range(6) if t != g]
        crops = [norm_crop(src, kps[0])] + [
            norm_crop(cv2.imread(os.path.join(eval_dir, 'fake', str(t), name)), kps[0]) for t in targets]
        emb = app.models['recognition'].get_feat(crops)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        rows += [f'{name},{g},{t},{float(emb[0] @ emb[k])!r}' for k, t in enumerate(targets, 1)]

with open(os.path.join(eval_dir, 'arcface_pairs.csv'), 'w') as f:
    f.write('\n'.join(rows) + '\n')
print(f'{len(rows) - 1} pairs, {no_face} sources without a detected face')
