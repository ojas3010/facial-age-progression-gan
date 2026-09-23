"""MiVOLO v2 age-group accuracy on the images saved by scripts/evaluate.py.

Runs in the separate CPU venv, since MiVOLO pins torch 2.5.1:
    ..\\venv-mivolo\\Scripts\\python.exe scripts\\age_mivolo.py <eval_dir>

Real: MiVOLO's accuracy on the real val images of each group - the ceiling
for the generated-image number. Fake: share of images translated to group g
that MiVOLO places in g. Age: mean predicted age. Per-image predictions go
to <eval_dir>/ages_mivolo.csv.
"""
import os
import sys

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from ml_core.dataset import get_age_group

AGE_RANGES = ['0-10', '11-20', '21-30', '31-40', '41-50', '51+']  # get_age_group()
# trust_remote_code executes this commit's code, which was reviewed before
# first use. Re-review before changing the revision.
REPO, REVISION = 'iitolstykh/mivolo_v2', '53393526c220e34cdd7b722b36d22b6f9e5f4241'

model = AutoModelForImageClassification.from_pretrained(
    REPO, revision=REVISION, trust_remote_code=True, torch_dtype=torch.float32).eval()
processor = AutoImageProcessor.from_pretrained(REPO, revision=REVISION, trust_remote_code=True)


def predict_ages(paths, batch_size=32):
    ages = []
    for k in range(0, len(paths), batch_size):
        faces = [cv2.imread(p) for p in paths[k:k + batch_size]]  # BGR, as the processor expects
        # The processor turns None into a blank input, so a failed read must not pass silently
        assert all(f is not None for f in faces), f'unreadable image in {paths[k:k + batch_size]}'
        with torch.no_grad():
            out = model(faces_input=processor(images=faces)['pixel_values'],
                        body_input=processor(images=[None] * len(faces))['pixel_values'])  # face only, no body crop
        ages += out.age_output.flatten().tolist()
    return ages


eval_dir = sys.argv[1]
rows = ['kind,group,file,age']
print(f"{'Group':<7}{'Ages':<7}{'Real':>7}{'Acc':>7}{'Age':>6}{'Fake':>7}{'Acc':>7}{'Age':>6}")
for g, age_range in enumerate(AGE_RANGES):
    line = f'{g:<7}{age_range:<7}'
    for kind in ('real', 'fake'):
        d = os.path.join(eval_dir, kind, str(g))
        files = sorted(os.listdir(d))
        ages = predict_ages([os.path.join(d, f) for f in files])
        # MiVOLO regresses integer age labels, so round before binning
        acc = np.mean([get_age_group(round(a)) == g for a in ages])
        line += f'{len(files):>7}{100 * acc:>6.1f}%{np.mean(ages):>6.1f}'
        # Full precision: make_tables.py re-bins these, and a 2-decimal
        # 10.50 rounds differently than the 10.503 it came from
        rows += [f'{kind},{g},{f},{a!r}' for f, a in zip(files, ages)]
    print(line)
with open(os.path.join(eval_dir, 'ages_mivolo.csv'), 'w') as f:
    f.write('\n'.join(rows) + '\n')
