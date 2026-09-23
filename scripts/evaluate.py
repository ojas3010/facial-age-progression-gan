"""Per-age-group FID, KID and ArcFace identity similarity for a generator checkpoint.

Usage: python scripts/evaluate.py <G checkpoint> [out_dir] [image_dir]

Translates every val image (same split as training) to each age group other
than its own, saving real/<g>/<i>.png (val image i, true group g) and
fake/<g>/<i>.png (val image i translated to group g) for scripts/age_mivolo.py.
Pass a numbered checkpoint such as ml_core/models/100000-G.ckpt: training
replaces latest-G.ckpt in place, and holding it open can make that save fail.
"""
import os
import sys

import numpy as np
import torch
import torch_fidelity
from insightface.app import FaceAnalysis
from insightface.utils.face_align import norm_crop
from torchvision.utils import save_image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from ml_core.dataset import get_loaders
from ml_core.model import Generator, generator_state_dict, label2onehot

AGE_RANGES = ['0-10', '11-20', '21-30', '31-40', '41-50', '51+']  # get_age_group()

ckpt = sys.argv[1]
out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    ROOT, 'results', 'eval', os.path.basename(ckpt).removesuffix('.ckpt'))
image_dir = sys.argv[3] if len(sys.argv) > 3 else os.path.join(ROOT, 'ml_core', 'data', 'utkface')
if os.path.isdir(out_dir) and os.listdir(out_dir):
    sys.exit(f'{out_dir} is not empty; delete it or pass another out_dir')  # stale images would skew metrics
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

G = Generator(c_dim=6, repeat_num=6).to(device).eval()
G.load_state_dict(generator_state_dict(torch.load(ckpt, map_location=device, weights_only=True)))

# SCRFD detector + ArcFace R50 (buffalo_l) on CPU. det_size 128 matches the
# images: at the default 640 the upscaled faces are too large to detect.
app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'recognition'],
                   providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(128, 128))


def to_bgr(x):
    """[-1, 1] NCHW batch -> uint8 BGR HWC arrays, quantized like the saved PNGs."""
    x = ((x + 1) / 2).clamp(0, 1).mul(255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    return [np.ascontiguousarray(im[..., ::-1]) for im in x]


for kind in ('real', 'fake'):
    for g in range(6):
        os.makedirs(os.path.join(out_dir, kind, str(g)), exist_ok=True)

_, val_loader = get_loaders(image_dir, batch_size=64, num_workers=0)
sims = [[] for _ in range(6)]  # ArcFace cosine(source, output) per target group
n_seen = no_face = 0
with torch.no_grad():
    for x, y in val_loader:
        x = x.to(device)
        fakes = [G(x, label2onehot(torch.full((len(x),), g), 6).to(device)) for g in range(6)]
        src, outs = to_bgr(x), [to_bgr(f) for f in fakes]
        crops, pairs = [], []
        for j, src_g in enumerate(y.tolist()):
            name = f'{n_seen + j:05d}.png'
            save_image((x[j] + 1) / 2, os.path.join(out_dir, 'real', str(src_g), name))
            targets = [g for g in range(6) if g != src_g]  # same-group translations excluded
            for g in targets:
                save_image((fakes[g][j] + 1) / 2, os.path.join(out_dir, 'fake', str(g), name))
            # Landmarks come from the real source and are reused for its outputs:
            # G keeps the face in place, and detection could fail on artifacts
            _, kps = app.det_model.detect(src[j], max_num=1)
            if len(kps) == 0:
                no_face += 1
                continue
            crops += [norm_crop(src[j], kps[0])] + [norm_crop(outs[g][j], kps[0]) for g in targets]
            pairs.append(targets)
        n_seen += len(x)
        if crops:
            emb = app.models['recognition'].get_feat(crops)
            emb /= np.linalg.norm(emb, axis=1, keepdims=True)
            k = 0
            for targets in pairs:
                for t, g in enumerate(targets, 1):
                    sims[g].append(float(emb[k] @ emb[k + t]))
                k += len(targets) + 1

n_real = [len(os.listdir(os.path.join(out_dir, 'real', str(g)))) for g in range(6)]
kid_size = min(1000, *n_real)  # one subset size for every group
print(f'Val images: {n_seen}, no face detected (left out of ArcFace): {no_face}, KID subset size: {kid_size}')
print(f"{'Group':<7}{'Ages':<7}{'Real':>6}{'Fake':>6}{'FID':>9}{'KID x1e3':>16}{'ArcFace':>9}")
rows = ['group,ages,n_real,n_fake,fid,kid_mean,kid_std,arcface_mean']
for g, ages in enumerate(AGE_RANGES):
    # save_cpu_ram=True means no DataLoader workers: on Windows each worker
    # re-imports this unguarded script. FID/KID values are unaffected.
    m = torch_fidelity.calculate_metrics(
        input1=os.path.join(out_dir, 'fake', str(g)), input2=os.path.join(out_dir, 'real', str(g)),
        cuda=device.type == 'cuda', fid=True, kid=True, kid_subset_size=kid_size, cache=False,
        save_cpu_ram=True, verbose=False)
    fid, kid, kid_std = (m[torch_fidelity.KEY_METRIC_FID], m[torch_fidelity.KEY_METRIC_KID_MEAN],
                         m[torch_fidelity.KEY_METRIC_KID_STD])
    n_fake, arc = n_seen - n_real[g], np.mean(sims[g])
    print(f'{g:<7}{ages:<7}{n_real[g]:>6}{n_fake:>6}{fid:>9.2f}{1e3 * kid:>9.2f} +- {1e3 * kid_std:<4.2f}{arc:>9.3f}')
    rows.append(f'{g},{ages},{n_real[g]},{n_fake},{fid:.4f},{kid:.6f},{kid_std:.6f},{arc:.4f}')
with open(os.path.join(out_dir, 'metrics.csv'), 'w') as f:
    f.write('\n'.join(rows) + '\n')
