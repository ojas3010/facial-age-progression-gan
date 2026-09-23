"""Mean KID and ArcFace across the 6 target groups, per training checkpoint.

Usage: python scripts/plot_checkpoints.py [eval_root] [log_path]

Reads <eval_root>/<iter>-G/metrics.csv (scripts/evaluate.py) for every
evaluated checkpoint and writes results/checkpoint_metrics.png. Means are
unweighted over the 6 target groups; KID uses the same subset size at every
checkpoint, so its values are comparable across iterations. Iterations where
training resumed from a checkpoint (from the log) are marked with a dotted line.
"""
import csv
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
eval_root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'results', 'eval')
log_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, 'train_log.txt')

raw = open(log_path, 'rb').read() if os.path.exists(log_path) else b''
# PowerShell 5.1 `>` and Tee-Object write UTF-16 with a BOM
text = raw.decode('utf-16' if raw[:2] in (b'\xff\xfe', b'\xfe\xff') else 'utf-8', errors='ignore')
resumes = [int(n) for n in re.findall(r'Starting training from iteration (\d+)', text) if int(n) > 0]

points = []  # (iteration, mean KID x1e3, mean ArcFace)
for name in os.listdir(eval_root):
    m, path = re.fullmatch(r'(\d+)-G', name), os.path.join(eval_root, name, 'metrics.csv')
    if m and os.path.exists(path):
        with open(path) as f:
            rows = list(csv.DictReader(f))
        points.append((int(m[1]), 1e3 * sum(float(r['kid_mean']) for r in rows) / len(rows),
                       sum(float(r['arcface_mean']) for r in rows) / len(rows)))
points.sort()
it, kid, arc = zip(*points)

plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#c3c2b7', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
                     'axes.grid': True, 'axes.grid.axis': 'y', 'grid.color': '#e1e0d9', 'grid.linewidth': 0.6})
fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
for ax, y, title in ((axes[0], kid, r'Mean KID $\times 10^3$ (lower is better)'),
                     (axes[1], arc, 'Mean ArcFace similarity (higher is better)')):
    ax.plot(it, y, 'o-', color='#2a78d6', lw=1.5, ms=5)
    ticks = it if len(it) <= 8 else it[::2]  # every other label once they would crowd
    ax.set_xticks(ticks, [f'{i // 1000}k' for i in ticks])
    ax.set_xlabel('Iteration')
    ax.set_title(title, fontsize=10)
    for r in resumes:
        ax.axvline(r, color='#898781', lw=0.8, ls=':')
        ax.text(r, ax.get_ylim()[1], ' resumed', fontsize=8, color='#52514e', va='top')
fig.tight_layout()
out = os.path.join(ROOT, 'results', 'checkpoint_metrics.png')
fig.savefig(out, dpi=200, bbox_inches='tight')

# Descriptive only, not a selection rule: the paper reports the final checkpoint,
# fixed before any later metrics existed (choosing a checkpoint by these val
# scores and then reporting them would inflate the results). A plateau is the
# first checkpoint c whose next two 20k steps each improve mean KID by < 5%.
STEP, THRESH = 20000, 0.05
kid_at = dict(zip(it, kid))
gain = {c: (kid_at[c] - kid_at[c + STEP]) / kid_at[c] for c in it if c + STEP in kid_at}

print(f"{'Iteration':>10}{'KID x1e3':>10}{'ArcFace':>9}{'KID gain, next 20k':>20}")
for c, k, a in points:
    print(f'{c:>10,}{k:>10.2f}{a:>9.3f}' + (f'{100 * gain[c]:>19.1f}%' if c in gain else ''))
plateau = next((c for c in it if c + STEP in gain and gain[c] < THRESH and gain[c + STEP] < THRESH), None)
if plateau is not None:
    print(f'KID plateau from {plateau:,}: the next two steps improve it by {100 * gain[plateau]:.1f}% and '
          f'{100 * gain[plateau + STEP]:.1f}%, both < {100 * THRESH:.0f}%')
else:
    print(f'No KID plateau up to {it[-1]:,}: the curve had not flattened (no two consecutive 20k steps < {100 * THRESH:.0f}%)')
print(f'Wrote {out}')
