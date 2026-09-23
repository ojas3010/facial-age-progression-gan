"""Plot train/val D and G loss from a training log; print wall-clock time.

Usage: python scripts/plot_losses.py [log_path] [out_dir]
Writes d_loss.png and g_loss.png to out_dir (default: results/).
"""
import os
import re
import sys
from datetime import timedelta

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'train_log.txt')
out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, 'results')

TIME = re.compile(r'Elapsed \[(?:(\d+) days?, )?(\d+):(\d+):(\d+)\]')
TRAIN = re.compile(r'Iteration \[(\d+)/\d+\], D_loss \[([^\]]+)\](?:, G_loss \[([^\]]+)\])?')
VAL = re.compile(r'Val \[iter (\d+)\] D_loss \[([^\]]+)\], G_loss \[([^\]]+)\]')

raw = open(log_path, 'rb').read()
# PowerShell 5.1 `>` and Tee-Object write UTF-16 with a BOM
text = raw.decode('utf-16' if raw[:2] in (b'\xff\xfe', b'\xfe\xff') else 'utf-8', errors='ignore')

# iteration -> (D, G). A resumed run re-logs the iterations after its
# checkpoint; the later (resumed) value wins.
train, val = {}, {}
total = run = 0  # Elapsed restarts at 0 in every run appended to the log
for line in text.splitlines():
    if 'Starting training' in line:
        total, run = total + run, 0
    if m := TIME.search(line):
        d, h, mi, s = (int(x or 0) for x in m.groups())
        run = ((d * 24 + h) * 60 + mi) * 60 + s
    if m := TRAIN.search(line):
        train[int(m[1])] = (float(m[2]), float(m[3] or 'nan'))
    elif m := VAL.search(line):
        val[int(m[1])] = (float(m[2]), float(m[3]))
total += run
if not train:
    sys.exit(f'No "Iteration [...]" lines in {log_path}')

it = np.array(sorted(train))
tr = np.array([train[i] for i in it])
vit = np.array(sorted(val))
va = np.array([val[i] for i in vit]).reshape(-1, 2)
w = min(50, len(it))  # moving-average window, in log lines
step = it[1] - it[0] if len(it) > 1 else 1

plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#c3c2b7', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
                     'axes.grid': True, 'axes.grid.axis': 'y', 'grid.color': '#e1e0d9', 'grid.linewidth': 0.6})
os.makedirs(out_dir, exist_ok=True)
for k, name in enumerate(('D', 'G')):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.axhline(0, color='#c3c2b7', lw=0.8)
    ax.plot(it, tr[:, k], color='#2a78d6', alpha=0.2, lw=0.6, label='Train')
    ax.plot(it[w - 1:], np.convolve(tr[:, k], np.ones(w) / w, 'valid'), color='#2a78d6', lw=1.5,
            label=f'Train, {w * step}-iteration moving average')
    ax.plot(vit, va[:, k], 'o-', color='#eb6834', lw=1.5, ms=4, label='Val (fixed batch)')
    ax.set(xlabel='Iteration', ylabel=f'{name} loss')
    ax.legend(frameon=False)
    fig.savefig(os.path.join(out_dir, f'{name.lower()}_loss.png'), dpi=200, bbox_inches='tight')

print(f'Iterations {it[0]}-{it[-1]}, {len(vit)} val points')
print(f'Wall-clock training time: {timedelta(seconds=total)}')
