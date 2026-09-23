"""Qualitative figure: one random val face per source age group, shown in every age group.

Usage: python scripts/sample_grid.py [eval_dir]

Reads the images scripts/evaluate.py saved and writes results/qualitative_<ckpt>.png.
Row i is a face from age group i; column j is that face translated to group j.
The diagonal (j = i) is the unmodified input, outlined, since evaluate.py does
not generate same-group translations.
"""
import os
import random
import sys

from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

SEED = 0  # faces are drawn at random with this seed, not hand-picked
AGE_RANGES = ['0–10', '11–20', '21–30', '31–40', '41–50', '51+']  # get_age_group()
S, GAP, HEAD = 256, 8, 64  # tile (128 px images, 2x nearest-neighbour), gap, header height

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
eval_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'results', 'eval', '100000-G')

rng = random.Random(SEED)
picks = [rng.choice(sorted(os.listdir(os.path.join(eval_dir, 'real', str(g))))) for g in range(6)]

grid = Image.new('RGB', (6 * S + 5 * GAP, HEAD + 6 * S + 5 * GAP), 'white')
draw = ImageDraw.Draw(grid)
# DejaVu Sans ships with matplotlib; PIL's built-in font has no en dash
font = ImageFont.truetype(font_manager.findfont('DejaVu Sans'), 36)
for j, label in enumerate(AGE_RANGES):
    draw.text((j * (S + GAP) + S // 2, HEAD // 2), label, fill='#0b0b0b', font=font, anchor='mm')
for i, name in enumerate(picks):
    for j in range(6):
        path = os.path.join(eval_dir, 'real' if i == j else 'fake', str(j), name)
        x, y = j * (S + GAP), HEAD + i * (S + GAP)
        grid.paste(Image.open(path).convert('RGB').resize((S, S), Image.NEAREST), (x, y))
        if i == j:
            draw.rectangle([x, y, x + S - 1, y + S - 1], outline='#2a78d6', width=6)

out = os.path.join(ROOT, 'results', f'qualitative_{os.path.basename(os.path.normpath(eval_dir))}.png')
grid.save(out)
print(f'Seed {SEED}, faces (row = source group 0-5): {picks}\nWrote {out}')
