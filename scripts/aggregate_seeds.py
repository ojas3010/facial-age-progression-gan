"""Mean +- sample std (ddof=1) across training seeds of the per-age-group results.

Usage: python scripts/aggregate_seeds.py <eval_dir> [<eval_dir> ...] [--image-dir DIR] [--out CSV]

Each eval_dir is one seed's scripts/evaluate.py folder, after scripts/age_mivolo.py
and scripts/arcface_pairs.py have run in it. Each run's values follow
make_tables.py's rules, so a single eval_dir reproduces the results table of its
tables_<ckpt>.tex. KID's within-run std (the spread over KID subsets) is left out:
the +- here is the spread across seeds only. Real-image accuracy and MAE depend
only on the val images and MiVOLO, so their std across seeds should be 0; a
nonzero value means the eval folders do not hold the same real images.
Pass the same image_dir that evaluate.py used.
"""
import argparse
import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from ml_core.dataset import get_age_group, get_loaders

AGES = ['0-10', '11-20', '21-30', '31-40', '41-50', '51+']  # get_age_group()
# (key, column, format), in make_tables.py's results-table order
METRICS = [('fid', 'FID', '.2f'), ('kid_x1e3', 'KID x1e3', '.2f'), ('arcface', 'ArcFace', '.3f'),
           ('real_acc', 'Real acc %', '.1f'), ('real_mae', 'Real MAE', '.2f'),
           ('fake_acc', 'Gen acc %', '.1f'), ('fake_age', 'Mean est age', '.1f')]


def run_metrics(eval_dir, true_age):
    """{(metric, group): value} for one eval folder. Group 'All' pools every
    group, for the metrics make_tables.py pools (ArcFace, accuracy, MAE)."""
    with open(os.path.join(eval_dir, 'metrics.csv')) as f:
        rows = list(csv.DictReader(f))
    with open(os.path.join(eval_dir, 'ages_mivolo.csv')) as f:  # (kind, group, val index, estimated age)
        est = [(r['kind'], int(r['group']), int(r['file'].split('.')[0]), float(r['age'])) for r in csv.DictReader(f)]
    with open(os.path.join(eval_dir, 'arcface_pairs.csv')) as f:  # (target group, similarity)
        pairs = [(int(r['trg_group']), float(r['arcface'])) for r in csv.DictReader(f)]
    assert [int(r['group']) for r in rows] == list(range(6)), f'{eval_dir}: unexpected metrics.csv rows'
    assert all(get_age_group(true_age[i]) == g for k, g, i, a in est if k == 'real'), \
        f'{eval_dir} does not match this val split'

    out = {}
    for g in [*range(6), None]:
        group = 'All' if g is None else AGES[g]
        keep = lambda gg: g is None or gg == g
        if g is not None:
            out['fid', group] = float(rows[g]['fid'])
            out['kid_x1e3', group] = 1e3 * float(rows[g]['kid_mean'])
            out['fake_age', group] = np.mean([a for k, gg, i, a in est if k == 'fake' and gg == g])
        out['arcface', group] = np.mean([s for gg, s in pairs if keep(gg)])
        for kind in ('real', 'fake'):  # age_mivolo.py's rule: round, then bin
            out[f'{kind}_acc', group] = 100 * np.mean(
                [get_age_group(round(a)) == gg for k, gg, i, a in est if k == kind and keep(gg)])
        out['real_mae', group] = np.mean([abs(a - true_age[i]) for k, gg, i, a in est if k == 'real' and keep(gg)])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('eval_dirs', nargs='+')
    parser.add_argument('--image-dir', default=os.path.join(ROOT, 'ml_core', 'data', 'utkface'))
    parser.add_argument('--out', help='CSV of mean, std and every run value per metric and group')
    args = parser.parse_args()

    # evaluate.py's <i>.png is the i-th val image in loader order
    _, val_loader = get_loaders(args.image_dir, num_workers=0)
    true_age = [int(os.path.basename(val_loader.dataset.dataset.image_paths[k]).split('_')[0])
                for k in val_loader.dataset.indices]
    runs = [run_metrics(d, true_age) for d in args.eval_dirs]
    n = len(runs)

    def stats(key):
        values = [r[key] for r in runs]
        return values, np.mean(values), np.std(values, ddof=1) if n > 1 else float('nan')

    print(f'{n} run(s): ' + ', '.join(args.eval_dirs))
    print('Cells: mean +- sample std (ddof=1) across runs' if n > 1 else 'One run: std needs at least 2')
    width = 17 if n > 1 else 14
    print(f"{'Group':<7}" + ''.join(f'{col:>{width}}' for _, col, _ in METRICS))
    csv_rows = [['metric', 'group', 'n_runs', 'mean', 'std', *args.eval_dirs]]
    for group in [*AGES, 'All']:
        line = f'{group:<7}'
        for key, _, fmt in METRICS:
            if (key, group) not in runs[0]:
                line += f"{'--':>{width}}"
                continue
            values, mean, std = stats((key, group))
            line += f'{f"{mean:{fmt}} +- {std:{fmt}}" if n > 1 else f"{mean:{fmt}}":>{width}}'
            # float(): numpy 2's repr would write np.float64(...) into the CSV
            csv_rows.append([key, group, n, repr(float(mean)), '' if n == 1 else repr(float(std)),
                             *(repr(float(v)) for v in values)])
        print(line)
    if args.out:
        with open(args.out, 'w', newline='') as f:
            csv.writer(f).writerows(csv_rows)
        print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
