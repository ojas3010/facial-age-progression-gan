"""LaTeX tables for the paper: training setup, dataset split, per-age-group results,
and results by gender and race.

Usage: python scripts/make_tables.py [eval_dir] [out.tex] [image_dir]

Needs, in eval_dir: metrics.csv (scripts/evaluate.py), ages_mivolo.csv
(scripts/age_mivolo.py) and arcface_pairs.csv (scripts/arcface_pairs.py).
Pass the same image_dir that evaluate.py used. Writes results/tables_<ckpt>.tex
by default. The tables use booktabs: \\usepackage{booktabs}.
"""
import ast
import csv
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from ml_core.dataset import get_age_group, get_loaders

AGES = ['0--10', '11--20', '21--30', '31--40', '41--50', '51+']  # get_age_group()
GENDERS = ['Male', 'Female']  # UTKFace filename codes 0-1
RACES = ['White', 'Black', 'Asian', 'Indian', 'Others']  # UTKFace filename codes 0-4
eval_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'results', 'eval', '100000-G')
ckpt = os.path.basename(os.path.normpath(eval_dir))
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, 'results', f'tables_{ckpt}.tex')
image_dir = sys.argv[3] if len(sys.argv) > 3 else os.path.join(ROOT, 'ml_core', 'data', 'utkface')

# Setup: train.py's argparse defaults plus its fixed architecture (the run used no flags)
d = {}
for node in ast.walk(ast.parse(open(os.path.join(ROOT, 'ml_core', 'train.py')).read())):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        kw = {k.arg: k.value for k in node.keywords}
        if node.func.attr == 'add_argument' and 'default' in kw:
            try:
                d[node.args[0].value.lstrip('-')] = ast.literal_eval(kw['default'])
            except ValueError:  # non-literal defaults (paths)
                pass
        elif node.func.attr == 'update' and node.args and isinstance(node.args[0], ast.Dict):
            d.update(ast.literal_eval(node.args[0]))
setup = [('Iterations', f"{d['num-iters']:,}"), ('Batch size', d['batch-size']),
         ('Critic steps per generator step', d['n-critic']),
         ('Optimizer', rf"Adam, $\beta_1 = {d['beta1']}$, $\beta_2 = {d['beta2']}$"),
         ('Learning rate (G, D)', f"{d['g-lr']:g}, {d['d-lr']:g}"),
         (r'$\lambda_{\mathrm{cls}}$, $\lambda_{\mathrm{rec}}$, $\lambda_{\mathrm{gp}}$',
          f"{d['lambda-cls']:g}, {d['lambda-rec']:g}, {d['lambda-gp']:g}"),
         ('Image size', rf"${d['image_size']} \times {d['image_size']}$"), ('Age domains', d['c_dim']),
         ('Residual blocks in G', d['g_repeat_num']), ('Strided conv layers in D', d['d_repeat_num']),
         ('Seed', d['seed'])]

# Dataset split, and the val file behind each saved image: evaluate.py's <i>.png
# is the i-th val image in loader order, so the same get_loaders() call recovers it
train_loader, val_loader = get_loaders(image_dir, num_workers=0)
train, val = (Counter(l.dataset.dataset.labels[i] for i in l.dataset.indices) for l in (train_loader, val_loader))
val_names = [os.path.basename(val_loader.dataset.dataset.image_paths[k]) for k in val_loader.dataset.indices]


def labels(name):
    """True age, gender and race codes from [age]_[gender]_[race]_[date]; None if a field is missing."""
    p = name.split('_')
    ok = len(p) >= 4 and p[1] in ('0', '1') and p[2] in ('0', '1', '2', '3', '4')
    return int(p[0]), int(p[1]) if ok else None, int(p[2]) if ok else None


true_age, gender, race = zip(*map(labels, val_names))

with open(os.path.join(eval_dir, 'metrics.csv')) as f:
    metrics = list(csv.DictReader(f))
with open(os.path.join(eval_dir, 'ages_mivolo.csv')) as f:  # (kind, group, val index, estimated age)
    est = [(r['kind'], int(r['group']), int(r['file'].split('.')[0]), float(r['age'])) for r in csv.DictReader(f)]
with open(os.path.join(eval_dir, 'arcface_pairs.csv')) as f:  # (val index, target group, similarity)
    pairs = [(int(r['file'].split('.')[0]), int(r['trg_group']), float(r['arcface'])) for r in csv.DictReader(f)]
assert sum(int(m['n_real']) for m in metrics) == len(val_names) and all(
    get_age_group(true_age[i]) == g for k, g, i, a in est if k == 'real'), 'eval_dir does not match this val split'


# keep(g, i) selects images by group g and source val index i
def acc(kind, keep):  # % whose MiVOLO estimate lands in the group; age_mivolo.py's rule: round, then bin
    return f"{100 * np.mean([get_age_group(round(a)) == g for k, g, i, a in est if k == kind and keep(g, i)]):.1f}"


def mae(keep):  # real images only: estimate vs the filename's true age
    return f"{np.mean([abs(a - true_age[i]) for k, g, i, a in est if k == 'real' and keep(g, i)]):.2f}"


def mean_age(keep):
    return f"{np.mean([a for k, g, i, a in est if k == 'fake' and keep(g, i)]):.1f}"


def arcface(keep):
    return f'{np.mean([s for i, g, s in pairs if keep(g, i)]):.3f}'


def sub_row(name, keep, n):
    return rf'\quad {name} & {n:,} & {acc("real", keep)} & {acc("fake", keep)} & {arcface(keep)} \\'


split = [rf'{a} & {train[g]:,} & {val[g]:,} & {train[g] + val[g]:,} \\' for g, a in enumerate(AGES)]
split.append('\\midrule\n' rf'All & {train.total():,} & {val.total():,} & {train.total() + val.total():,} \\')

res = []
for g, a in enumerate(AGES):
    m, keep = metrics[g], (lambda gg, i, g=g: gg == g)
    res.append(rf"{a} & {float(m['fid']):.2f} & {1e3 * float(m['kid_mean']):.2f} $\pm$ {1e3 * float(m['kid_std']):.2f}"
               rf" & {arcface(keep)} & {acc('real', keep)} & {mae(keep)} & {acc('fake', keep)} & {mean_age(keep)} \\")
every = lambda g, i: True
res.append('\\midrule\n' rf"All & -- & -- & {arcface(every)} & {acc('real', every)} & {mae(every)} & {acc('fake', every)} & -- \\")

subs = []
for title, codes, names in (('Gender', gender, GENDERS), ('Race', race, RACES)):
    subs += [r'\midrule'] if subs else []
    subs.append(rf'\multicolumn{{5}}{{l}}{{\emph{{{title}}}}} \\')
    subs += [sub_row(name, lambda g, i, c=c, codes=codes: codes[i] == c, codes.count(c)) for c, name in enumerate(names)]
n_unlabeled = race.count(None)
unlabeled = f' Validation images whose filenames lack these labels are left out ({n_unlabeled}).' if n_unlabeled else ''
n_real = [int(m['n_real']) for m in metrics]

TEMPLATE = r"""\begin{table}[t]
\centering
\caption{Training setup: the defaults of \texttt{train.py}, which the reported run used unchanged.}
\label{tab:setup}
\begin{tabular}{ll}
\toprule
Setting & Value \\
\midrule
@SETUP@
\bottomrule
\end{tabular}
\end{table}

\begin{table}[t]
\centering
\caption{UTKFace split used for training and evaluation: a seeded (seed 42) random 80/20 split of the @TOTAL@ images.}
\label{tab:dataset}
\begin{tabular}{lrrr}
\toprule
Age group & Train & Val & Total \\
\midrule
@SPLIT@
\bottomrule
\end{tabular}
\end{table}

\begin{table*}[t]
\centering
\caption{Results per target age group on the @NVAL@ validation images (checkpoint \texttt{@CKPT@}). Each image is translated
to every age group except its own. FID and KID compare the translations into a group with the real validation images of
that group (KID: 100 subsets of @KID@); FID depends on the size of the real set, so compare groups by KID. ArcFace is the
mean cosine similarity between source and output. MiVOLO~v2 estimates each image's age: accuracy is the share of estimates
that fall in the group, and MAE is the mean absolute error against the age in the filename. Real-image accuracy is the
ceiling for generated-image accuracy.}
\label{tab:results}
\begin{tabular}{lrrrrrrr}
\toprule
 & & & & \multicolumn{2}{c}{Real val images} & \multicolumn{2}{c}{Generated images} \\
\cmidrule(lr){5-6} \cmidrule(lr){7-8}
Target group & FID $\downarrow$ & KID $\times 10^{3}$ $\downarrow$ & ArcFace $\uparrow$ & Acc.\ (\%) & MAE (yr) & Acc.\ (\%) $\uparrow$ & Mean est.\ age \\
\midrule
@RESULTS@
\bottomrule
\end{tabular}
\end{table*}

\begin{table}[t]
\centering
\caption{Age accuracy and ArcFace similarity by the source image's UTKFace gender and race labels, pooled over target
groups; N counts validation images. Subgroups differ in age distribution, so differences are not purely
demographic.@UNLABELED@}
\label{tab:subgroups}
\begin{tabular}{lrrrr}
\toprule
 & & \multicolumn{2}{c}{Age acc.\ (\%)} & \\
\cmidrule(lr){3-4}
Subgroup & N & Real & Generated & ArcFace $\uparrow$ \\
\midrule
@SUBGROUPS@
\bottomrule
\end{tabular}
\end{table}
"""
tex = (TEMPLATE.replace('@SETUP@', '\n'.join(rf'{k} & {v} \\' for k, v in setup))
       .replace('@TOTAL@', f'{train.total() + val.total():,}').replace('@SPLIT@', '\n'.join(split))
       .replace('@NVAL@', f'{len(val_names):,}').replace('@CKPT@', ckpt.replace('_', r'\_'))
       .replace('@KID@', str(min(1000, *n_real)))  # evaluate.py's KID subset size rule
       .replace('@RESULTS@', '\n'.join(res)).replace('@SUBGROUPS@', '\n'.join(subs))
       .replace('@UNLABELED@', unlabeled))
with open(out, 'w', encoding='utf-8') as f:
    f.write(tex)
print(tex + f'% Wrote {out}')
