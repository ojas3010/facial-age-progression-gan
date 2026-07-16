import os
import sys
import datetime
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import argparse
import random

# Ensure ml_core is accessible as a package regardless of cwd (matches
# ml_core/inference.py and backend/main.py)
_ML_CORE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(_ML_CORE_DIR))

from ml_core.model import Generator, Discriminator
from ml_core.dataset import get_loaders

def gradient_penalty(y, x, device):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    weight = torch.ones(y.size()).to(device)
    dydx = torch.autograd.grad(outputs=y,
                               inputs=x,
                               grad_outputs=weight,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]
    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
    return torch.mean((dydx_l2norm - 1)**2)

def label2onehot(labels, dim):
    """Convert label indices to one-hot vectors."""
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    out[np.arange(batch_size), labels.long()] = 1
    return out

class Solver(object):
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.build_model()

    def build_model(self):
        self.G = Generator(c_dim=self.config['c_dim'], repeat_num=self.config['g_repeat_num'])
        self.D = Discriminator(image_size=self.config['image_size'], c_dim=self.config['c_dim'], repeat_num=self.config['d_repeat_num'])

        self.g_optimizer = optim.Adam(self.G.parameters(), self.config['g_lr'], [self.config['beta1'], self.config['beta2']])
        self.d_optimizer = optim.Adam(self.D.parameters(), self.config['d_lr'], [self.config['beta1'], self.config['beta2']])

        self.G.to(self.device)
        self.D.to(self.device)

    def restore_model(self, resume_iters):
        print(f'Loading the trained models from step {resume_iters}...')
        G_path = os.path.join(self.config['model_save_dir'], f'{resume_iters}-G.ckpt')
        D_path = os.path.join(self.config['model_save_dir'], f'{resume_iters}-D.ckpt')
        self.G.load_state_dict(torch.load(G_path, map_location=self.device))
        self.D.load_state_dict(torch.load(D_path, map_location=self.device))

        # Restore optimizer states so training resumes with intact Adam momentum
        opt_path = os.path.join(self.config['model_save_dir'], f'{resume_iters}-opt.ckpt')
        if os.path.exists(opt_path):
            opt_state = torch.load(opt_path, map_location=self.device)
            self.g_optimizer.load_state_dict(opt_state['g_optimizer'])
            self.d_optimizer.load_state_dict(opt_state['d_optimizer'])

    def get_latest_checkpoint(self):
        models_dir = self.config['model_save_dir']
        if not os.path.exists(models_dir):
            return 0
        checkpoints = [f for f in os.listdir(models_dir) if f.endswith('-G.ckpt') and f != 'latest-G.ckpt']
        if not checkpoints:
            return 0
        iters = [int(f.split('-')[0]) for f in checkpoints]
        return max(iters)

    def train(self):
        # Data loaders: deterministic 80/20 train/val split
        data_loader, val_loader = get_loaders(self.config['image_dir'], self.config['image_size'],
                                              self.config['batch_size'], self.config.get('num_workers', 4))

        # Losses
        criterion_cls = nn.CrossEntropyLoss()
        criterion_rec = nn.L1Loss()

        start_iters = self.get_latest_checkpoint()
        if start_iters > 0:
            self.restore_model(start_iters)

        print(f"Starting training from iteration {start_iters}...")
        start_time = time.time()

        data_iter = iter(data_loader)
        g_loss = None  # unset until the first n_critic-th generator step

        # Fixed batch of real images for periodic sample-grid generation
        x_fixed, _ = next(iter(data_loader))
        x_fixed = x_fixed[:8].to(self.device)

        # Fixed val batch with fixed target labels: comparable val losses
        # across save steps (logging only, no early stopping)
        x_val = None
        if val_loader is not None:
            x_val, label_val = next(iter(val_loader))
            val_gen = torch.Generator().manual_seed(42)
            label_val_trg = label_val[torch.randperm(label_val.size(0), generator=val_gen)]
            c_val_org = label2onehot(label_val, self.config['c_dim']).to(self.device)
            c_val_trg = label2onehot(label_val_trg, self.config['c_dim']).to(self.device)
            x_val = x_val.to(self.device)
            label_val = label_val.to(self.device)
            label_val_trg = label_val_trg.to(self.device)

        for i in range(start_iters, self.config['num_iters']):
            try:
                x_real, label_org = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                try:
                    x_real, label_org = next(data_iter)
                except StopIteration:
                    print("Dataset is empty. Cannot train.")
                    return

            # Generate target domain labels randomly
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]

            c_org = label2onehot(label_org, self.config['c_dim']).to(self.device)
            c_trg = label2onehot(label_trg, self.config['c_dim']).to(self.device)
            x_real = x_real.to(self.device)
            label_org = label_org.to(self.device)
            label_trg = label_trg.to(self.device)

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            out_src, out_cls = self.D(x_real)
            d_loss_real = - torch.mean(out_src)
            d_loss_cls = criterion_cls(out_cls, label_org)

            # Compute loss with fake images.
            x_fake = self.G(x_real, c_trg)
            out_src, out_cls = self.D(x_fake.detach())
            d_loss_fake = torch.mean(out_src)

            # NOTE: no AMP / GradScaler / torch.compile in this loop, ever. The
            # gradient penalty below is a double-backward - autograd.grad with
            # create_graph=True, then backward through that gradient. GradScaler
            # scales the loss, so the inner gradient norm would be computed on
            # scaled values and never unscaled; the penalty (||g||-1)^2 comes out
            # wrong by the scale factor. Nothing crashes, nothing warns; the run
            # just trains wrong. This rule is not stale. Do not "optimize" this.
            # Compute loss for gradient penalty.
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src, _ = self.D(x_hat)
            d_loss_gp = gradient_penalty(out_src, x_hat, self.device)

            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + self.config['lambda_cls'] * d_loss_cls + self.config['lambda_gp'] * d_loss_gp
            self.d_optimizer.zero_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #

            if (i+1) % self.config['n_critic'] == 0:
                # Original-to-target domain.
                x_fake = self.G(x_real, c_trg)
                out_src, out_cls = self.D(x_fake)
                g_loss_fake = - torch.mean(out_src)
                g_loss_cls = criterion_cls(out_cls, label_trg)

                # Target-to-original domain (Cycle).
                x_reconst = self.G(x_fake, c_org)
                g_loss_rec = criterion_rec(x_reconst, x_real)

                # Backward and optimize.
                g_loss = g_loss_fake + self.config['lambda_rec'] * g_loss_rec + self.config['lambda_cls'] * g_loss_cls
                self.g_optimizer.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()

            # Print out training information.
            if (i+1) % self.config['log_step'] == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                g_loss_part = f", G_loss [{g_loss.item():.4f}]" if g_loss is not None else ""
                print(f"Elapsed [{et}], Iteration [{i+1}/{self.config['num_iters']}], D_loss [{d_loss.item():.4f}]{g_loss_part}")

            # Save model checkpoints.
            if (i+1) % self.config['model_save_step'] == 0:
                G_path = os.path.join(self.config['model_save_dir'], f'{i+1}-G.ckpt')
                D_path = os.path.join(self.config['model_save_dir'], f'{i+1}-D.ckpt')
                opt_path = os.path.join(self.config['model_save_dir'], f'{i+1}-opt.ckpt')
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                torch.save({'g_optimizer': self.g_optimizer.state_dict(),
                            'd_optimizer': self.d_optimizer.state_dict()}, opt_path)
                # Keep a stable alias for the inference backend, which loads latest-G.ckpt
                latest_path = os.path.join(self.config['model_save_dir'], 'latest-G.ckpt')
                torch.save(self.G.state_dict(), latest_path)
                print(f'Saved model checkpoints into {self.config["model_save_dir"]}...')

                # D/opt checkpoints exist only to resume training; G checkpoints
                # feed evaluation curves and are kept forever. get_latest_checkpoint()
                # discovers the resume point via -G.ckpt, so pruning D/opt here
                # cannot break checkpoint discovery. This runs after the save
                # above so an interrupted save never leaves zero resume state.
                keep_last = self.config.get('keep_last')
                if keep_last:
                    opt_iters = sorted(
                        int(f.split('-')[0])
                        for f in os.listdir(self.config['model_save_dir'])
                        if f.endswith('-opt.ckpt')
                    )
                    for old_iter in opt_iters[:-keep_last]:
                        for suffix in ('-D.ckpt', '-opt.ckpt'):
                            old_path = os.path.join(self.config['model_save_dir'], f'{old_iter}{suffix}')
                            if os.path.exists(old_path):
                                os.remove(old_path)

                # Val losses on the fixed val batch (gradient penalty needs
                # grads, so the D val loss omits the GP term)
                if x_val is not None:
                    with torch.no_grad():
                        out_src_val, out_cls_val = self.D(x_val)
                        x_val_fake = self.G(x_val, c_val_trg)
                        out_src_vfake, out_cls_vfake = self.D(x_val_fake)
                        d_val = (- torch.mean(out_src_val) + torch.mean(out_src_vfake)
                                 + self.config['lambda_cls'] * criterion_cls(out_cls_val, label_val))
                        g_val = (- torch.mean(out_src_vfake)
                                 + self.config['lambda_rec'] * criterion_rec(self.G(x_val_fake, c_val_org), x_val)
                                 + self.config['lambda_cls'] * criterion_cls(out_cls_vfake, label_val_trg))
                    print(f"Val [iter {i+1}] D_loss [{d_val.item():.4f}], G_loss [{g_val.item():.4f}]")

                # Sample grid: fixed real images translated to every target age group
                with torch.no_grad():
                    x_concat = [x_fixed]
                    for age in range(self.config['c_dim']):
                        c_trg = label2onehot(torch.full((x_fixed.size(0),), age, dtype=torch.long), self.config['c_dim']).to(self.device)
                        x_concat.append(self.G(x_fixed, c_trg))
                    x_concat = torch.cat(x_concat, dim=3)
                    x_concat = ((x_concat + 1) / 2).clamp(0, 1)  # denormalize
                    sample_dir = os.path.join(os.path.dirname(self.config['model_save_dir']), 'samples')
                    os.makedirs(sample_dir, exist_ok=True)
                    save_image(x_concat, os.path.join(sample_dir, f'{i+1}.png'), nrow=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='WGAN-GP training for StarGAN facial age progression')

    # Training
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-iters', type=int, default=100000)
    parser.add_argument('--n-critic', type=int, default=5)
    parser.add_argument('--g-lr', type=float, default=0.0001)
    parser.add_argument('--d-lr', type=float, default=0.0001)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--lambda-cls', type=float, default=1)
    parser.add_argument('--lambda-rec', type=float, default=10)
    parser.add_argument('--lambda-gp', type=float, default=10)

    # IO / logging
    parser.add_argument('--image-dir', type=str,
                        default=os.path.join(_ML_CORE_DIR, 'data/utkface'))
    parser.add_argument('--model-save-dir', type=str,
                        default=os.path.join(_ML_CORE_DIR, 'models'))
    parser.add_argument('--log-step', type=int, default=10)
    parser.add_argument('--model-save-step', type=int, default=1000)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--keep-last', type=int, default=3,
                        help='D/opt checkpoint pairs kept for resume; G checkpoints are never pruned')

    args = parser.parse_args()
    config = vars(args)

    # Seeds weight init, shuffle order, and augmentation draws. Deliberately
    # does NOT touch the train/val split seed (dataset.py, pinned 42) or the
    # fixed val-batch generator (train.py, pinned 42) - the val set must stay
    # identical across runs regardless of --seed, or val losses stop being
    # comparable across experiments.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Architecture is deliberately not exposed as flags: a checkpoint trained
    # with a different c_dim / image_size / repeat_num will not load in
    # inference.py, and c_dim is welded to the 6 age domains in dataset.py.
    config.update({
        'c_dim': 6,
        'image_size': 128,
        'g_repeat_num': 6,
        'd_repeat_num': 6,
    })

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA not available - refusing to train on CPU.\n"
            "The default PyPI torch wheel is CPU-only on Windows. Install the CUDA build:\n"
            "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130"
        )

    # Input shape is fixed every iteration (128x128, constant batch size), which
    # is exactly the case cudnn.benchmark exists for: it lets cuDNN profile and
    # pick the fastest conv algorithm for that one shape instead of the default
    # heuristic. Trade-off: algorithm selection becomes non-deterministic, so
    # runs stay seed-reproducible in init/data order but are NOT bit-exact.
    # Not measured on this machine (no CUDA available here) - this is standard
    # practice for fixed-shape training, not a claimed speedup.
    torch.backends.cudnn.benchmark = True

    os.makedirs(config['model_save_dir'], exist_ok=True)

    solver = Solver(config)
    solver.train()
