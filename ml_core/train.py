import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
from model import Generator, Discriminator
from dataset import get_loader

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
        # Data loader
        data_loader = get_loader(self.config['image_dir'], self.config['image_size'], self.config['batch_size'])
        
        # Losses
        criterion_cls = nn.CrossEntropyLoss()
        criterion_rec = nn.L1Loss()
        
        start_iters = self.get_latest_checkpoint()
        if start_iters > 0:
            self.restore_model(start_iters)
        
        print(f"Starting training from iteration {start_iters}...")
        start_time = time.time()
        
        data_iter = iter(data_loader)
        
        for i in range(start_iters, self.config['num_iters']):
            try:
                x_real, label_org = next(data_iter)
            except:
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
                print(f"Elapsed [{et}], Iteration [{i+1}/{self.config['num_iters']}], D_loss [{d_loss.item():.4f}], G_loss [{g_loss.item():.4f}]")

            # Save model checkpoints.
            if (i+1) % self.config['model_save_step'] == 0:
                G_path = os.path.join(self.config['model_save_dir'], f'{i+1}-G.ckpt')
                D_path = os.path.join(self.config['model_save_dir'], f'{i+1}-D.ckpt')
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print(f'Saved model checkpoints into {self.config["model_save_dir"]}...')

import datetime
import numpy as np

if __name__ == '__main__':
    config = {
        'c_dim': 6, # 6 age groups
        'image_size': 128,
        'g_repeat_num': 6,
        'd_repeat_num': 6,
        'g_lr': 0.0001,
        'd_lr': 0.0001,
        'beta1': 0.5,
        'beta2': 0.999,
        'image_dir': 'data/utkface',
        'batch_size': 8,
        'num_iters': 100000,
        'n_critic': 5,
        'lambda_cls': 1,
        'lambda_rec': 10,
        'lambda_gp': 10,
        'log_step': 10,
        'model_save_step': 1000,
        'model_save_dir': 'models'
    }
    os.makedirs(config['model_save_dir'], exist_ok=True)
    solver = Solver(config)
    solver.train()
