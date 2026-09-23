import torch
import torch.nn as nn
import torch.nn.functional as F

# InstanceNorm buffers written by checkpoints trained before the Generator
# switched to track_running_stats=False (see Generator). They are unused now.
_LEGACY_NORM_BUFFERS = ('.running_mean', '.running_var', '.num_batches_tracked')


def label2onehot(labels, dim):
    """Convert label indices to one-hot vectors."""
    return F.one_hot(labels.long(), dim).float()


def generator_state_dict(checkpoint):
    """Normalize a saved Generator checkpoint for a strict load_state_dict.

    Accepts a bare state_dict or a {'state_dict': ...} wrapper, strips the
    'module.' prefix DataParallel adds, and drops the InstanceNorm running
    stats that older checkpoints carry, so they keep loading unchanged.
    """
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    state_dict = {}
    for k, v in checkpoint.items():
        if k.startswith('module.'):
            k = k[len('module.'):]
        if not k.endswith(_LEGACY_NORM_BUFFERS):
            state_dict[k] = v
    return state_dict


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=False)
        )

    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):
    """StarGAN Generator Network.

    InstanceNorm layers do not track running stats. With tracking on, train()
    normalizes each image by its own statistics but eval() switches to the
    running averages, so the served model (inference.py calls eval()) would
    not compute what was trained, and the training sample grids (train mode)
    would not show what the backend serves. Without tracking, both modes use
    per-image statistics and are identical.
    """
    def __init__(self, conv_dim=64, c_dim=6, repeat_num=6):
        super(Generator, self).__init__()

        layers = []
        # Initial Conv
        layers.append(nn.Conv2d(3 + c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=False))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=False))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck (Residual layers)
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=False))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        # Output layer
        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())

        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        return self.main(x)


class Discriminator(nn.Module):
    """StarGAN Discriminator Network."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=6, repeat_num=6):
        super(Discriminator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        self.main = nn.Sequential(*layers)
        
        # Output 1: Real vs Fake (PatchGAN)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Output 2: Domain classification
        kernel_size = image_size // (2**repeat_num)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))

