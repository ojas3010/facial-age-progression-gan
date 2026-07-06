import torch
from torchvision import transforms
from PIL import Image
import os
import sys

# Ensure ml_core is in path for imports to work if run from backend
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_core.model import Generator
import numpy as np

def label2onehot(labels, dim):
    """Convert label indices to one-hot vectors."""
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    out[np.arange(batch_size), labels.long()] = 1
    return out

class AgeProgressor:
    def __init__(self, model_path=None, c_dim=6, image_size=128):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.c_dim = c_dim
        self.image_size = image_size
        
        self.G = Generator(c_dim=c_dim, repeat_num=6)
        
        if model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                # Remove 'module.' prefix if present (from DataParallel)
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('module.'):
                        new_state_dict[k[7:]] = v
                    else:
                        new_state_dict[k] = v
                
                self.G.load_state_dict(new_state_dict, strict=False)
                print(f"Loaded model from {model_path}")
            except Exception as e:
                print(f"Warning: Failed to load model weights: {e}")
                print("Using uninitialized weights (results will be noise).")
        else:
            print("Warning: Model file not found. Using uninitialized weights (results will be noise).")
            
        self.G.to(self.device)
        self.G.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def progress_age(self, image_pil, target_age_group):
        """
        image_pil: PIL Image
        target_age_group: int (0 to 5)
        Returns: PIL Image
        """
        # Prepare input
        x = self.transform(image_pil).unsqueeze(0).to(self.device)
        
        # Prepare target domain label
        target_label = torch.tensor([target_age_group])
        c_trg = label2onehot(target_label, self.c_dim).to(self.device)
        
        # Inference
        with torch.no_grad():
            x_fake = self.G(x, c_trg)
            
        # Denormalize and convert back to PIL
        x_fake = (x_fake.squeeze(0).cpu() + 1) / 2.0
        x_fake = x_fake.clamp(0, 1)
        x_fake = transforms.ToPILImage()(x_fake)
        
        return x_fake

# For testing independently
if __name__ == '__main__':
    # Initialize without weights
    progressor = AgeProgressor()
    # Create a dummy image
    dummy_img = Image.fromarray(np.uint8(np.random.rand(256, 256, 3) * 255))
    # Target age group 5 (50+)
    out_img = progressor.progress_age(dummy_img, 5)
    print(f"Generated image size: {out_img.size}")
