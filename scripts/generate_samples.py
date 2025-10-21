import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torchvision.utils import save_image
from src.helper_lib import build_generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_from_checkpoint(ckpt_path, z_dim=100, num_samples=64):
    G = build_generator(z_dim).to(device)
    G.load_state_dict(torch.load(ckpt_path, map_location=device))
    G.eval()

    noise = torch.randn(num_samples, z_dim, device=device)
    with torch.no_grad():
        fake_images = G(noise).cpu()

    os.makedirs("outputs/generated", exist_ok=True)
    save_image((fake_images + 1) / 2, "outputs/generated/samples.png", nrow=8)
    print("Generated images saved to outputs/generated/samples.png")

if __name__ == "__main__":
    latest_ckpt = sorted(os.listdir("outputs/checkpoints"))[-1]
    generate_from_checkpoint(os.path.join("outputs/checkpoints", latest_ckpt))
