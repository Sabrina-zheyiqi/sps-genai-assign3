import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

from src.helper_lib import build_generator, build_discriminator


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():

    z_dim = 100
    lr = 0.0002
    batch_size = 128
    epochs = 10


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # [-1, 1]
    ])
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(root='./data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True
    )

 
    G = build_generator(z_dim).to(device)
    D = build_discriminator().to(device)

    criterion = nn.BCEWithLogitsLoss()
    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    fixed_noise = torch.randn(64, z_dim, device=device)

    for epoch in range(epochs):
        G.train(); D.train()
        for batch_idx, (real, _) in enumerate(tqdm(dataloader)):
            real = real.to(device)
            bs = real.size(0)

            noise = torch.randn(bs, z_dim, device=device)
            fake = G(noise).detach()

            real_labels = torch.ones(bs, device=device)
            fake_labels = torch.zeros(bs, device=device)

            D_real = D(real).squeeze()
            D_fake = D(fake).squeeze()

            loss_D_real = criterion(D_real, real_labels)
            loss_D_fake = criterion(D_fake, fake_labels)
            loss_D = loss_D_real + loss_D_fake

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            noise = torch.randn(bs, z_dim, device=device)
            fake = G(noise)
            output = D(fake).squeeze()
            loss_G = criterion(output, real_labels)  # 希望被判为真

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

        print(f"Epoch [{epoch+1}/{epochs}]  Loss D: {loss_D:.4f}, Loss G: {loss_G:.4f}")

     
        with torch.no_grad():
            fake_images = G(fixed_noise).detach().cpu()
            save_image((fake_images + 1) / 2, f"outputs/samples/epoch_{epoch+1:03d}.png", nrow=8)

        torch.save(G.state_dict(), f"outputs/checkpoints/G_epoch_{epoch+1:03d}.pt")
        torch.save(D.state_dict(), f"outputs/checkpoints/D_epoch_{epoch+1:03d}.pt")

if __name__ == "__main__":
    os.makedirs("outputs/samples", exist_ok=True)
    os.makedirs("outputs/checkpoints", exist_ok=True)
    main()
