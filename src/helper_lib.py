import torch
import torch.nn as nn

# -----------------------------
# 1 Generator
# -----------------------------
class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.fc = nn.Linear(z_dim, 7*7*128) 
        self.bn1 = nn.BatchNorm1d(7*7*128)
        self.relu = nn.ReLU(True)

        # ConvTranspose2D layer
        self.deconv = nn.Sequential(
            # 128 -> 64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 64 -> 1
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = self.bn1(x)
        x = self.relu(x)
        x = x.view(-1, 128, 7, 7)  # reshape
        x = self.deconv(x)
        return x  # (batch_size, 1, 28, 28)


# -----------------------------
# 2 Discriminator
# -----------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            # (1,28,28)
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fc = nn.Linear(128*7*7, 1)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 128*7*7)
        x = self.fc(x)
        return x  


def build_generator(z_dim=100):
    return Generator(z_dim=z_dim)

def build_discriminator():
    return Discriminator()
