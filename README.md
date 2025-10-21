# Assignment 3 – GAN Architecture (MNIST Handwritten Digit Generation)

## 🧩 Architecture

### Generator 
| Layer | Input → Output | Parameters | Activation |
|-------|----------------|-------------|-------------|
| Fully Connected | (BATCH_SIZE, 100) → (BATCH_SIZE, 7×7×128) | - | ReLU |
| Reshape | 7×7×128 | - | - |
| ConvTranspose2D | 128 → 64, k=4, s=2, p=1 → 14×14 | BatchNorm2D + ReLU |
| ConvTranspose2D | 64 → 1, k=4, s=2, p=1 → 28×28 | Tanh |

### Discriminator
| Layer | Input → Output | Parameters | Activation |
|-------|----------------|-------------|-------------|
| Conv2D | 1 → 64, k=4, s=2, p=1 → 14×14 | - | LeakyReLU(0.2) |
| Conv2D | 64 → 128, k=4, s=2, p=1 → 7×7 | BatchNorm2D + LeakyReLU(0.2) |
| Flatten + Linear | 128×7×7 → 1 | - | Sigmoid |


## ⚙️ Training Configuration

| Parameter | Value |
|------------|--------|
| Dataset | MNIST (28×28 grayscale) |
| Latent dimension (z) | 100 |
| Optimizer | Adam (lr=0.0002, betas=(0.5, 0.999)) |
| Loss function | Binary Cross Entropy |
| Batch size | 128 |
| Epochs | 10 |
| Device | CPU / MPS (Apple Silicon) |

### Command to Train and Generate New Digits:
```bash
python scripts/train_mnist_gan.py
python scripts/generate_samples.py
