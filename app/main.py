# app/main.py
from fastapi import FastAPI
import torch
from torchvision.utils import make_grid
import base64, io
from src.helper_lib import build_generator

app = FastAPI()

DEVICE = torch.device("cpu")
Z_DIM = 100
CHECKPOINT = "outputs/checkpoints/G_epoch_010.pt"

# Load generator once at startup
GEN = build_generator(Z_DIM).to(DEVICE)
GEN.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
GEN.eval()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/generate")
def generate(num: int = 16):
    z = torch.randn(num, Z_DIM, device=DEVICE)
    with torch.no_grad():
        imgs = GEN(z).cpu()

    grid = make_grid(imgs, nrow=int(num**0.5), normalize=True)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(4,4))
    plt.axis("off")
    buf = io.BytesIO()
    plt.imshow(grid.permute(1,2,0))
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close()

    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {"image_base64_png": img_base64}
