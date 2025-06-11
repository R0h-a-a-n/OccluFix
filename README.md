# OccluFix

OccluFix is a deep learning model for facial image inpainting. Built using PyTorch, it reconstructs occluded regions in face images using a lightweight U-Net architecture. The model supports advanced loss functions and training strategies for high-quality and identity-aware generation.

---

## Features

- Inpaints occluded regions in face images
- Supports PatchGAN and Global Discriminator adversarial losses
- Perceptual, style, total variation, and L1 loss integration
- Mixed precision training with `torch.amp` for faster performance
- Modular and extensible PyTorch implementation
- Compatible with custom datasets and occlusion strategies

---

## Model Architecture

OccluFix is based on a modified U-Net encoder-decoder design:

- **Input**: Masked RGB image + binary mask (4 channels total)
- **Output**: Reconstructed RGB image
- **Losses Used**:
  - L1 Loss (reconstruction)
  - VGG-based Perceptual Loss
  - Total Variation Loss
  - Patch Discriminator Loss
  - Global Discriminator Loss
  - Style Loss (Gram matrix-based)

---

## Dataset

- **CelebA** face dataset (~200,000 images)
- Preprocessing includes:
  - Resizing to 256×256
  - On-the-fly masking (random rectangular occlusions)
- Training subset:
  - Typically 100k to 150k images for faster training
- Custom datasets can be used with the same pipeline

---

## Training

### Step 1: Clone and Setup

```bash
git clone https://github.com/R0h-a-a-n/OccluFix.git
cd OccluFix
pip install torch torchvision tqdm facenet-pytorch
```

Step 2: Prepare Dataset
Download CelebA and extract inside the project folder:

Copy
Edit
occlufix/
├── celeba/
│   └── imgs/
│       └── 000001.jpg ...

Step 3: Train the Model
bash
Copy
Edit
python train.py
Mixed precision (via torch.cuda.amp) is enabled by default. Training checkpoints will be saved every epoch.

Inference
After training:

python
Copy
Edit
from model import InpaintingUNet
from utils import show_inpainting_result
import torch

model = InpaintingUNet().to(device)
model.load_state_dict(torch.load("unmasknet_epoch_25_globalgan.pth"))
model.eval()

# Run inference using dataloader or custom image
Future Work
Add ArcFace identity loss for identity-aware inpainting

Manual sketch or edge-based guidance support

Live camera inpainting demo

Web deployment (TorchScript or ONNX)

