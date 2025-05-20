# OccluFix
A lightweight PyTorch model for inpainting and reconstructing occluded face images using UNet and mixed precision training.

# OccluFix 🎭
Occufix is a PyTorch-based deep learning model for face inpainting — reconstructing occluded or masked regions of face images. Trained on the CelebA dataset with custom masking, UnmaskNet uses a lightweight U-Net architecture and supports mixed-precision training for fast performance on GPUs like NVIDIA T4.

---

## Features

-  Masked face reconstruction using deep learning
-  Random + manual occlusion support
-  Fast training with mixed precision (`torch.amp`)
-  Uses CelebA dataset (configurable for custom data)
-  Modular PyTorch code for easy extension

---

## Model Architecture

UnmaskNet is a simple U-Net-like encoder-decoder:
- Input: Masked image + binary mask (4 channels total)
- Output: Reconstructed RGB image

---

## Dataset

- **CelebA** dataset (~200k face images)
- Only 50k used for training (for speed)
- Preprocessing: resized to 256×256, normalized, masked on-the-fly

---

## Training

```bash
# Clone repo
git clone https://github.com/R0h-a-a-n/OccluFix.git
cd OccluFix

# Install dependencies
pip install torch torchvision tqdm facenet-pytorch

# Train model
python train.py
