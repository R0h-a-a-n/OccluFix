# OccluFix

OccluFix is a deep learning model for facial image inpainting. Built using PyTorch, it reconstructs occluded regions in face images using a lightweight U-Net architecture. The model supports advanced loss functions and training strategies for high-quality and identity-aware generation.

---

![image](https://github.com/user-attachments/assets/69e21812-382c-4126-b069-0c61c0696ab7)

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

### Step 2: Prepare Dataset
Download CelebA and extract inside the project folder:
```
Copy
Edit
occlufix/
├── celeba/
│   └── imgs/
│       └── 000001.jpg ...
```

### Step 3: Train the Model
```bash
python train.py
```

Mixed precision (via torch.cuda.amp) is enabled by default.
Training checkpoints will be saved every epoch.

---


### Inference after training:

```python
G = InpaintingUNet().to(device)
G.load_state_dict(torch.load("unmasknet_epoch_25_globalgan.pth"))
G.eval()

with torch.no_grad():
    sample_batch = next(iter(dataloader))
    masked_imgs, masks, originals = sample_batch
    masked_imgs = masked_imgs.to(device)
    masks = masks[:, :1, :, :].to(device)
    outputs = G(masked_imgs, masks).cpu()

def show_inpainting_result(originals, masked, reconstructed):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 4, figsize=(12, 9))
    for i in range(4):
        axs[0, i].imshow(originals[i].permute(1, 2, 0))
        axs[0, i].set_title("Original")
        axs[1, i].imshow(masked[i].permute(1, 2, 0))
        axs[1, i].set_title("Masked")
        axs[2, i].imshow(reconstructed[i].permute(1, 2, 0))
        axs[2, i].set_title("Reconstructed")
        for j in range(3):
            axs[j, i].axis("off")
    plt.tight_layout()
    plt.show()

show_inpainting_result(originals, masked_imgs.cpu(), outputs)
print("Pixel range:", outputs.min().item(), outputs.max().item())

```
---

### Future Work

1. Add ArcFace identity loss for identity-aware inpainting

2. Manual sketch or edge-based guidance support

3. Live camera inpainting demo

4. Web deployment (TorchScript or ONNX)


