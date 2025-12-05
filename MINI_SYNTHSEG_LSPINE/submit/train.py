import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from models.unet import UNet, CombinedLoss
from synthetic_generator import SpineSynthGenerator


# ------------------------------------------------------------
# Dataset Definitions
# ------------------------------------------------------------

class SyntheticSpineDataset(Dataset):
    """
    Dataset for Model A (Synthetic-only training).
    Uses SPIDER label maps and generates synthetic MRI on-the-fly.
    """
    def __init__(self, label_dir):
        self.label_paths = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir)])
        self.generator = SpineSynthGenerator()

    def __len__(self):
        return len(self.label_paths)

    def __getitem__(self, idx):
        label_path = self.label_paths[idx]

        # Load label map
        label_map = np.array(Image.open(label_path)).astype(np.int64)

        # Generate synthetic MRI image
        img = self.generator.generate_synthetic_mri(label_map)

        # Normalize to [0,1]
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        img = torch.tensor(img).unsqueeze(0)      # (1, H, W)
        label = torch.tensor(label_map).long()    # (H, W)

        return img, label


class RealSpineDataset(Dataset):
    """
    Dataset for Model B (Real T1 MRI training).
    Expects:
      image_dir  -> real T1 MRI
      label_dir  -> segmentation labels
    """
    def __init__(self, image_dir, label_dir):
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
        self.label_paths = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir)])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load MRI & label
        img = np.array(Image.open(self.image_paths[idx])).astype(np.float32)
        label = np.array(Image.open(self.label_paths[idx])).astype(np.int64)

        # Normalize image to [0,1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        img = torch.tensor(img).unsqueeze(0)       # (1, H, W)
        label = torch.tensor(label).long()

        return img, label


# ------------------------------------------------------------
# Training Function
# ------------------------------------------------------------

def train_model(model, dataloader, device, epochs, save_path):
    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for img, label in dataloader:
            img = img.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            pred = model(img)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"[Epoch {epoch}/{epochs}] Loss: {avg_loss:.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Saved model → {save_path}")


# ------------------------------------------------------------
# Main training launcher
# ------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Directories ------------------------------
    SYN_LABEL_DIR = "data/SPIDER_labels/"              # label maps for synthetic generation
    REAL_IMG_DIR = "data/SPIDER_T1_train/images/"      # real MRI images
    REAL_LABEL_DIR = "data/SPIDER_T1_train/labels/"    # real segmentation labels

    # Hyperparameters --------------------------
    batch_size = 4
    epochs = 20
    num_classes = 14

    # Datasets --------------------------------
    print("Loading datasets...")
    syn_dataset = SyntheticSpineDataset(SYN_LABEL_DIR)
    real_dataset = RealSpineDataset(REAL_IMG_DIR, REAL_LABEL_DIR)

    syn_loader = DataLoader(syn_dataset, batch_size=batch_size, shuffle=True)
    real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=True)

    # ------------------------------------------
    # Train Model A (Synthetic-only)
    # ------------------------------------------
    print("\n=== Training Model A: Synthetic-only ===")
    model_A = UNet(in_channels=1, num_classes=num_classes, base_features=16)
    train_model(
        model=model_A,
        dataloader=syn_loader,
        device=device,
        epochs=epochs,
        save_path="models/saved/model_A_synthetic.pth"
    )

    # ------------------------------------------
    # Train Model B (Real-only)
    # ------------------------------------------
    print("\n=== Training Model B: Real-only ===")
    model_B = UNet(in_channels=1, num_classes=num_classes, base_features=16)
    train_model(
        model=model_B,
        dataloader=real_loader,
        device=device,
        epochs=epochs,
        save_path="models/saved/model_B_real.pth"
    )


if __name__ == "__main__":
    main()
