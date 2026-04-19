"""
classifier/train_classifier.py

Train the three-class subfigure classifier used in MatFigCtx.

Classes:
    0 - line_plot       (property curves, stress-strain, etc.)
    1 - xrd             (X-ray diffraction patterns)
    2 - microstructure  (SEM, TEM, OM, EBSD, etc.)

Training setup (as reported in the paper):
    - Base model : ResNet-50 pre-trained on ImageNet-1K
    - Backbone   : frozen (only the FC head is trained)
    - Dataset    : 342 labelled panel images, 114 per class
    - Split      : stratified 80/20 train/test
    - Augment    : random flip, rotation ±15°, colour jitter
    - Optimiser  : Adam, lr=1e-3, weight_decay=1e-4
    - Epochs     : up to 30, early stopping (patience=5)
    - Batch size : 16

Output:
    Saves  classifier/image_classifier.pt
    Format: {"model_state_dict": ..., "class_names": [...]}
    This format is expected by core/subfig_classifier.py.

Usage:
    python classifier/train_classifier.py --data-dir /path/to/annotated_panels
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

CLASS_NAMES = ["line_plot", "xrd", "microstructure"]

# ── Augmentation ──────────────────────────────────────────
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def build_model(num_classes: int) -> nn.Module:
    """ResNet-50 with frozen backbone and a fresh FC head."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    for param in model.parameters():      # freeze backbone
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)   # trainable head
    return model


def accuracy(logits, labels):
    return (logits.argmax(dim=1) == labels).float().mean().item()


def train(data_dir: str, output_path: str,
          batch_size: int = 16, lr: float = 1e-3,
          weight_decay: float = 1e-4, max_epochs: int = 30,
          patience: int = 5, val_ratio: float = 0.2,
          seed: int = 42):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    # ── Dataset ───────────────────────────────────────────
    # Expected layout: data_dir/line_plot/*.jpg  etc.
    full_dataset = datasets.ImageFolder(data_dir, transform=TRAIN_TRANSFORM)
    class_names  = full_dataset.classes
    print(f"[INFO] Classes found: {class_names}")
    print(f"[INFO] Total samples: {len(full_dataset)}")

    n_val   = int(len(full_dataset) * val_ratio)
    n_train = len(full_dataset) - n_val
    train_set, val_set = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )
    # Apply val transform to the validation split
    val_set.dataset.transform = VAL_TRANSFORM

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    # ── Model ─────────────────────────────────────────────
    model     = build_model(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.fc.parameters(), lr=lr, weight_decay=weight_decay
    )

    # ── Training loop ─────────────────────────────────────
    best_val_loss = float("inf")
    no_improve    = 0

    for epoch in range(1, max_epochs + 1):
        # Train
        model.train()
        train_loss = train_acc = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(imgs)
            train_acc  += accuracy(logits, labels) * len(imgs)

        train_loss /= n_train
        train_acc  /= n_train

        # Validate
        model.eval()
        val_loss = val_acc = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits    = model(imgs)
                val_loss += criterion(logits, labels).item() * len(imgs)
                val_acc  += accuracy(logits, labels) * len(imgs)
        val_loss /= n_val
        val_acc  /= n_val

        print(f"Epoch {epoch:3d}/{max_epochs}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve    = 0
            ckpt = {
                "model_state_dict": model.state_dict(),
                "class_names":      class_names,
                "epoch":            epoch,
                "val_acc":          val_acc,
            }
            torch.save(ckpt, output_path)
            print(f"  → Saved best checkpoint (val_loss={best_val_loss:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

    print(f"\n[Done] Best checkpoint saved to: {output_path}")
    print(f"       val_acc={ckpt['val_acc']:.4f}  epoch={ckpt['epoch']}")


def main():
    parser = argparse.ArgumentParser(description="Train MatFigCtx subfigure classifier")
    parser.add_argument("--data-dir",   required=True,
                        help="Root folder with one subfolder per class "
                             "(line_plot/, xrd/, microstructure/)")
    parser.add_argument("--output",     default="classifier/image_classifier.pt",
                        help="Path to save the checkpoint")
    parser.add_argument("--batch-size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--patience",   type=int,   default=5)
    parser.add_argument("--val-ratio",  type=float, default=0.2)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    os.makedirs(Path(args.output).parent, exist_ok=True)
    train(
        data_dir    = args.data_dir,
        output_path = args.output,
        batch_size  = args.batch_size,
        lr          = args.lr,
        max_epochs  = args.epochs,
        patience    = args.patience,
        val_ratio   = args.val_ratio,
        seed        = args.seed,
    )


if __name__ == "__main__":
    main()
