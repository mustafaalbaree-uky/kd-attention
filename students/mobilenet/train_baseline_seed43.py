"""
Baseline training: MobileNetV2 on ImageNette - seed 43 variant.

Identical to train_baseline.py in every way except the random seed,
which is set to 43 (instead of 42). Used to establish a floor reference:
the expected divergence between two models differing only by random seed.
"""
import csv
import random
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as tv_datasets
import torchvision.models as models
import torchvision.transforms as transforms
import yaml
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent          # students/mobilenet/
_ROOT = _HERE.parent.parent            # project root

with open(_ROOT / "config.yaml") as f:
    cfg = yaml.safe_load(f)

SEED        = 43
NUM_CLASSES = cfg["training"]["num_classes"]
IMG_SIZE    = cfg["dataset"]["image_size"]
BATCH_SIZE  = cfg["dataset"]["batch_size"]

NUM_EPOCHS  = 30
LR          = 0.01

_DATA_DIR       = _ROOT / "data"
_IMAGENETTE_DIR = _DATA_DIR / "imagenette2-320"
_IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"

# ── Reproducibility ────────────────────────────────────────────────────────────
random.seed(43)
np.random.seed(43)
torch.manual_seed(43)
torch.cuda.manual_seed_all(43)

# ── Transforms ─────────────────────────────────────────────────────────────────
_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    _norm,
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    _norm,
])


# ── Dataset download ───────────────────────────────────────────────────────────
def download_imagenette():
    if _IMAGENETTE_DIR.exists():
        return
    _DATA_DIR.mkdir(exist_ok=True)
    tgz = _DATA_DIR / "imagenette2-320.tgz"
    print("Downloading ImageNette (~330 MB) …")
    urllib.request.urlretrieve(
        _IMAGENETTE_URL, tgz,
        reporthook=lambda n, bs, ts: print(
            f"\r  {min(n * bs / ts * 100, 100):.1f}%", end="", flush=True
        ),
    )
    print("\nExtracting …")
    with tarfile.open(tgz) as t:
        t.extractall(_DATA_DIR)
    tgz.unlink()
    print("Done.\n")


# ── Model ──────────────────────────────────────────────────────────────────────
def load_student(num_classes):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


# ── Train / eval helpers ───────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    sum_loss = correct = seen = 0

    for images, labels in tqdm(loader, leave=False, desc="  train"):
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss   = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs        = labels.size(0)
        sum_loss += loss.item() * bs
        correct  += (logits.argmax(1) == labels).sum().item()
        seen     += bs

    return sum_loss / seen, correct / seen


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = seen = 0
    for images, labels in tqdm(loader, leave=False, desc="  val  "):
        images, labels = images.to(device), labels.to(device)
        correct += (model(images).argmax(1) == labels).sum().item()
        seen    += labels.size(0)
    return correct / seen


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    download_imagenette()

    g = torch.Generator().manual_seed(43)
    train_ds   = tv_datasets.ImageFolder(_IMAGENETTE_DIR / "train", transform=train_transform)
    val_ds     = tv_datasets.ImageFolder(_IMAGENETTE_DIR / "val",   transform=val_transform)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True, generator=g
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True
    )
    print(f"Train: {len(train_ds):,}  |  Val: {len(val_ds):,}")

    student = load_student(NUM_CLASSES).to(device)

    optimizer = torch.optim.SGD(student.parameters(), lr=LR,
                                momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    ckpt_dir    = _HERE / "checkpoints"
    results_dir = _HERE / "results"
    ckpt_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    (results_dir / "logs").mkdir(exist_ok=True)
    csv_path  = results_dir / "logs" / "mobilenet_baseline_seed43_training_log.csv"
    ckpt_path = ckpt_dir / "mobilenet_baseline_seed43.pth"

    with csv_path.open("w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_acc"])

    best_val_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        tr_loss, tr_acc = train_one_epoch(student, train_loader, optimizer, device)
        val_acc = evaluate(student, val_loader, device)
        scheduler.step()

        print(f"  loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  val_acc={val_acc:.4f}")

        with csv_path.open("a", newline="") as f:
            csv.writer(f).writerow([
                epoch, f"{tr_loss:.6f}", f"{tr_acc:.6f}", f"{val_acc:.6f}",
            ])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(student.state_dict(), ckpt_path)
            print(f"  Saved checkpoint (val_acc={val_acc:.4f})")

    print(f"\nTraining complete. Best val_acc: {best_val_acc:.4f}")
    print(f"Checkpoint -> {ckpt_path}")
    print(f"Log        -> {csv_path}")


if __name__ == "__main__":
    main()
