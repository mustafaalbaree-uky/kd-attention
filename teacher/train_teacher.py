"""
Teacher fine-tuning: ResNet-50 on ImageNette.

Fine-tunes the full network (backbone + new FC head) so the teacher produces
well-calibrated soft labels for KD. Must be run before any student train_kd.py.

Saves best checkpoint to teacher/checkpoints/teacher_finetuned.pth.
All hyperparameters are read from config.yaml at the project root.
"""
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import yaml
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_HERE = Path(__file__).parent       # teacher/
_ROOT = _HERE.parent                # project root

with open(_ROOT / "config.yaml") as f:
    cfg = yaml.safe_load(f)

SEED        = cfg["training"]["seed"]
NUM_CLASSES = cfg["training"]["num_classes"]
IMG_SIZE    = cfg["dataset"]["image_size"]
BATCH_SIZE  = cfg["dataset"]["batch_size"]
DATASET     = cfg["dataset"]["name"]

NUM_EPOCHS = 15
LR         = 0.001

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

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


class ImagenetteDataset(Dataset):
    """HuggingFace ImageNette split wrapped as a torch Dataset."""

    def __init__(self, hf_split, transform):
        self.data = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self.transform(image), item["label"]


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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading dataset …")
    raw = load_dataset(DATASET, "full_size")
    train_ds = ImagenetteDataset(raw["train"],      train_transform)
    val_ds   = ImagenetteDataset(raw["validation"], val_transform)

    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True, generator=g)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    teacher = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    teacher.fc = nn.Linear(teacher.fc.in_features, NUM_CLASSES)
    teacher = teacher.to(device)

    optimizer = torch.optim.SGD(teacher.parameters(), lr=LR,
                                momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    ckpt_dir = _HERE / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    teacher_ckpt = ckpt_dir / "teacher_finetuned.pth"
    csv_path     = _HERE / "teacher_training_log.csv"

    with csv_path.open("w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_acc"])

    best_val_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        tr_loss, tr_acc = train_one_epoch(teacher, train_loader, optimizer, device)
        val_acc = evaluate(teacher, val_loader, device)
        scheduler.step()

        print(f"  loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  val_acc={val_acc:.4f}")

        with csv_path.open("a", newline="") as f:
            csv.writer(f).writerow([
                epoch, f"{tr_loss:.6f}", f"{tr_acc:.6f}", f"{val_acc:.6f}",
            ])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(teacher.state_dict(), teacher_ckpt)
            print(f"  Saved checkpoint (val_acc={val_acc:.4f})")

    print(f"\nTeacher fine-tuning complete. Best val_acc: {best_val_acc:.4f}")
    print(f"Checkpoint -> {teacher_ckpt}")
    print(f"Log        -> {csv_path}")


if __name__ == "__main__":
    main()
