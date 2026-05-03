"""
Knowledge Distillation training: MobileNetV2 student <- frozen ResNet-50 teacher.

KD loss = alpha * T^2 * KL(student_soft || teacher_soft)
        + (1 - alpha) * CrossEntropy(student_logits, hard_labels)

Teacher is fully frozen. Checkpoint saved only when val_acc improves.
All hyperparameters are read from config.yaml at the project root.
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

_HERE = Path(__file__).parent       # students/mobilenet/
_ROOT = _HERE.parent.parent         # project root

with open(_ROOT / "config.yaml") as f:
    cfg = yaml.safe_load(f)

SEED        = cfg["training"]["seed"]
NUM_CLASSES = cfg["training"]["num_classes"]
T           = cfg["training"]["kd_temperature"]
ALPHA       = cfg["training"]["kd_alpha"]
IMG_SIZE    = cfg["dataset"]["image_size"]
BATCH_SIZE  = cfg["dataset"]["batch_size"]

NUM_EPOCHS = 30
LR         = 0.01

_DATA_DIR       = _ROOT / "data"
_IMAGENETTE_DIR = _DATA_DIR / "imagenette2-320"
_IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"

TEACHER_CKPT = _ROOT / "teacher" / "checkpoints" / "teacher_finetuned.pth"

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


def load_teacher(num_classes):
    if not TEACHER_CKPT.exists():
        raise FileNotFoundError(
            f"{TEACHER_CKPT} not found. Run teacher/train_teacher.py first."
        )
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(TEACHER_CKPT, map_location="cpu"))
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_student(num_classes):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def kd_loss(s_logits, t_logits, labels, T, alpha):
    loss_hard  = F.cross_entropy(s_logits, labels)
    s_log_soft = F.log_softmax(s_logits / T, dim=1)
    t_soft     = F.softmax(t_logits / T, dim=1)
    loss_kd    = F.kl_div(s_log_soft, t_soft, reduction="batchmean") * (T ** 2)
    return alpha * loss_kd + (1.0 - alpha) * loss_hard, loss_kd, loss_hard


def train_one_epoch(student, teacher, loader, optimizer, device, T, alpha):
    student.train()
    sum_total = sum_kd = sum_hard = correct = seen = 0

    for images, labels in tqdm(loader, leave=False, desc="  train"):
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            t_logits = teacher(images)

        s_logits = student(images)
        loss, l_kd, l_hard = kd_loss(s_logits, t_logits, labels, T, alpha)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = labels.size(0)
        sum_total += loss.item() * bs
        sum_kd    += l_kd.item() * bs
        sum_hard  += l_hard.item() * bs
        correct   += (s_logits.argmax(1) == labels).sum().item()
        seen      += bs

    return sum_total / seen, sum_kd / seen, sum_hard / seen, correct / seen


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
    print(f"Device : {device}")
    print(f"Teacher: {TEACHER_CKPT}\n")

    download_imagenette()

    g = torch.Generator().manual_seed(SEED)
    train_ds = tv_datasets.ImageFolder(_IMAGENETTE_DIR / "train", transform=train_transform)
    val_ds   = tv_datasets.ImageFolder(_IMAGENETTE_DIR / "val",   transform=val_transform)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True, generator=g
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True
    )
    print(f"Train: {len(train_ds):,}  |  Val: {len(val_ds):,}")

    teacher = load_teacher(NUM_CLASSES).to(device).eval()
    student = load_student(NUM_CLASSES).to(device)

    optimizer = torch.optim.SGD(student.parameters(), lr=LR,
                                momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    ckpt_dir    = _HERE / "checkpoints"
    results_dir = _HERE / "results"
    ckpt_dir.mkdir(exist_ok=True)
    (results_dir / "logs").mkdir(parents=True, exist_ok=True)

    student_ckpt = ckpt_dir / "mobilenet_kd.pth"
    csv_path     = results_dir / "logs" / "mobilenet_kd_training_log.csv"

    with csv_path.open("w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss_total", "train_loss_kd",
             "train_loss_hard", "train_acc", "val_acc"]
        )

    best_val_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        tr_loss, tr_kd, tr_hard, tr_acc = train_one_epoch(
            student, teacher, train_loader, optimizer, device, T, ALPHA
        )
        val_acc = evaluate(student, val_loader, device)
        scheduler.step()

        print(f"  loss={tr_loss:.4f}  kd={tr_kd:.4f}  hard={tr_hard:.4f}"
              f"  train_acc={tr_acc:.4f}  val_acc={val_acc:.4f}")

        with csv_path.open("a", newline="") as f:
            csv.writer(f).writerow([
                epoch, f"{tr_loss:.6f}", f"{tr_kd:.6f}",
                f"{tr_hard:.6f}", f"{tr_acc:.6f}", f"{val_acc:.6f}",
            ])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(student.state_dict(), student_ckpt)
            print(f"  Saved checkpoint (val_acc={val_acc:.4f})")

    print(f"\nTraining complete. Best val_acc: {best_val_acc:.4f}")
    print(f"Checkpoint -> {student_ckpt}")
    print(f"Log        -> {csv_path}")


if __name__ == "__main__":
    main()
