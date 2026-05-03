"""
Evaluate teacher and both student checkpoints on the ImageNette validation set.

Downloads ImageNette from fast.ai if not already present.
Writes students/<arch>/results/metrics/<arch>_accuracy.csv.

Usage (from project root):
    python shared/evaluate.py --student resnet18
    python shared/evaluate.py --student mobilenet
    python shared/evaluate.py --student densenet
"""
import argparse
import csv
import random
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as tv_datasets
import torchvision.models as models
import torchvision.transforms as transforms
import yaml
from tqdm import tqdm

_HERE = Path(__file__).parent       # shared/
_ROOT = _HERE.parent                # project root

with open(_ROOT / "config.yaml") as f:
    cfg = yaml.safe_load(f)

SEED        = cfg["training"]["seed"]
NUM_CLASSES = cfg["training"]["num_classes"]
IMG_SIZE    = cfg["dataset"]["image_size"]
BATCH_SIZE  = cfg["dataset"]["batch_size"]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
_DATA_DIR       = _ROOT / "data"
_IMAGENETTE_DIR = _DATA_DIR / "imagenette2-320"

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
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


def load_resnet50(ckpt_path):
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    m.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return m.eval().to(DEVICE)


def load_resnet18(ckpt_path):
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    m.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return m.eval().to(DEVICE)


def load_mobilenet(ckpt_path):
    m = models.mobilenet_v2(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, NUM_CLASSES)
    m.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return m.eval().to(DEVICE)


def load_densenet(ckpt_path):
    m = models.densenet121(weights=None)
    m.classifier = nn.Linear(m.classifier.in_features, NUM_CLASSES)
    m.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return m.eval().to(DEVICE)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = seen = 0
    for imgs, labels in tqdm(loader, leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        correct += (model(imgs).argmax(1) == labels).sum().item()
        seen    += labels.size(0)
    return correct / seen


_STUDENT_LOADERS = {
    "resnet18":  ("resnet18",  load_resnet18),
    "mobilenet": ("mobilenet", load_mobilenet),
    "densenet":  ("densenet",  load_densenet),
}

_STUDENT_KD_NAMES = {
    "resnet18":  "student_kd_resnet18",
    "mobilenet": "student_kd_mobilenet",
    "densenet":  "student_kd_densenet",
}

_STUDENT_BL_NAMES = {
    "resnet18":  "student_baseline_resnet18",
    "mobilenet": "student_baseline_mobilenet",
    "densenet":  "student_baseline_densenet",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", required=True,
                        choices=["resnet18", "mobilenet", "densenet"],
                        help="Student architecture to evaluate")
    args = parser.parse_args()

    arch        = args.student
    student_dir = _ROOT / "students" / arch
    _, load_student = _STUDENT_LOADERS[arch]

    teacher_ckpt  = _ROOT / "teacher" / "checkpoints" / "teacher_finetuned.pth"
    kd_ckpt       = student_dir / "checkpoints" / f"{arch}_kd.pth"
    baseline_ckpt = student_dir / "checkpoints" / f"{arch}_baseline.pth"

    print(f"Device  : {DEVICE}")
    print(f"Student : {arch}")
    print(f"Teacher : {teacher_ckpt}\n")

    download_imagenette()

    val_ds = tv_datasets.ImageFolder(_IMAGENETTE_DIR / "val", transform=transform)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )
    print(f"Val samples: {len(val_ds):,}\n")

    models_cfg = [
        ("teacher_resnet50",       teacher_ckpt,  load_resnet50),
        (_STUDENT_KD_NAMES[arch],  kd_ckpt,       load_student),
        (_STUDENT_BL_NAMES[arch],  baseline_ckpt, load_student),
    ]

    results_dir = student_dir / "results"
    (results_dir / "metrics").mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "metrics" / f"{arch}_accuracy.csv"

    rows = []
    for name, ckpt_path, loader_fn in models_cfg:
        print(f"Evaluating {name} …")
        model = loader_fn(ckpt_path)
        acc   = evaluate(model, val_loader)
        print(f"  {name}: test_accuracy = {acc:.4f}")
        rows.append((name, str(ckpt_path), acc))
        del model
        torch.cuda.empty_cache()

    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model_name", "checkpoint", "test_accuracy"])
        for name, ckpt_path, acc in rows:
            w.writerow([name, ckpt_path, f"{acc:.4f}"])

    print(f"\nSaved -> {csv_path}")


if __name__ == "__main__":
    main()
