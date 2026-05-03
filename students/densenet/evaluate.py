"""
Evaluate teacher and both DenseNet-121 student checkpoints on the ImageNette val set.

Downloads ImageNette from fast.ai if not already present.
Writes students/densenet/results/metrics/densenet_accuracy.csv.
"""
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

_HERE = Path(__file__).parent       # students/densenet/
_ROOT = _HERE.parent.parent         # project root

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

TEACHER_CKPT = _ROOT / "teacher" / "checkpoints" / "teacher_finetuned.pth"

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


def main():
    print(f"Device  : {DEVICE}")
    print(f"Teacher : {TEACHER_CKPT}\n")
    download_imagenette()

    val_ds = tv_datasets.ImageFolder(_IMAGENETTE_DIR / "val", transform=transform)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )
    print(f"Val samples: {len(val_ds):,}\n")

    models_cfg = [
        ("teacher_resnet50",          TEACHER_CKPT,                              load_resnet50),
        ("student_kd_densenet",       _HERE / "checkpoints" / "densenet_kd.pth",       load_densenet),
        ("student_baseline_densenet", _HERE / "checkpoints" / "densenet_baseline.pth", load_densenet),
    ]

    results_dir = _HERE / "results"
    (results_dir / "metrics").mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "metrics" / "densenet_accuracy.csv"

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
