"""
Grad-CAM floor generation for ResNet-18: seed-42 vs seed-43 baselines.

Loads both baseline checkpoints, generates Grad-CAM for every ImageNette
validation image, and saves two normalized maps per image to arrays_seed43/.
Keys: baseline_seed42, baseline_seed43.

Run after train_baseline.py and train_baseline_seed43.py.
"""
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as tv_datasets
import torchvision.models as models
import torchvision.transforms as transforms
from pytorch_grad_cam import GradCAM

_HERE = Path(__file__).parent       # students/resnet18/
_ROOT = _HERE.parent.parent         # project root

NUM_CLASSES = 10
IMG_SIZE    = 224

CKPT_SEED42 = _HERE / "checkpoints" / "resnet18_baseline.pth"
CKPT_SEED43 = _HERE / "checkpoints" / "resnet18_baseline_seed43.pth"

DATA_DIR       = _ROOT / "data"
IMAGENETTE_DIR = DATA_DIR / "imagenette2-320"
IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"

OUT_ARRAYS_SEED43 = _HERE / "results" / "gradcam_full" / "arrays_seed43"

IMAGENETTE_LABELS = {
    "n01440764": "tench",           "n02102040": "english_springer",
    "n02979186": "cassette_player", "n03000684": "chain_saw",
    "n03028079": "church",          "n03394916": "french_horn",
    "n03417042": "garbage_truck",   "n03425413": "gas_pump",
    "n03445777": "golf_ball",       "n03888257": "parachute",
}

_MEAN  = np.array([0.485, 0.456, 0.406])
_STD   = np.array([0.229, 0.224, 0.225])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=_MEAN.tolist(), std=_STD.tolist()),
])


def download_imagenette():
    if IMAGENETTE_DIR.exists():
        return
    DATA_DIR.mkdir(exist_ok=True)
    tgz = DATA_DIR / "imagenette2-320.tgz"
    print("Downloading ImageNette (~330 MB) …")
    urllib.request.urlretrieve(
        IMAGENETTE_URL, tgz,
        reporthook=lambda n, bs, ts: print(
            f"\r  {min(n * bs / ts * 100, 100):.1f}%", end="", flush=True
        ),
    )
    print("\nExtracting …")
    with tarfile.open(tgz) as t:
        t.extractall(DATA_DIR)
    tgz.unlink()
    print("Done.\n")


def load_resnet18(path):
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.eval().to(DEVICE)


def compute_raw_gradcam(cam_obj, model, input_tensor):
    with torch.no_grad():
        model(input_tensor)
    cam_obj(input_tensor=input_tensor, targets=None)
    acts  = cam_obj.activations_and_grads.activations[0].cpu().numpy()
    grads = cam_obj.activations_and_grads.gradients[0].cpu().numpy()
    weights = np.mean(grads, axis=(2, 3), keepdims=True)
    return np.maximum(np.sum(weights * acts, axis=1)[0], 0.0)


def normalize_map(raw):
    total = raw.sum()
    return raw / total if total > 0 else np.ones_like(raw) / raw.size


def main():
    print(f"Device  : {DEVICE}")
    print(f"Seed-42 : {CKPT_SEED42}")
    print(f"Seed-43 : {CKPT_SEED43}\n")

    download_imagenette()
    OUT_ARRAYS_SEED43.mkdir(parents=True, exist_ok=True)

    dataset     = tv_datasets.ImageFolder(IMAGENETTE_DIR / "val", transform=transform)
    class_names = [IMAGENETTE_LABELS.get(c, c) for c in dataset.classes]
    n_total     = len(dataset)
    print(f"Val images: {n_total:,}\n")

    print("Loading models …")
    model_seed42 = load_resnet18(CKPT_SEED42)
    model_seed43 = load_resnet18(CKPT_SEED43)

    cam_seed42 = GradCAM(model=model_seed42, target_layers=[model_seed42.layer4[-1]])
    cam_seed43 = GradCAM(model=model_seed43, target_layers=[model_seed43.layer4[-1]])

    class_counter = {}
    for ds_idx in range(n_total):
        img_tensor, true_label = dataset[ds_idx]
        input_tensor = img_tensor.unsqueeze(0).to(DEVICE)
        true_class   = class_names[true_label]
        class_counter[true_class] = class_counter.get(true_class, 0) + 1
        stem = f"{true_class}_{class_counter[true_class]:04d}"

        raw42 = compute_raw_gradcam(cam_seed42, model_seed42, input_tensor)
        raw43 = compute_raw_gradcam(cam_seed43, model_seed43, input_tensor)

        np.savez(
            OUT_ARRAYS_SEED43 / f"{stem}.npz",
            baseline_seed42=normalize_map(raw42),
            baseline_seed43=normalize_map(raw43),
        )

        if ds_idx > 0 and ds_idx % 200 == 0:
            print(f"  {ds_idx}/{n_total} images processed …")

    n_arrays = len(list(OUT_ARRAYS_SEED43.glob("*.npz")))
    print(f"\nDone.  Arrays: {n_arrays:,}  ->  {OUT_ARRAYS_SEED43}")


if __name__ == "__main__":
    main()
