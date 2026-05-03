"""
Grad-CAM generation for all ~3,925 ImageNette validation images — ResNet-18 student.

Loads teacher (ResNet-50), KD student, and baseline student checkpoints.
For each image, computes a normalized Grad-CAM map for all three models and
saves them to a single .npz file. Saves 10 sample overlay figures.

Outputs:
  students/resnet18/results/gradcam_full/arrays/   — one .npz per image
  students/resnet18/results/gradcam_full/figures/  — 10 sample PNG overlays
"""
import tarfile
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as tv_datasets
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image as PILImage
from pytorch_grad_cam import GradCAM
from tqdm import tqdm

_HERE = Path(__file__).parent       # students/resnet18/
_ROOT = _HERE.parent.parent         # project root

SEED        = 42
NUM_CLASSES = 10
IMG_SIZE    = 224

TEACHER_CKPT  = _ROOT / "teacher" / "checkpoints" / "teacher_finetuned.pth"
KD_CKPT       = _HERE / "checkpoints" / "resnet18_kd.pth"
BASELINE_CKPT = _HERE / "checkpoints" / "resnet18_baseline.pth"

DATA_DIR       = _ROOT / "data"
IMAGENETTE_DIR = DATA_DIR / "imagenette2-320"
IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"

OUT_FIGS   = _HERE / "results" / "gradcam_full" / "figures"
OUT_ARRAYS = _HERE / "results" / "gradcam_full" / "arrays"

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

try:
    _BILINEAR = PILImage.Resampling.BILINEAR
except AttributeError:
    _BILINEAR = PILImage.BILINEAR

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


def load_resnet50(path):
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.eval().to(DEVICE)


def load_resnet18(path):
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.eval().to(DEVICE)


def compute_raw_gradcam(cam_obj, model, input_tensor):
    with torch.no_grad():
        logits = model(input_tensor)
    pred_idx = int(logits.argmax(1).item())
    cam_obj(input_tensor=input_tensor, targets=None)
    acts  = cam_obj.activations_and_grads.activations[0].cpu().numpy()
    grads = cam_obj.activations_and_grads.gradients[0].cpu().numpy()
    weights = np.mean(grads, axis=(2, 3), keepdims=True)
    raw = np.maximum(np.sum(weights * acts, axis=1)[0], 0.0)
    return raw, pred_idx


def normalize_map(raw):
    total = raw.sum()
    return raw / total if total > 0 else np.ones_like(raw) / raw.size


def denormalize(tensor):
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    return np.clip(img * _STD + _MEAN, 0.0, 1.0).astype(np.float32)


def make_overlay(img, raw_map, alpha=0.5):
    H, W = img.shape[:2]
    u8 = (raw_map / (raw_map.max() + 1e-8) * 255).astype(np.uint8)
    resized = np.array(PILImage.fromarray(u8).resize((W, H), _BILINEAR)) / 255.0
    heat = plt.get_cmap("jet")(resized)[:, :, :3]
    return np.clip(alpha * heat + (1 - alpha) * img, 0.0, 1.0).astype(np.float32)


def save_figure(img, raw_maps, pred_names, true_class, stem):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img)
    axes[0].set_title(f"Original\n(true: {true_class})", fontsize=9)
    axes[0].axis("off")
    for ax, (key, label) in zip(axes[1:], [
        ("teacher",    "Teacher (ResNet-50)"),
        ("kd_student", "KD Student (ResNet-18)"),
        ("baseline",   "Baseline (ResNet-18)"),
    ]):
        ax.imshow(make_overlay(img, raw_maps[key]))
        ax.set_title(f"{label}\npred: {pred_names[key]}", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(OUT_FIGS / f"{stem}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    print(f"Device  : {DEVICE}")
    print(f"Teacher : {TEACHER_CKPT}")
    print(f"KD      : {KD_CKPT}")
    print(f"Baseline: {BASELINE_CKPT}\n")

    download_imagenette()
    OUT_FIGS.mkdir(parents=True, exist_ok=True)
    OUT_ARRAYS.mkdir(parents=True, exist_ok=True)

    dataset     = tv_datasets.ImageFolder(IMAGENETTE_DIR / "val", transform=transform)
    class_names = [IMAGENETTE_LABELS.get(c, c) for c in dataset.classes]
    print(f"Test images: {len(dataset):,}  |  Classes: {class_names}\n")

    print("Loading models …")
    teacher    = load_resnet50(TEACHER_CKPT)
    kd_student = load_resnet18(KD_CKPT)
    baseline   = load_resnet18(BASELINE_CKPT)

    cam_teacher  = GradCAM(model=teacher,    target_layers=[teacher.layer4[-1]])
    cam_kd       = GradCAM(model=kd_student, target_layers=[kd_student.layer4[-1]])
    cam_baseline = GradCAM(model=baseline,   target_layers=[baseline.layer4[-1]])
    model_cams   = [
        ("teacher",    teacher,    cam_teacher),
        ("kd_student", kd_student, cam_kd),
        ("baseline",   baseline,   cam_baseline),
    ]

    rng = np.random.default_rng(SEED)
    figure_indices = set(rng.choice(len(dataset), size=10, replace=False).tolist())

    class_counter = {}
    for ds_idx in tqdm(range(len(dataset)), desc="Grad-CAMs"):
        img_tensor, true_label = dataset[ds_idx]
        input_tensor = img_tensor.unsqueeze(0).to(DEVICE)
        true_class   = class_names[true_label]
        class_counter[true_class] = class_counter.get(true_class, 0) + 1
        stem = f"{true_class}_{class_counter[true_class]:04d}"

        img_hw3      = denormalize(img_tensor)
        raw_maps     = {}
        pred_names   = {}
        pred_indices = {}

        for key, model, cam_obj in model_cams:
            raw, pred_idx     = compute_raw_gradcam(cam_obj, model, input_tensor)
            raw_maps[key]     = raw
            pred_indices[key] = pred_idx
            pred_names[key]   = class_names[pred_idx]

        norm_maps = {k: normalize_map(v) for k, v in raw_maps.items()}
        np.savez(
            OUT_ARRAYS / f"{stem}.npz",
            teacher=norm_maps["teacher"],
            kd_student=norm_maps["kd_student"],
            baseline=norm_maps["baseline"],
            true_label=np.int64(true_label),
            teacher_pred=np.int64(pred_indices["teacher"]),
            kd_pred=np.int64(pred_indices["kd_student"]),
            baseline_pred=np.int64(pred_indices["baseline"]),
        )
        if ds_idx in figure_indices:
            save_figure(img_hw3, raw_maps, pred_names, true_class, stem)

    n_figs   = len(list(OUT_FIGS.glob("*.png")))
    n_arrays = len(list(OUT_ARRAYS.glob("*.npz")))
    print(f"\nDone.  Figures: {n_figs:,}  |  Arrays: {n_arrays:,}")
    print(f"Arrays  -> {OUT_ARRAYS}")
    print(f"Figures -> {OUT_FIGS}")


if __name__ == "__main__":
    main()
