import os
import shutil
import random
from pathlib import Path

import cv2
import numpy as np

# ---- hard-disable wandb / external logging ----
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
os.environ["ANOMALIB_LOGGER"] = "none"


# ----------------------------
# Config
# ----------------------------
AUG_DIR = Path("test_augments")               # your generated images
DATA_ROOT = Path("patchcore_data")
TRAIN_GOOD = DATA_ROOT / "train" / "good"
TEST_GOOD = DATA_ROOT / "test" / "good"      # we'll treat this as val for visualization

VAL_RATIO = 0.20
SEED = 42

IMAGE_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 5

HEATMAP_DIR = Path("heatmaps_val")           # outputs saved here
HEATMAP_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Helpers
# ----------------------------
def clear_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)
    for p in d.glob("*"):
        if p.is_file():
            p.unlink()


def ensure_rgb_png(src_path: Path, dst_path: Path):
    """Read image (possibly RGBA) and write as 3-channel BGR PNG."""
    img = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read image: {src_path}")

    # If RGBA (BGRA in OpenCV), drop alpha
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    # If grayscale, convert to BGR
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    ok = cv2.imwrite(str(dst_path), img)
    if not ok:
        raise RuntimeError(f"Failed to write image: {dst_path}")


def make_train_val_split():
    imgs = sorted([p for p in AUG_DIR.glob("*.png")])
    if not imgs:
        raise FileNotFoundError(f"No .png files found in: {AUG_DIR.resolve()}")

    # fresh folders
    clear_dir(TRAIN_GOOD)
    clear_dir(TEST_GOOD)

    random.seed(SEED)
    random.shuffle(imgs)

    n_val = max(1, int(len(imgs) * VAL_RATIO))
    val_imgs = imgs[:n_val]
    train_imgs = imgs[n_val:]

    # copy + convert to RGB
    for p in train_imgs:
        ensure_rgb_png(p, TRAIN_GOOD / p.name)

    for p in val_imgs:
        ensure_rgb_png(p, TEST_GOOD / p.name)

    print(f"✅ Split complete: train={len(train_imgs)}, val(test/good)={len(val_imgs)}")
    print("Train dir:", TRAIN_GOOD.resolve())
    print("Val dir:", TEST_GOOD.resolve())


def overlay_heatmap_on_image(image_bgr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    heatmap can be:
      - HxW float in [0,1] or [0,255]
      - HxW uint8
      - HxWx3
    returns BGR overlay image
    """
    h, w = image_bgr.shape[:2]

    hm = heatmap
    if hm.ndim == 3 and hm.shape[2] == 3:
        hm_color = cv2.resize(hm, (w, h))
        if hm_color.dtype != np.uint8:
            hm_color = np.clip(hm_color, 0, 255).astype(np.uint8)
    else:
        hm = cv2.resize(hm, (w, h))
        if hm.dtype != np.uint8:
            # assume 0..1 or 0..255 floats
            hm = np.clip(hm, 0, 1) if hm.max() <= 1.5 else np.clip(hm, 0, 255)
            hm = (hm * 255).astype(np.uint8) if hm.max() <= 1 else hm.astype(np.uint8)

        hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(image_bgr, 1.0, hm_color, alpha, 0)
    return overlay


def save_heatmaps_on_val(ckpt_path: Path):
    """
    Loads TorchInferencer and saves overlay heatmaps for each image in patchcore_data/test/good
    """
    from anomalib.deploy import TorchInferencer

    inferencer = TorchInferencer(path=str(ckpt_path), device="cuda")

    val_imgs = sorted(TEST_GOOD.glob("*.png"))
    if not val_imgs:
        raise RuntimeError(f"No validation images found in {TEST_GOOD.resolve()}")

    for p in val_imgs:
        out = inferencer.predict(image=str(p))

        # anomalib returns different object shapes depending on version:
        # Try common fields safely.
        # We want:
        # - original image (read ourselves)
        # - heatmap / anomaly map from inferencer output
        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            continue

        heatmap = None
        # Most common: out.anomaly_map (tensor/np) or out.heat_map / out.heatmap
        for key in ["anomaly_map", "anomaly_map_"]:
            if hasattr(out, key):
                heatmap = getattr(out, key)
                break
        if heatmap is None and hasattr(out, "heat_map"):
            heatmap = out.heat_map
        if heatmap is None and hasattr(out, "heatmap"):
            heatmap = out.heatmap

        if heatmap is None:
            # fallback: try dict-like
            if isinstance(out, dict):
                heatmap = out.get("anomaly_map") or out.get("heatmap") or out.get("heat_map")

        if heatmap is None:
            raise RuntimeError(
                "Could not find heatmap/anomaly_map in TorchInferencer output. "
                "Tell me your anomalib version or print(out) once and I’ll map the right field."
            )

        # convert torch -> numpy if needed
        try:
            import torch
            if isinstance(heatmap, torch.Tensor):
                heatmap = heatmap.detach().cpu().numpy()
        except Exception:
            pass

        # squeeze to HxW
        heatmap = np.array(heatmap)
        heatmap = heatmap.squeeze()

        overlay = overlay_heatmap_on_image(img_bgr, heatmap, alpha=0.45)

        out_path = HEATMAP_DIR / f"{p.stem}_overlay.png"
        cv2.imwrite(str(out_path), overlay)

    print(f"✅ Saved heatmap overlays to: {HEATMAP_DIR.resolve()}")


def train_patchcore_and_get_ckpt():
    """
    Trains Patchcore and returns the best/last checkpoint path depending on version.
    """
    from lightning.pytorch import Trainer as Engine
    from anomalib.data import Folder
    from anomalib.models import Patchcore

    datamodule = Folder(
        name="patchcore_augmented",
        root=DATA_ROOT,
        normal_dir="train/good",
        abnormal_dir="test",     # folder exists; we only have test/good but that's fine for running test loop
        image_size=IMAGE_SIZE,
        train_batch_size=BATCH_SIZE,
        eval_batch_size=BATCH_SIZE,
        num_workers=0,           # ✅ Windows-safe
        task="classification",
    )

    model = Patchcore(
        backbone="resnet18",
        layers=["layer2", "layer3"],
        num_neighbors=17,
    )

    engine = Engine(
        max_epochs=EPOCHS,
        accelerator="gpu",
        devices=1,
        logger=False,
        num_sanity_val_steps=0,  # ✅ avoids early vstack issues
        enable_checkpointing=True,
    )

    engine.fit(model=model, datamodule=datamodule)

    # Find a usable checkpoint/model export to use with TorchInferencer.
    # Anomalib commonly writes to results/...; easiest: search for latest *.pt in results
    results_dir = Path("results")
    if not results_dir.exists():
        raise RuntimeError("No results/ directory found after training. Check trainer output/logs.")

    candidates = list(results_dir.rglob("model.pt"))
    if not candidates:
        # sometimes anomalib exports under weights/torch/model.pt or similar
        candidates = list(results_dir.rglob("*.pt"))

    if not candidates:
        raise RuntimeError("Could not find any exported .pt model under results/. Training may not have exported.")

    # choose most recently modified
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    ckpt_path = candidates[0]
    print("✅ Using exported model:", ckpt_path.resolve())
    return ckpt_path


def main():
    make_train_val_split()
    ckpt = train_patchcore_and_get_ckpt()
    save_heatmaps_on_val(ckpt)


if __name__ == "__main__":
    main()
