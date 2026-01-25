import os
import cv2
import albumentations as A
import numpy as np

# -------------------------
# Config
# -------------------------
INPUT_IMAGE = "tester.png"
OUTPUT_DIR = os.path.join("train", "good")   # ✅ changed
NUM_IMAGES = 50

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Albumentations pipeline
# -------------------------
augment = A.Compose(
    [
        A.OneOf(
            [
                A.Rotate(
                    limit=2,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    mask_value=0,
                    p=1.0,
                ),
                A.ElasticTransform(
                    alpha=1.0,
                    sigma=25,
                    alpha_affine=1,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    mask_value=0,
                    p=1.0,
                ),
                A.Compose(
                    [
                        A.Rotate(
                            limit=3,
                            border_mode=cv2.BORDER_CONSTANT,
                            value=0,
                            mask_value=0,
                            p=1.0,
                        ),
                        A.ElasticTransform(
                            alpha=1.0,
                            sigma=25,
                            alpha_affine=1,
                            border_mode=cv2.BORDER_CONSTANT,
                            value=0,
                            mask_value=0,
                            p=1.0,
                        ),
                    ]
                ),
            ],
            p=1.0,
        )
    ]
)

# -------------------------
# Load image WITH alpha
# -------------------------
img = cv2.imread(INPUT_IMAGE, cv2.IMREAD_UNCHANGED)
if img is None:
    raise FileNotFoundError(f"Could not load {INPUT_IMAGE}")

if img.shape[2] != 4:
    raise ValueError("Image does not contain an alpha channel")

# Split RGB and alpha
rgb = img[:, :, :3]
alpha = img[:, :, 3]

# -------------------------
# Generate augmentations
# -------------------------
for i in range(NUM_IMAGES):
    augmented = augment(image=rgb, mask=alpha)

    aug_rgb = augmented["image"]
    aug_alpha = augmented["mask"]

    # Recombine RGBA
    rgba = np.dstack([aug_rgb, aug_alpha])

    out_path = os.path.join(OUTPUT_DIR, f"aug_{i:03d}.png")
    cv2.imwrite(out_path, rgba)

print(f"✅ Generated {NUM_IMAGES} transparent PNGs in '{OUTPUT_DIR}'")
