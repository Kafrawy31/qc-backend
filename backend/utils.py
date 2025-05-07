# utils.py
import io
import base64
import cv2
import numpy as np
import matplotlib.pyplot as plt

def save_image(image: np.ndarray, filename: str):
    cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def encode_image_to_base64(image: np.ndarray) -> str:
    _, buffer = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode("utf-8")

def visualize_mask_with_box(mask, image, box=None, alpha=0.6, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, origin="upper")
    if box is not None:
        x1, y1, x2, y2 = box[0]
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                             fill=False, edgecolor="red", linewidth=2)
        ax.add_patch(rect)
    ax.imshow(mask, alpha=alpha, cmap="jet", origin="upper")
    ax.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format="jpeg", bbox_inches="tight")
    buf.seek(0)

    if save_path:
        with open(save_path, "wb") as f:
            f.write(buf.getvalue())

    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def handle_exception(error, label="Error"):
    print(f"{label} occurred:\n", error)
    return {
        "status": "error",
        "message": str(error),
        "type": label
    }
