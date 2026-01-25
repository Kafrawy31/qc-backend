
# def run_ocr_in_box(image: np.ndarray, box: list):
#     """
#     Performs CRAFT + TrOCR OCR only inside the given bounding box.
#     Saves output image (ocr_output.jpg) and text (ocr_output.txt).
#     """
#     init_ocr_models()

#     x1, y1, x2, y2 = map(int, box)
#     cropped = image[y1:y2, x1:x2]
#     original_crop = cropped.copy()

#     CRAFT_MAX_SIZE = 2048  # try 2048 first; 2560 if you have VRAM
#     img_resized, target_ratio, _ = resize_aspect_ratio(cropped, CRAFT_MAX_SIZE, interpolation=cv2.INTER_LINEAR)
#     ratio_h = ratio_w = 1 / target_ratio
#     x = normalizeMeanVariance(img_resized)
#     x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float().to(device)

#     with torch.no_grad():
#         y, _ = craft_model(x)

#     score_text = y[0, :, :, 0].cpu().data.numpy()
#     score_link = y[0, :, :, 1].cpu().data.numpy()

#     boxes, _ = getDetBoxes(score_text, score_link, text_threshold=0.7, link_threshold=0.4, low_text=0.4, poly=False)
#     boxes = np.array(boxes) * (2 / target_ratio)

#     # Merge boxes by line
#     merged_boxes = []
#     used = [False] * len(boxes)
#     for i, b1 in enumerate(boxes):
#         if used[i]: continue
#         group = [b1]; used[i] = True
#         for j in range(i + 1, len(boxes)):
#             if used[j]: continue
#             b2 = boxes[j]
#             if abs(np.mean(b1[:,1]) - np.mean(b2[:,1])) < 15:  # same line
#                 group.append(b2); used[j] = True
#         all_x = np.concatenate([b[:, 0] for b in group])
#         all_y = np.concatenate([b[:, 1] for b in group])
#         merged = np.array([
#             [np.min(all_x), np.min(all_y)],
#             [np.max(all_x), np.min(all_y)],
#             [np.max(all_x), np.max(all_y)],
#             [np.min(all_x), np.max(all_y)]
#         ])
#         merged_boxes.append(merged)

#     # Run TrOCR
#     results = []
#     for idx, box in enumerate(merged_boxes):
#         box = np.int32(box)
#         cx1 = max(int(np.min(box[:, 0])), 0)
#         cy1 = max(int(np.min(box[:, 1])), 0)
#         cx2 = min(int(np.max(box[:, 0])), cropped.shape[1])
#         cy2 = min(int(np.max(box[:, 1])), cropped.shape[0])

#         region = cropped[cy1:cy2, cx1:cx2]
#         if region.size == 0: continue
#         pil_image = Image.fromarray(cv2.cvtColor(region, cv2.COLOR_BGR2RGB)).convert("RGB")
#         pixel_values = trocr_processor(images=pil_image, return_tensors="pt").pixel_values.to(device)

#         with torch.no_grad():
#             generated_ids = trocr_model.generate(pixel_values)
#             text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
#             results.append((box, text.strip()))

#     # Draw and save
#     for box, text in results:
#         box = np.int32(box)
#         cv2.polylines(cropped, [box.reshape((-1, 1, 2))], True, (0, 255, 0), 2)
#         cv2.putText(cropped, text, (int(box[0][0]), int(box[0][1]) - 5),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

#     cv2.imwrite("ocr_output.jpg", cropped)
#     with open("ocr_output.txt", "w", encoding="utf-8") as f:
#         for _, text in results:
#             f.write(text + "\n")

#     print("✅ Saved OCR outputs: ocr_output.jpg and ocr_output.txt")



import time
import base64
import io
import numpy as np
from PIL import Image, ImageOps
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
import torch
import cv2
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO
from utils import save_image, encode_image_to_base64, visualize_mask_with_box, handle_exception
from craft import CRAFT
from craft_utils import getDetBoxes
from imgproc import resize_aspect_ratio, normalizeMeanVariance
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from collections import OrderedDict
from datetime import datetime
import json
from pathlib import Path


craft_model = None
trocr_processor = None
trocr_model = None

# Load SAM model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b.pth")
predictor = SamPredictor(sam)
sam.to(device)

# Load YOLO barcode model
barcode_model = YOLO("./best.pt")
barcode_model.to(device)


# --- helpers ---
def b64_png_from_rgb_np(rgb_np: np.ndarray) -> str:
    """Encode an RGB uint8 numpy image to PNG base64."""
    if rgb_np.dtype != np.uint8:
        rgb_np = np.clip(rgb_np, 0, 255).astype(np.uint8)
    pil = Image.fromarray(rgb_np, mode="RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=False)  # lossless
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)
    
    
app = FastAPI()


class SegmentRequest(BaseModel):
    image: str
    box: list  # [x1, y1, x2, y2]

def reset_log(log_file="log.txt"):
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("")  # overwrite

def log_line(message: str, log_file="log.txt"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {message}\n")

class StepTimer:
    def __init__(self, log_file="log.txt"):
        self.log_file = log_file
        self.t0 = time.perf_counter()
        self.last = self.t0

    def mark(self, label: str):
        now = time.perf_counter()
        step_ms = (now - self.last) * 1000
        total_ms = (now - self.t0) * 1000
        log_line(f"{label}: {step_ms:.1f} ms (total {total_ms:.1f} ms)", self.log_file)
        self.last = now

def detect_segment_and_crop(image: np.ndarray, box: list) -> dict:
    predictor.set_image(image)
    input_box = np.array([box])
    
    start = time.time()
    # SAM prediction uses the full resolution image
    masks, scores, _ = predictor.predict(box=input_box, multimask_output=False)
    elapsed = time.time() - start

    mask = masks[0]
    
    # --- NEW DYNAMIC CROP LOGIC ---
    ys, xs = np.where(mask)
    
    if len(xs) == 0 or len(ys) == 0:
        # Fallback: if mask is empty, crop using the input guidance box
        print("Empty mask detected, falling back to input box")
        x1, y1, x2, y2 = map(int, box[0])
    else:
        # 1. Find the bounding box of the detected object (mask)
        x1, x2 = np.min(xs), np.max(xs)
        y1, y2 = np.min(ys), np.max(ys)

        # 2. Add some padding (e.g., 50px or 10%) so the object isn't touching the edges
        h, w = image.shape[:2]
        padding = 50 
        
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

    # 3. Perform the crop
    cropped = image[y1:y2, x1:x2]

    return {
        "cropped_image": cropped,
        "mask": mask,
        "inference_time": elapsed,
        "box": input_box,
        "full_image": image
    }

def extract_object_as_png(image: np.ndarray, mask: np.ndarray, output_path="object_extracted.png"):
    mask = (mask * 255).astype(np.uint8)
    image_rgba = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    image_rgba[:, :, 3] = mask
    cv2.imwrite(output_path, image_rgba)

def detect_and_draw_barcodes(image: np.ndarray, save_path="barcodes.jpg", conf_threshold=0.25):
    results = barcode_model(image, conf=conf_threshold)
    boxes = results[0].boxes

    image_with_boxes = image.copy()

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = box.conf[0].item()
        label = f"Barcode: {conf:.2f}"
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(image_with_boxes, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imwrite(save_path, cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR))
    print(f"✅ Barcode image saved as {save_path}")
    return save_path

def init_ocr_models():
    global craft_model, trocr_processor, trocr_model
    if craft_model is None:
        craft_model = CRAFT()
        state_dict = torch.load("craft_mlt_25k.pth", map_location=device)
        if list(state_dict.keys())[0].startswith("module"):
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v
            state_dict = new_state_dict
        craft_model.load_state_dict(state_dict)
        craft_model = craft_model.to(device).eval()

    if trocr_model is None or trocr_processor is None:
        trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed", use_fast=True)
        trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
        trocr_model = trocr_model.to(device).eval()

CRAFT_MAX_SIZE = 2048  # 2048 is a good start for iPhone full-res crops
def resize_for_trocr(pil_img: Image.Image, target_h=384, max_w=1024) -> Image.Image:
    w, h = pil_img.size
    if h <= 0:
        return pil_img
    new_w = int(w * (target_h / h))
    new_w = max(1, min(new_w, max_w))
    return pil_img.resize((new_w, target_h), Image.BICUBIC)

def run_ocr_in_box(image: np.ndarray, box: list):
    init_ocr_models()

    x1, y1, x2, y2 = map(int, box)
    cropped = image[y1:y2, x1:x2]

    # Resize + normalize for CRAFT (higher cap for more detail)
    img_resized, target_ratio, _ = resize_aspect_ratio(
        cropped, CRAFT_MAX_SIZE, interpolation=cv2.INTER_LINEAR
    )

    x = normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        y, _ = craft_model(x)

    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    boxes, _ = getDetBoxes(
        score_text, score_link,
        text_threshold=0.7, link_threshold=0.3, low_text=0.4,
        poly=False
    )

    # map boxes back to 'cropped' coordinates
    boxes = np.array(boxes) * (2 / target_ratio)

    # Merge boxes by line (your logic unchanged)
    merged_boxes = []
    used = [False] * len(boxes)
    for i, b1 in enumerate(boxes):
        if used[i]:
            continue
        group = [b1]
        used[i] = True
        for j in range(i + 1, len(boxes)):
            if used[j]:
                continue
            b2 = boxes[j]
            if abs(np.mean(b1[:, 1]) - np.mean(b2[:, 1])) < 15:
                group.append(b2)
                used[j] = True

        all_x = np.concatenate([b[:, 0] for b in group])
        all_y = np.concatenate([b[:, 1] for b in group])

        merged = np.array([
            [np.min(all_x), np.min(all_y)],
            [np.max(all_x), np.min(all_y)],
            [np.max(all_x), np.max(all_y)],
            [np.min(all_x), np.max(all_y)]
        ])
        merged_boxes.append(merged)

    # Run TrOCR with aspect-ratio resizing (stable, high fidelity)
    results = []
    for b in merged_boxes:
        b = np.int32(b)
        cx1 = max(int(np.min(b[:, 0])), 0)
        cy1 = max(int(np.min(b[:, 1])), 0)
        cx2 = min(int(np.max(b[:, 0])), cropped.shape[1])
        cy2 = min(int(np.max(b[:, 1])), cropped.shape[0])

        region = cropped[cy1:cy2, cx1:cx2]
        if region.size == 0:
            continue

        pil_image = Image.fromarray(cv2.cvtColor(region, cv2.COLOR_BGR2RGB)).convert("RGB")
        pil_image = resize_for_trocr(pil_image, target_h=384, max_w=1024)

        pixel_values = trocr_processor(images=pil_image, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
            generated_ids = trocr_model.generate(pixel_values)
            text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            results.append((b, text.strip()))

    # Draw/save (same as you already had)
    for b, text in results:
        b = np.int32(b)
        cv2.polylines(cropped, [b.reshape((-1, 1, 2))], True, (0, 255, 0), 2)
        cv2.putText(
            cropped, text,
            (int(b[0][0]), int(b[0][1]) - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2
        )

    cv2.imwrite("ocr_output.jpg", cropped)
    with open("ocr_output.txt", "w", encoding="utf-8") as f:
        for _, text in results:
            f.write(text + "\n")

def log_time(message: str, log_file="log.txt"):
    with open(log_file, "a") as f:
        f.write(message + "\n")

@app.post("/segment")
async def segment(
    file: UploadFile = File(...),
    box: str = Form(...),
    dimensions: str = Form(None),
):
    try:
        reset_log("log.txt")
        timer = StepTimer("log.txt")

        box_list = json.loads(box)
        dims = json.loads(dimensions) if dimensions else None
        timer.mark("Parsed frontend fields")

        image_bytes = await file.read()
        timer.mark(f"Read upload bytes ({len(image_bytes)} bytes)")

        pil = Image.open(io.BytesIO(image_bytes))
        log_line(f"Raw PIL size: {pil.size}", "log.txt")

        pil_image = ImageOps.exif_transpose(pil).convert("RGB")
        image = np.array(pil_image)
        timer.mark(f"Decoded image to numpy {image.shape[:2]}")

        log_line(f"Frontend box: {box_list}", "log.txt")
        if dims:
            log_line(f"Frontend dimensions: {dims}", "log.txt")

        result = detect_segment_and_crop(image, box_list)
        timer.mark("SAM segment + crop")

        save_image(result["cropped_image"], "output.jpg")
        timer.mark("Saved output.jpg")

        vis_b64 = visualize_mask_with_box(result["mask"], image, result["box"], save_path="mask.jpg")
        timer.mark("Saved mask.jpg + encoded visualization")

        crop_b64 = encode_image_to_base64(result["cropped_image"])
        timer.mark("Encoded cropped image to base64")

        extract_object_as_png(result["full_image"], result["mask"], output_path="object_extracted.png")
        timer.mark("Saved object_extracted.png")

        detect_and_draw_barcodes(image, save_path="barcodes.jpg")
        timer.mark("YOLO barcode detect + saved barcodes.jpg")

        run_ocr_in_box(image, box_list)
        timer.mark("CRAFT + TrOCR OCR")

        return {
            "status": "success",
            "inference_time": result["inference_time"],
            "cropped_base64": crop_b64,
            "mask_visualization_base64": vis_b64,
        }

    except Exception as e:
        return handle_exception(e, "Segmentation Error")
