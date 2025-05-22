import time
import base64
import io
import numpy as np
from PIL import Image, ImageOps
from fastapi import FastAPI
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

app = FastAPI()

class SegmentRequest(BaseModel):
    image: str
    box: list  # [x1, y1, x2, y2]

def detect_segment_and_crop(image: np.ndarray, box: list) -> dict:
    predictor.set_image(image)
    input_box = np.array([box])
    start = time.time()
    masks, scores, _ = predictor.predict(box=input_box, multimask_output=False)
    elapsed = time.time() - start

    mask = masks[0]
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("No object detected in mask.")

    # Center crop
    cx, cy = (np.min(xs) + np.max(xs)) // 2, (np.min(ys) + np.max(ys)) // 2
    crop_size = 512
    half = crop_size // 2
    h, w = image.shape[:2]
    crop_x1 = max(0, min(w, cx - half))
    crop_y1 = max(0, min(h, cy - half))
    crop_x2 = min(w, crop_x1 + crop_size)
    crop_y2 = min(h, crop_y1 + crop_size)
    crop_x1 = max(0, crop_x2 - crop_size)
    crop_y1 = max(0, crop_y2 - crop_size)
    cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]

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

def run_ocr_in_box(image: np.ndarray, box: list):
    """
    Performs CRAFT + TrOCR OCR only inside the given bounding box.
    Saves output image (ocr_output.jpg) and text (ocr_output.txt).
    """
    init_ocr_models()

    x1, y1, x2, y2 = map(int, box)
    cropped = image[y1:y2, x1:x2]
    original_crop = cropped.copy()

    # Resize and normalize
    img_resized, target_ratio, _ = resize_aspect_ratio(cropped, 1280, interpolation=cv2.INTER_LINEAR)
    ratio_h = ratio_w = 1 / target_ratio
    x = normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        y, _ = craft_model(x)

    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    boxes, _ = getDetBoxes(score_text, score_link, text_threshold=0.7, link_threshold=0.4, low_text=0.4, poly=False)
    boxes = np.array(boxes) * (2 / target_ratio)

    # Merge boxes by line
    merged_boxes = []
    used = [False] * len(boxes)
    for i, b1 in enumerate(boxes):
        if used[i]: continue
        group = [b1]; used[i] = True
        for j in range(i + 1, len(boxes)):
            if used[j]: continue
            b2 = boxes[j]
            if abs(np.mean(b1[:,1]) - np.mean(b2[:,1])) < 15:  # same line
                group.append(b2); used[j] = True
        all_x = np.concatenate([b[:, 0] for b in group])
        all_y = np.concatenate([b[:, 1] for b in group])
        merged = np.array([
            [np.min(all_x), np.min(all_y)],
            [np.max(all_x), np.min(all_y)],
            [np.max(all_x), np.max(all_y)],
            [np.min(all_x), np.max(all_y)]
        ])
        merged_boxes.append(merged)

    # Run TrOCR
    results = []
    for idx, box in enumerate(merged_boxes):
        box = np.int32(box)
        cx1 = max(int(np.min(box[:, 0])), 0)
        cy1 = max(int(np.min(box[:, 1])), 0)
        cx2 = min(int(np.max(box[:, 0])), cropped.shape[1])
        cy2 = min(int(np.max(box[:, 1])), cropped.shape[0])

        region = cropped[cy1:cy2, cx1:cx2]
        if region.size == 0: continue
        pil_image = Image.fromarray(cv2.cvtColor(region, cv2.COLOR_BGR2RGB)).convert("RGB").resize((384, 384))
        pixel_values = trocr_processor(images=pil_image, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
            generated_ids = trocr_model.generate(pixel_values)
            text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            results.append((box, text.strip()))

    # Draw and save
    for box, text in results:
        box = np.int32(box)
        cv2.polylines(cropped, [box.reshape((-1, 1, 2))], True, (0, 255, 0), 2)
        cv2.putText(cropped, text, (int(box[0][0]), int(box[0][1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imwrite("ocr_output.jpg", cropped)
    with open("ocr_output.txt", "w", encoding="utf-8") as f:
        for _, text in results:
            f.write(text + "\n")

    print("✅ Saved OCR outputs: ocr_output.jpg and ocr_output.txt")

def log_time(message: str, log_file="log.txt"):
    with open(log_file, "a") as f:
        f.write(message + "\n")

@app.post("/segment")
async def segment(req: SegmentRequest):
    try:
        start_total = time.time()

        # Decode base64 image
        start_decode = time.time()
        if "," in req.image:
            _, encoded = req.image.split(",", 1)
        else:
            encoded = req.image
        image_data = base64.b64decode(encoded)
        pil_image = ImageOps.exif_transpose(Image.open(io.BytesIO(image_data))).convert("RGB")
        image = np.array(pil_image)
        end_decode = time.time()
        log_time(f"[TIMING] Image decoding: {end_decode - start_decode:.3f} seconds")

        log_time(f"Received image size: {image.shape[:2]}")
        log_time(f"Frontend box: {req.box}")

        # SAM segmentation and cropping
        start_sam = time.time()
        result = detect_segment_and_crop(image, req.box)
        end_sam = time.time()
        log_time(f"[TIMING] SAM segmentation + crop: {end_sam - start_sam:.3f} seconds")

        save_image(result["cropped_image"], "output.jpg")

        start_vis = time.time()
        vis_b64 = visualize_mask_with_box(result["mask"], image, result["box"], save_path="mask.jpg")
        crop_b64 = encode_image_to_base64(result["cropped_image"])
        end_vis = time.time()
        log_time(f"[TIMING] Mask visualization & encoding: {end_vis - start_vis:.3f} seconds")

        start_extract = time.time()
        extract_object_as_png(result["full_image"], result["mask"], output_path="object_extracted.png")
        end_extract = time.time()
        log_time(f"[TIMING] Extract transparent object: {end_extract - start_extract:.3f} seconds")

        # Barcode detection (draws on full image, saves as barcodes.jpg)
        start_barcode = time.time()
        detect_and_draw_barcodes(image, save_path="barcodes.jpg")
        end_barcode = time.time()
        log_time(f"[TIMING] Barcode detection & save: {end_barcode - start_barcode:.3f} seconds")

        # OCR inside box (CRAFT + TrOCR)
        start_ocr = time.time()
        run_ocr_in_box(image, req.box)
        end_ocr = time.time()
        log_time(f"[TIMING] OCR in box (CRAFT + TrOCR): {end_ocr - start_ocr:.3f} seconds")

        end_total = time.time()
        log_time(f"[TIMING] Total pipeline time: {end_total - start_total:.3f} seconds")

        return {
            "status": "success",
            "inference_time": result["inference_time"],
            "cropped_base64": crop_b64,
            "mask_visualization_base64": vis_b64,
            "message": "Transparent object saved as object_extracted.png, barcodes saved as barcodes.jpg"
        }

    except Exception as e:
        return handle_exception(e, "Segmentation Error")
