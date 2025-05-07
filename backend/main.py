import time
import base64
import io
import numpy as np
from PIL import Image, ImageOps
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import cv2
import pytesseract
from segment_anything import sam_model_registry, SamPredictor
from utils import save_image, encode_image_to_base64, visualize_mask_with_box, handle_exception

# Manually specify tesseract path (update this path if yours is different)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b.pth")
predictor = SamPredictor(sam)
sam.to(device)

app = FastAPI()

class SegmentRequest(BaseModel):
    image: str
    box: list  # [x1, y1, x2, y2]

def detect_segment_and_crop(image: np.ndarray, box: list) -> dict:
    predictor.set_image(image)
    input_box = np.array([box])
    start = time.time()
    masks, scores, _ = predictor.predict(box=input_box)
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
        "box": input_box
    }

def combine_detected_segments_from_array(img: np.ndarray, display_steps=True):
    """
    OCR pipeline that receives an image (as a NumPy array), detects text segments,
    annotates them, and prints performance metrics.

    Args:
        img: Input image as a NumPy array (e.g., cropped from SAM)
        display_steps: If True, shows intermediate steps using OpenCV

    Returns:
        result: Annotated image with text boxes and labels
        detected_word_positions: List of (text, (x, y, w, h)) for each word
        performance: Dictionary of performance metrics
    """
    def display_cv_image(image, title="Image"):
        cv2.imshow(title, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    start_time = time.time()
    result = img.copy()

    if display_steps:
        print("Original Image:")
        display_cv_image(img)

    # Preprocess
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.0, beta=0.05)
    gray_resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    if display_steps:
        print("Resized Grayscale Image:")
        display_cv_image(gray_resized)

    binary = cv2.adaptiveThreshold(
        gray_resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((1, 1), np.uint8))

    if display_steps:
        print("Binary Image:")
        display_cv_image(closed)

    inference_start = time.time()
    config = r'--oem 3 --psm 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ: '
    ocr_data = pytesseract.image_to_data(closed, output_type=pytesseract.Output.DICT, config=config)
    inference_end = time.time()
    inference_time = inference_end - inference_start

    text_conf = [ocr_data['text'][i] for i in range(len(ocr_data['text'])) if int(ocr_data['conf'][i]) >= 40]
    full_text = " ".join(text_conf)
    segments = full_text.split()

    print(f"\n🧠 OCR Detected Full Text: '{full_text}'")
    print(f"OCR Segments: {segments}")
    print(f"OCR Inference Time: {inference_time:.4f}s")

    result_resized = cv2.resize(result, None, fx=1, fy=1)
    detected_word_positions = []

    for i in range(len(ocr_data['text'])):
        word = ocr_data['text'][i].strip()
        if word and int(ocr_data['conf'][i]) >= 40:
            x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
            cv2.rectangle(result_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(result_resized, word, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            detected_word_positions.append((word, (x, y, w, h)))
            print(f"🟩 Word '{word}' found at (x={x}, y={y}, w={w}, h={h})")

    if display_steps:
        print("Final Annotated OCR Image:")
        display_cv_image(result_resized)

    end_time = time.time()
    total_time = end_time - start_time

    performance = {
        'total_time': total_time,
        'inference_time': inference_time,
        'total_segments': len(detected_word_positions)
    }

    print(f"\nPerformance Summary:")
    print(f"Total Time: {total_time:.4f}s")
    print(f"Text Segments Found: {len(detected_word_positions)}")

    # Save the image with annotations
    cv2.imwrite("ocr_with_boxes.jpg", result_resized)

    return result_resized, detected_word_positions, performance


@app.post("/segment")
async def segment(req: SegmentRequest):
    try:
        if "," in req.image:
            _, encoded = req.image.split(",", 1)
        else:
            encoded = req.image
        image_data = base64.b64decode(encoded)
        pil_image = ImageOps.exif_transpose(Image.open(io.BytesIO(image_data))).convert("RGB")
        image = np.array(pil_image)

        print("Received image size:", image.shape[:2])
        print("Frontend box:", req.box)

        result = detect_segment_and_crop(image, req.box)

        save_image(result["cropped_image"], "output.jpg")
        vis_b64 = visualize_mask_with_box(result["mask"], image, result["box"], save_path="mask.jpg")
        crop_b64 = encode_image_to_base64(result["cropped_image"])

        # 🔍 OCR Detection
        ocr_result_img, ocr_boxes, ocr_metrics = combine_detected_segments_from_array(result["cropped_image"], display_steps=False)
        cv2.imwrite("ocr_with_boxes.jpg", ocr_result_img)

        print("\n🧠 OCR Text Segments:")
        for word, (x, y, w, h) in ocr_boxes:
            print(f" - '{word}' at x={x}, y={y}, w={w}, h={h}")

        return {
            "status": "success",
            "inference_time": result["inference_time"],
            "cropped_base64": crop_b64,
            "mask_visualization_base64": vis_b64
        }

    except Exception as e:
        return handle_exception(e, "Segmentation Error")
