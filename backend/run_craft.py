import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from collections import OrderedDict

from craft import CRAFT
from craft_utils import getDetBoxes
from imgproc import resize_aspect_ratio, normalizeMeanVariance


from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from datetime import datetime


from shapely.geometry import box as shapely_box
from shapely.ops import unary_union
import numpy as np

def merge_boxes(boxes, x_thresh=10, y_thresh=10):
    """
    Group nearby boxes (same line) into merged boxes.
    """
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

            # Horizontal and vertical distance between b1 and b2
            b1_x_center = np.mean(b1[:, 0])
            b2_x_center = np.mean(b2[:, 0])
            b1_y_center = np.mean(b1[:, 1])
            b2_y_center = np.mean(b2[:, 1])

            if abs(b1_y_center - b2_y_center) < y_thresh and abs(b1_x_center - b2_x_center) < x_thresh * 5:
                group.append(b2)
                used[j] = True

        # Merge grouped boxes into a single enclosing rectangle
        all_x = np.concatenate([b[:, 0] for b in group])
        all_y = np.concatenate([b[:, 1] for b in group])
        merged = np.array([
            [np.min(all_x), np.min(all_y)],
            [np.max(all_x), np.min(all_y)],
            [np.max(all_x), np.max(all_y)],
            [np.min(all_x), np.max(all_y)],
        ])
        merged_boxes.append(merged)

    return merged_boxes


# Load TrOCR model + processor
def load_trocr_model():
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed",use_fast=True)
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    model = model.cuda().eval() if torch.cuda.is_available() else model.eval()
    return processor, model

# Perform OCR on each box
def ocr_with_trocr(image, boxes, processor, model):
    results = []
    for idx, box in enumerate(boxes):
        box = np.int32(box)
        x_min = np.min(box[:, 0])
        y_min = np.min(box[:, 1])
        x_max = np.max(box[:, 0])
        y_max = np.max(box[:, 1])

        # Crop and convert to PIL
        cropped = image[int(y_min):int(y_max), int(x_min):int(x_max)]
        if cropped.size == 0:
            continue
        pil_image = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)).convert("RGB")
        pil_image  = pil_image.resize((384, 384), Image.BILINEAR)
        np_img = np.array(pil_image)

        # Grayscale and back to BGR
        np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
        np_img = cv2.cvtColor(np_img, cv2.COLOR_GRAY2BGR)

        # Convert back to PIL if needed by TrOCR
        pil_image = Image.fromarray(np_img) 

        # Preprocess for TrOCR
        pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.cuda() if torch.cuda.is_available() else pixel_values

        # Predict
        with torch.no_grad():
            generated_ids = model.generate(pixel_values)
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print(f"[{idx}] {text.strip()}")
        results.append((box, text.strip()))

    return results

# Draw text annotations
def draw_ocr_results(image, results):
    for box, text in results:
        box = np.int32(box)
        x, y = box[0][0], box[0][1] - 10  # Top-left corner
        cv2.putText(image, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    return image


# Load the model with GPU support if available
def load_craft_model(weight_path='weights/craft_mlt_25k.pth'):
    net = CRAFT()
    state_dict = torch.load(weight_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    
    if list(state_dict.keys())[0].startswith("module"):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        state_dict = new_state_dict

    net.load_state_dict(state_dict)
    net = net.cuda().eval() if torch.cuda.is_available() else net.eval()
    return net

# Detect text
def detect_text(net, image_path):
    image = cv2.imread(image_path)
    img_resized, target_ratio, _ = resize_aspect_ratio(image, 1280, interpolation=cv2.INTER_LINEAR)
    print(img_resized.shape)
    print(target_ratio)
    ratio_h = ratio_w = 1 / target_ratio
    print(ratio_h, ratio_w)

    x = normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float()
    x = x.cuda() if torch.cuda.is_available() else x

    with torch.no_grad():
        y, _ = net(x)


    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    boxes, _ = getDetBoxes(score_text, score_link, text_threshold=0.7, link_threshold=0.4, low_text=0.4, poly=False)
    boxes = np.array(boxes) * (2/target_ratio)

    return image, boxes

# Draw boxes on the image
def draw_boxes(image, boxes):
    for box in boxes:
        box = np.int32(box)
        cv2.polylines(image, [box.reshape((-1, 1, 2))], isClosed=True, color=(0, 255, 0), thickness=2)
    return image

# Main runner
if __name__ == '__main__':
    image_path = 'IMG_3964.jpg'  # Replace with your own image file
    craft_loading_time = datetime.now()
    
    model = load_craft_model()
    craft_loading_time = datetime.now() - craft_loading_time
    print(f"CRAFT model loaded in {craft_loading_time} seconds")
    craft_inference_time = datetime.now()
    image, boxes = detect_text(model, image_path)
    craft_inference_time = datetime.now() - craft_inference_time
    print(f"Segmentation time: {craft_inference_time} seconds")
    result = draw_boxes(image.copy(), boxes)
    
    trocr_loading_time = datetime.now()
    processor, trocr_model = load_trocr_model()
    trocr_loading_time = datetime.now() - trocr_loading_time
    print(f"TrOCR model loaded in {trocr_loading_time} seconds")
    trocr_inference_time = datetime.now()
    merged_boxes = merge_boxes(boxes)
    ocr_results = ocr_with_trocr(image, merged_boxes, processor, trocr_model)
    trocr_inference_time = datetime.now() - trocr_inference_time
    print(f"Inference time: {trocr_inference_time} seconds")
    result_with_text = draw_ocr_results(result.copy(), ocr_results)
    cv2.imwrite("output.jpg", result_with_text)

    # Show final result with text
    plt.figure(figsize=(12, 12))
    plt.imshow(cv2.cvtColor(result_with_text, cv2.COLOR_BGR2RGB))
    plt.title("CRAFT + TrOCR OCR")
    plt.axis('off')
    plt.show()

