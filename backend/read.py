import cv2
import numpy as np
import pytesseract
import time
from difflib import SequenceMatcher
import matplotlib.pyplot as plt


def combine_detected_segments(image_path, ground_truth=None, display_steps=True):
    """
    Alternative approach: Detect the entire text first, then segment it based on 
    spaces or known patterns, and annotate each segment separately.
    
    Args:
        image_path: Path to the input image
        ground_truth: Optional ground truth text for accuracy evaluation
        display_steps: If True, display intermediate processing steps
    
    Returns:
        Annotated image with text segments highlighted and labeled,
        detected word positions, and performance metrics
    """
    # Start timing
    start_time = time.time()
    
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    
    result = img.copy()
    
    # Display original image
    if display_steps:
        print("Original Image:")
        display_cv_image(img)
    
    # Preprocess for better recognition
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.0, beta=0.05) 
    # Resize to improve OCR accuracy
    scale_factor = 2
    gray_resized = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    
    if display_steps:
        print("Resized Grayscale Image:")
        display_cv_image(gray_resized)
    
    # Apply threshold
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # For digits with connecting strokes like 4
    kernel = np.ones((2, 2), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # For digits with loops like 5 and 9
    kernel = np.ones((1, 1), np.uint8)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    
    if display_steps:
        print("Binary Image:")
        display_cv_image(closed)
    
    # Start timing OCR inference
    inference_start = time.time()
    
    # Get the entire text
    custom_config = r'--oem 3 --psm 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ: '
    ocr_data = pytesseract.image_to_data(closed, output_type=pytesseract.Output.DICT)
    ocr_data
    text = ocr_data['text']
    text_conf = [text[i] for i in range(len(text)) if ocr_data['conf'][i]>=40]
    full_text = " ".join(text_conf)
    
    # End timing OCR inference
    inference_end = time.time()
    inference_time = inference_end - inference_start
    
    # Split by spaces to get segments
    segments = full_text.split()
    print(f"Detected full text: '{full_text}'")
    print(f"Segments: {segments}")
    print(f"OCR inference time: {inference_time:.4f} seconds")
    
    # Get detailed OCR data to find word positions
    #ocr_data = pytesseract.image_to_data(closed, output_type=pytesseract.Output.DICT)
    #print(ocr_data)
    
    # Create a resized result image for annotation
    result_resized = cv2.resize(result, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC)
    
    # Process each word from OCR data
    detected_word_positions = []
    
    for i in range(len(ocr_data['text'])):
        word = ocr_data['text'][i].strip()
        
        # Only process non-empty words with sufficient confidence
        if word and int(ocr_data['conf'][i]) >= 40:
            # Extract box coordinates
            x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
            
            # Draw box
            cv2.rectangle(result_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Add text label
            cv2.putText(result_resized, word, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            detected_word_positions.append((word, (x, y, w, h)))
            print(f"Word '{word}' found at (x={x}, y={y}, w={w}, h={h})")
    
    if display_steps:
        print("Detected Words with Bounding Boxes:")
        display_cv_image(result_resized)
    
    # Resize back to original dimensions
    result = cv2.resize(result_resized, (img.shape[1], img.shape[0]))
    
    print("\nFinal Result (resized back to original dimensions):")
    display_cv_image(result)
    
    # End timing for total process
    end_time = time.time()
    total_time = end_time - start_time
    
    # Performance metrics
    performance = {
        'total_time': total_time,
        'total_segments': len(detected_word_positions),
        'inference_time': inference_time,
        'avg_inference_time': inference_time  # For consistent metrics structure with the first method
    }
    
    print(f"\nPerformance Metrics:")
    print(f"Total processing time: {total_time:.4f} seconds")
    print(f"OCR inference time: {inference_time:.4f} seconds")
    
    # Calculate accuracy if ground truth is provided
    # if ground_truth:
    #     accuracy_metrics = calculate_character_accuracy(full_text, ground_truth)
    #     performance.update(accuracy_metrics)
        
    #     print(f"\nAccuracy Metrics:")
    #     print(f"Character accuracy: {accuracy_metrics['character_accuracy']:.2f}%")
    #     print(f"Precision: {accuracy_metrics['precision']:.2f}%")
    #     print(f"Recall: {accuracy_metrics['recall']:.2f}%")
    #     print(f"F1 Score: {accuracy_metrics['f1_score']:.2f}%")
    #     print(f"Levenshtein distance: {accuracy_metrics['levenshtein_distance']}")
    
    return result, detected_word_positions, performance



def run_ocr_on_cropped_image(image: np.ndarray):
    import cv2
    import numpy as np
    import pytesseract

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.0, beta=0.05)
    resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    binary = cv2.adaptiveThreshold(resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((1, 1), np.uint8))

    config = r'--oem 3 --psm 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ: '
    ocr_data = pytesseract.image_to_data(closed, output_type=pytesseract.Output.DICT, config=config)

    results = []
    full_text = []

    for i in range(len(ocr_data['text'])):
        word = ocr_data['text'][i].strip()
        conf = int(ocr_data['conf'][i])
        if word and conf >= 40:
            x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
            results.append({'text': word, 'box': [x, y, x + w, y + h], 'confidence': conf})
            full_text.append(word)

    return {
        'text': " ".join(full_text) if full_text else "No text found",
        'boxes': results
    }


def run_full_ocr_pipeline_on_image_array(image: np.ndarray, save_path="ocr_result.jpg", display=False):
    """
    Accepts a BGR image (as a NumPy array), runs OCR preprocessing, recognition,
    draws bounding boxes with text labels, saves the result, and returns the detected text info.

    Args:
        image (np.ndarray): Input BGR image.
        save_path (str): Where to save the result image with boxes.
        display (bool): Whether to display intermediate images (disabled for backend use).

    Returns:
        full_text (str): Concatenated detected text.
        word_boxes (list): List of tuples (word, (x, y, w, h))
        metrics (dict): Timing and inference info.
    """

    start_time = time.time()

    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.0, beta=0.05)
    scale_factor = 2
    resized = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    binary = cv2.adaptiveThreshold(resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((1, 1), np.uint8))

    config = r'--oem 3 --psm 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ: '
    inference_start = time.time()
    ocr_data = pytesseract.image_to_data(closed, output_type=pytesseract.Output.DICT, config=config)
    inference_time = time.time() - inference_start

    result = original.copy()
    word_boxes = []
    full_text_tokens = []

    for i in range(len(ocr_data['text'])):
        word = ocr_data['text'][i].strip()
        if word and int(ocr_data['conf'][i]) >= 40:
            x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(result, word, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            word_boxes.append((word, (x, y, w, h)))
            full_text_tokens.append(word)

    cv2.imwrite(save_path, result)

    total_time = time.time() - start_time
    metrics = {
        "inference_time": inference_time,
        "total_time": total_time,
        "total_words": len(word_boxes)
    }

    return " ".join(full_text_tokens) if full_text_tokens else "No text found", word_boxes, metrics




import cv2
import time
import pytesseract
import numpy as np

