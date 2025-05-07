import os
import cv2
import numpy as np
from skimage import color, exposure, feature, io, transform, util
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2lab, deltaE_ciede2000
import matplotlib.pyplot as plt
from pathlib import Path
import glob
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import time
import torch
from segment_anything import sam_model_registry, SamPredictor

class ColorDeviationDetector:
    def __init__(self):
        """Initialize the color deviation detection system."""
        self.min_reference = None
        self.std_reference = None
        self.max_reference = None
        self.region_map = None
        self.color_tolerance_map = None
        self.target_size = None
        self.sam_predictor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def initialize_sam(self, model_type="vit_b", checkpoint_path=r"C:\Users\adamm\Solos\Wings\claude\model\sam_vit_b_01ec64.pth"):
        """Initialize the Segment Anything Model for packaging segmentation."""
        print(f"Initializing SAM model on {self.device}...")
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(self.device)
        self.sam_predictor = SamPredictor(sam)
        print("SAM model initialized successfully.")
    
    def segment_packaging(self, image, auto_box=True, return_mask=False):
        """
        Segment the packaging in an image using SAM.
        
        Args:
            image: Input RGB image
            auto_box: If True, generate automatic bounding box. If False, use center-based box.
            
        Returns:
            Original image with non-packaging areas masked out
        """
        if self.sam_predictor is None:
            print("Warning: SAM model not initialized. Returning original image.")
            return image
            
        # Make a copy of the image for SAM (it expects RGB)
        if len(image.shape) == 2:  # If grayscale
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image.copy()
            
        # Set image for SAM predictor
        self.sam_predictor.set_image(image_rgb)
        
        # Generate bounding box automatically or use center-based approach
        h, w = image_rgb.shape[:2]
        if auto_box:
            # Simple automatic bounding box: assume product is centered and occupies most of the frame
            # Padding by 10% on each side from center
            center_x, center_y = w // 2, h // 2
            box_w, box_h = int(w * 0.8), int(h * 0.8)
            x1 = max(0, center_x - box_w // 2)
            y1 = max(0, center_y - box_h // 2)
            x2 = min(w, center_x + box_w // 2)
            y2 = min(h, center_y + box_h // 2)
        else:
            # Use most of the image as the box
            x1, y1 = int(w * 0.05), int(h * 0.05)
            x2, y2 = int(w * 0.95), int(h * 0.95)
            
        input_box = np.array([[0, 0, w, h]])
        #print(f'input_box: {input_box}')
        
        # Get segmentation mask from SAM
        print("Generating packaging segmentation mask...")
        masks, scores, _ = self.sam_predictor.predict(
            box=input_box,
            multimask_output=True  # Get multiple mask predictions
        )
        
        # Use the highest scoring mask
        best_mask_idx = np.argmax(scores)
        packaging_mask = masks[best_mask_idx]
        
        # Apply mask to original image
        masked_image = image_rgb.copy()
        # Set non-packaging regions to black or any other background color
        masked_image[~packaging_mask] = 0  # Black background
        
        print(f"Packaging segmentation complete. Mask confidence: {scores[best_mask_idx]:.3f}")
        # packaging_mask is your boolean NumPy array
        masked_image = image_rgb.copy()
        masked_image[~packaging_mask] = 0

        if return_mask:
            return masked_image, packaging_mask
        else:
            return masked_image
        
    def load_reference_images(self, min_path, std_path, max_path, output_dir=None):
            """
            Load and preprocess reference images.
            
            Args:
                min_path: Path to minimum acceptable image
                std_path: Path to standard reference image
                max_path: Path to maximum acceptable image
            """
            print("Loading reference images...")
            
            # Initialize SAM model if not already done
            if self.sam_predictor is None:
                self.initialize_sam()
                
            # Load and preprocess reference images
            min_img = self._load_image(min_path)
            std_img = self._load_image(std_path)
            max_img = self._load_image(max_path)
            
            # Segment packaging in reference images
            min_segmented, min_mask = self.segment_packaging(min_img, return_mask=True)
            std_segmented, std_mask = self.segment_packaging(std_img, return_mask=True)
            max_segmented, max_mask = self.segment_packaging(max_img, return_mask=True)

            if output_dir:
                self._save_mask(min_mask, os.path.join(output_dir, "min_reference_mask.png"))
                self._save_mask(std_mask, os.path.join(output_dir, "std_reference_mask.png"))
                self._save_mask(max_mask, os.path.join(output_dir, "max_reference_mask.png"))
                
            # Apply preprocessing
            self.min_reference = self._initial_preprocessing(min_segmented)
            self.std_reference = self._initial_preprocessing(std_segmented)
            self.max_reference = self._initial_preprocessing(max_segmented)
            
            # Ensure all reference images have the same size
            if (self.min_reference.shape != self.std_reference.shape or 
                self.min_reference.shape != self.max_reference.shape):
                # Resize to standard reference size
                h, w = self.std_reference.shape[:2]
                self.min_reference = cv2.resize(self.min_reference, (w, h))
                self.max_reference = cv2.resize(self.max_reference, (w, h))

            orig_h, orig_w = self.std_reference.shape[:2]
            half_h, half_w = orig_h // 3, orig_w // 3
            self.min_reference = cv2.resize(self.min_reference, (half_w, half_h))
            self.std_reference = cv2.resize(self.std_reference, (half_w, half_h))
            self.max_reference = cv2.resize(self.max_reference, (half_w, half_h))
            # remember for test images
            self.target_size = (half_w, half_h)

            print(f"Using image sizes of {self.target_size}")
                
            print("Reference images loaded, segmented, and preprocessed.")
            self._generate_region_map()
            if output_dir:
                self._save_region_map(output_dir)
            self._create_color_tolerance_map()
            
    def _save_region_map(self, output_dir):
        """
        Overlay the region_map on the Standard reference image and save it.
        """
        os.makedirs(output_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(self.std_reference)
        # Use a categorical colormap to color each region
        ax.imshow(self.region_map, cmap='tab20', alpha=0.5)
        ax.axis('off')
        ax.set_title("Region Map Overlay")
        save_path = os.path.join(output_dir, "standard_region_map.png")
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved region map overlay to: {save_path}")

    def _load_image(self, image_path):
        """Load image and convert to RGB."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
        
    def _load_and_preprocess(self, image_path):
        """
        Load and preprocess a single image.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Preprocessed image
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Basic preprocessing
        img = self._initial_preprocessing(img)
        
        return img
    
    def _initial_preprocessing(self, image):
        """
        Apply enhanced preprocessing to an image for consumer packaged goods analysis.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image with color normalization and enhancements
        """
        # 1. Color Consistency - Convert to LAB and normalize channels
        lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab_image)
        
        # Apply CLAHE to L channel to improve contrast while keeping colors intact
        # This helps with text readability and detail preservation on packaging
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        
        # 2. Package-Specific Enhancement - Bilateral filtering to reduce noise while preserving edges
        # This preserves sharp color transitions in logos and package design elements
        l_filtered = cv2.bilateralFilter(l_clahe, d=5, sigmaColor=75, sigmaSpace=75)
        
        # 3. Specular Highlight Reduction for glossy packaging
        # Simple threshold-based highlight reduction
        _, highlight_mask = cv2.threshold(l_filtered, 240, 255, cv2.THRESH_BINARY)
        highlight_mask_inv = cv2.bitwise_not(highlight_mask)
        l_highlight_reduced = cv2.bitwise_and(l_filtered, l_filtered, mask=highlight_mask_inv)
        # Soften the highlight reduction effect
        l_final = cv2.addWeighted(l_filtered, 0.7, l_highlight_reduced, 0.3, 0)
        
        # 4. Shadow Reduction - Simple shadow reduction using bottom-hat transform
        kernel = np.ones((15, 15), np.uint8)
        shadow_reduce = cv2.morphologyEx(l_final, cv2.MORPH_BLACKHAT, kernel)
        l_shadow_reduced = cv2.add(l_final, shadow_reduce)
        
        # Merge channels back
        lab_processed = cv2.merge([l_shadow_reduced, a, b])
        
        # Convert back to RGB
        rgb_processed = cv2.cvtColor(lab_processed, cv2.COLOR_LAB2RGB)
        
        # 5. Reference-Based Method - Histogram equalization to standardize overall color distribution
        # This will be a base normalization - your load_reference_images method will handle more precise matching
        yuv_image = cv2.cvtColor(rgb_processed, cv2.COLOR_RGB2YUV)
        # Equalize only the Y channel (luminance)
        yuv_image[:,:,0] = cv2.equalizeHist(yuv_image[:,:,0])
        rgb_equalized = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2RGB)
        
        # 6. Apply mild denoising as a final step to clean up any processing artifacts
        # Less aggressive than your original denoising to preserve detail
        denoised = cv2.fastNlMeansDenoisingColored(rgb_equalized, None, 5, 5, 7, 15)
        
        return denoised
    
    def _generate_region_map(self):
        """Generate a region map from the standard reference image."""
        print("Generating region map...")
        # Convert to LAB color space for better color segmentation
        lab_image = cv2.cvtColor(self.std_reference, cv2.COLOR_RGB2LAB)
        
        # Apply mean shift segmentation
        shifted = cv2.pyrMeanShiftFiltering(lab_image, 10, 20)
        
        # Apply simple segmentation for demonstration
        # In a real system, more sophisticated segmentation would be used
        gray = cv2.cvtColor(shifted, cv2.COLOR_LAB2BGR)
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        
        # Threshold to create regions
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply morphological operations to clean up the regions
        kernel = np.ones((5,5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Connected component analysis
        num_labels, labels = cv2.connectedComponents(thresh)
        
        # Merge small regions (simplified approach)
        min_region_size = 100  # Minimum region size in pixels
        for i in range(1, num_labels):
            if np.sum(labels == i) < min_region_size:
                labels[labels == i] = 0
        
        # Re-label regions
        self.region_map = labels
        print(f"Generated region map with {len(np.unique(self.region_map))} unique regions.")
    
    def _create_color_tolerance_map(self):
        """Create a color tolerance map based on min and max reference images."""
        print("Creating color tolerance map...")
        # Convert reference images to LAB color space
        min_lab = cv2.cvtColor(self.min_reference, cv2.COLOR_RGB2LAB).astype(np.float32)
        std_lab = cv2.cvtColor(self.std_reference, cv2.COLOR_RGB2LAB).astype(np.float32)
        max_lab = cv2.cvtColor(self.max_reference, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # Calculate tolerance bounds
        min_tolerance = min_lab - std_lab
        max_tolerance = max_lab - std_lab
        
        # Store tolerance maps
        self.color_tolerance_map = {
            'min': min_tolerance,
            'std': std_lab,
            'max': max_tolerance
        }
        print("Color tolerance map created.")
    
    def _save_mask(self,
                   mask: np.ndarray,
                   path: str,
                   image: np.ndarray = None,
                   box: np.ndarray = None,
                   alpha: float = 0.6):
        """
        Saves either:
          • a raw mask (black & white), or
          • an overlay of mask+box on the original image.

        Args:
          mask: Boolean array or 0/1 mask.
          path: Full .png path where to write output.
          image: (Optional) RGB array to overlay.
          box:   (Optional) [[x1, y1, x2, y2]] bounding box matching the mask.
          alpha: Transparency for mask overlay.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # If no original image provided, just dump the mask
        if image is None or box is None:
            cv2.imwrite(path, (mask.astype(np.uint8) * 255))
            return

        # 1) Make a color‐converted copy of the image for drawing
        overlay = image.copy()

        # 2) Draw the box in red
        x1, y1, x2, y2 = box[0]
        cv2.rectangle(overlay, (x1, y1), (x2, y2),
                      color=(255, 0, 0), thickness=2)

        # 3) Prepare a colored mask (e.g. jet colormap)
        #    First scale mask to [0,255]
        mask_8u = (mask.astype(np.uint8) * 255)
        #    Apply a matplotlib colormap
        cmap = plt.get_cmap('jet')
        colored_mask = cmap(mask_8u / 255.0)[:, :, :3]  # drop alpha
        #    Convert from float [0–1] to uint8 [0–255]
        colored_mask = (colored_mask * 255).astype(np.uint8)

        # 4) Blend overlay and colored_mask
        blended = cv2.addWeighted(overlay, 1 - alpha,
                                  colored_mask, alpha, 0)

        # 5) Save the result
        #    Convert RGB->BGR for OpenCV write
        bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, bgr)


    def process_test_image(self, image_path, output_dir=None):
        """
        Process a test image and check for color deviations.
        
        Args:
            image_path: Path to the test image
            output_dir: Directory to save visualization results (optional)
            
        Returns:
            Dictionary containing deviation results
        """
        if self.min_reference is None or self.std_reference is None or self.max_reference is None:
            raise ValueError("Reference images must be loaded first")
        
        print(f"Processing test image: {image_path}")
        total_start = time.time()
        # Load test image
        test_image = self._load_image(image_path)
        
        # Segment packaging in test image
        segmented_test, test_mask = self.segment_packaging(test_image, return_mask=True)
        # dump it if requested
        if output_dir:
            base = Path(image_path).stem
            mask_path = os.path.join(output_dir, f"{base}_test_mask.png")
            self._save_mask(test_mask, mask_path)

        # Preprocess test image
        test_image = self._initial_preprocessing(segmented_test)
        
        total_end = time.time()
        elapsed_time_total = total_end - total_start
        print(f"Load and process time taken: {elapsed_time_total:.2f} seconds\n")

        if self.target_size is not None:
            test_image = cv2.resize(test_image, self.target_size)
        
        # Phase 2: Image Registration & Normalization
        total_start = time.time()
        aligned_image = self._align_image(test_image)
        normalized_image = self._normalize_illumination(aligned_image)
        total_end = time.time()
        elapsed_time_total = total_end - total_start
        print(f"Align & Normalization time taken: {elapsed_time_total:.2f} seconds\n")
        
        # Phase 4: Deviation Detection & Quantification
        total_start = time.time()
        deviation_results = self._detect_deviations(normalized_image, test_mask)
        total_end = time.time()
        elapsed_time_total = total_end - total_start
        print(f"Deviation Detection time taken: {elapsed_time_total:.2f} seconds\n")
        
        # Generate visualization if output directory is provided
        if output_dir:
            self._generate_visualization(normalized_image,
                             deviation_results,
                             image_path,
                             output_dir,
                             segmentation_mask=test_mask)
        
        return deviation_results
    
    def _align_image(self, image):
        """
        Align the test image with the reference image.
        
        Args:
            image: Input test image
            
        Returns:
            Aligned image
        """
        print("Aligning image...")
        # Convert images to grayscale for feature detection
        gray_reference = cv2.cvtColor(self.std_reference, cv2.COLOR_RGB2GRAY)
        gray_test = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Detect ORB features and compute descriptors
        orb = cv2.ORB_create(nfeatures=1000)
        kp1, des1 = orb.detectAndCompute(gray_reference, None)
        kp2, des2 = orb.detectAndCompute(gray_test, None)
        
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            print("Warning: Not enough features detected. Using original image.")
            # Resize to match reference
            return cv2.resize(image, (self.std_reference.shape[1], self.std_reference.shape[0]))
        
        # Create BFMatcher object and match descriptors
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        
        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Use top matches to find homography
        if len(matches) > 10:
            # Extract location of good matches
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
            
            # Find homography
            H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            
            if H is not None:
                # Apply homography
                h, w = self.std_reference.shape[:2]
                aligned = cv2.warpPerspective(image, H, (w, h))
                print("Image alignment successful.")
                return aligned
        
        print("Warning: Could not align image properly. Using resized image.")
        # Fallback to simple resize if alignment fails
        return cv2.resize(image, (self.std_reference.shape[1], self.std_reference.shape[0]))
    
    def _normalize_illumination(self, image):
        """
        Normalize illumination in the test image.
        
        Args:
            image: Input test image aligned with reference
            
        Returns:
            Illumination-normalized image
        """
        print("Normalizing illumination...")
        # Ensure input image is uint8 (common for images)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        
        # Convert to LAB color space
        lab_reference = cv2.cvtColor(self.std_reference, cv2.COLOR_RGB2LAB)
        lab_test = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Split channels
        l_ref, a_ref, b_ref = cv2.split(lab_reference)
        l_test, a_test, b_test = cv2.split(lab_test)
        
        # Match histograms of L channel
        # Convert to uint8 to avoid depth issues
        l_ref_uint8 = l_ref.astype(np.uint8)
        l_test_uint8 = l_test.astype(np.uint8)
        
        try:
            l_matched = exposure.match_histograms(l_test_uint8, l_ref_uint8)
        except Exception as e:
            print(f"Warning: Histogram matching failed: {e}. Using original L channel.")
            l_matched = l_test_uint8
        
        # Ensure all channels have the same size
        h, w = l_matched.shape
        a_test_resized = cv2.resize(a_test, (w, h))
        b_test_resized = cv2.resize(b_test, (w, h))
        
        # Ensure all channels are uint8
        l_matched = l_matched.astype(np.uint8)
        a_test_resized = a_test_resized.astype(np.uint8)
        b_test_resized = b_test_resized.astype(np.uint8)
        
        try:
            # Recombine channels
            lab_matched = cv2.merge([l_matched, a_test_resized, b_test_resized])
            
            # Convert back to RGB
            normalized = cv2.cvtColor(lab_matched, cv2.COLOR_LAB2RGB)
        except Exception as e:
            print(f"Warning: Channel merging or color conversion failed: {e}. Using original image.")
            normalized = image
        
        print("Illumination normalization complete.")
        return normalized  
    
    def _detect_deviations(self, test_image, segmentation_mask):
        """
        Detect color deviations between test_image and reference,
        only inside the SAM mask (resized to match).
        """
        # 1) Ensure the mask is the same H×W as test_image
        h, w = test_image.shape[:2]
        if segmentation_mask.shape != (h, w):
            # resize nearest-neighbor so True/False stays intact
            mask = cv2.resize(
                segmentation_mask.astype(np.uint8),
                (w, h),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        else:
            mask = segmentation_mask

        # 2) Convert to LAB
        test_lab = cv2.cvtColor(test_image, cv2.COLOR_RGB2LAB).astype(np.float32)
        std_lab  = self.color_tolerance_map['std']
        min_tol  = self.color_tolerance_map['min']
        max_tol  = self.color_tolerance_map['max']

        # 3) Force out‑of‑mask pixels to match Standard exactly
        for c in range(3):
            test_lab[:,:,c][~mask] = std_lab[:,:,c][~mask]

        # 4) Simple channel bounds check
        diff = test_lab - std_lab
        below = np.zeros((h, w), dtype=bool)
        above = np.zeros((h, w), dtype=bool)
        for c in range(3):
            below |= (diff[:,:,c] <  min_tol[:,:,c])
            above |= (diff[:,:,c] >  max_tol[:,:,c])
        deviation_mask = (below | above) & mask

        # 5) Compute ΔE CIEDE2000 (sample every 10px, then smooth)
        delta_e = np.zeros((h, w), dtype=float)
        for i in range(0, h, 10):
            for j in range(0, w, 10):
                lab1 = np.array([[test_lab[i,j]]])
                lab2 = np.array([[std_lab[i,j]]])
                val = deltaE_ciede2000(lab1, lab2)
                delta_e[i:i+10, j:j+10] = val
        delta_e = gaussian_filter(delta_e, sigma=3)
        delta_e[~mask] = 0  # zero out outside package

        # 6) Region‑by‑region stats (only inside mask)
        region_deviations = {}
        for region_id in np.unique(self.region_map):
            if region_id == 0:
                continue
            region_zone = (self.region_map == region_id) & mask
            count = np.sum(region_zone)
            if count == 0:
                continue
            dev_pct = np.sum(deviation_mask & region_zone) / count * 100
            avg_de = np.mean(delta_e[region_zone])
            region_deviations[region_id] = {
                'deviation_percentage': dev_pct,
                'avg_delta_e':          avg_de,
                'size':                 int(count)
            }

        # 7) Overall metrics (only inside mask)
        total_inside = np.sum(mask)
        overall_pct = np.sum(deviation_mask) / total_inside * 100 if total_inside else 0.0
        avg_de_all  = np.mean(delta_e[mask]) if total_inside else 0.0
        max_de_all  = np.max(delta_e[mask]) if total_inside else 0.0

        print(f"Detected {overall_pct:.2f}% overall deviation, average ΔE {avg_de_all:.2f}")

        return {
            'overall_deviation_percentage': overall_pct,
            'average_delta_e':              avg_de_all,
            'max_delta_e':                  max_de_all,
            'region_deviations':            region_deviations,
            'deviation_mask':               deviation_mask,
            'delta_e_map':                  delta_e
        }


    def _generate_visualization(self,
                                test_image,
                                deviation_results,
                                image_path,
                                output_dir,
                                segmentation_mask):
        """
        Save a 3‑panel figure:
          1) normalized image
          2) ΔE heatmap
          3) red/green overlay inside the package mask
        """
        # 1) Resize mask if needed
        h, w = test_image.shape[:2]
        if segmentation_mask.shape != (h, w):
            mask = cv2.resize(
                segmentation_mask.astype(np.uint8),
                (w, h),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        else:
            mask = segmentation_mask

        base = Path(image_path).stem
        os.makedirs(output_dir, exist_ok=True)

        plt.figure(figsize=(20, 10))

        # Panel 1: normalized test image
        plt.subplot(1, 3, 1)
        plt.imshow(test_image); plt.axis('off')
        plt.title('Normalized Test Image')

        # Panel 2: ΔE heatmap
        plt.subplot(1, 3, 2)
        de_map = deviation_results['delta_e_map']
        plt.imshow(de_map, cmap='hot', vmin=0, vmax=10)
        plt.colorbar(label='ΔE CIEDE2000'); plt.axis('off')
        plt.title('Color Difference Heatmap')

        # Panel 3: red/green overlay inside mask
        plt.subplot(1, 3, 3)
        dev_mask = deviation_results['deviation_mask']
        color_mask = np.zeros_like(test_image)
        color_mask[(mask &  dev_mask)] = [255, 0,   0]  # red out‑of‑bounds
        color_mask[(mask & ~dev_mask)] = [  0, 255, 0]  # green in‑bounds
        overlay = cv2.addWeighted(test_image, 0.5, color_mask, 0.5, 0)
        plt.imshow(overlay); plt.axis('off')
        pct = deviation_results['overall_deviation_percentage']
        plt.title(f'Deviation Overlay ({pct:.2f}% OOB)')

        # Save
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{base}_visualization.png"))
        plt.close()

        # Text report (unchanged)
        with open(os.path.join(output_dir, f"{base}_report.txt"), 'w') as f:
            f.write(f"Color Deviation Report for {base}\n")
            f.write("==================================\n\n")
            f.write(f"Overall Deviation: {pct:.2f}%\n")
            f.write(f"Average ΔE: {deviation_results['average_delta_e']:.2f}\n")
            f.write(f"Maximum ΔE: {deviation_results['max_delta_e']:.2f}\n\n")
            f.write("Region-by-Region Analysis:\n")
            for rid, stats in sorted(
                    deviation_results['region_deviations'].items(),
                    key=lambda x: x[1]['deviation_percentage'],
                    reverse=True):
                f.write(f"Region {rid}:\n")
                f.write(f"  - Size: {stats['size']} pixels\n")
                f.write(f"  - Deviation: {stats['deviation_percentage']:.2f}%\n")
                f.write(f"  - Average ΔE: {stats['avg_delta_e']:.2f}\n\n")

def process_images(min_path, std_path, max_path, test_folder, output_dir):
    """
    Process all test images in a folder.
    
    Args:
        min_path: Path to minimum reference image
        std_path: Path to standard reference image
        max_path: Path to maximum reference image
        test_folder: Folder containing test images
        output_dir: Directory to save visualization results
    """
    # Initialize detector
    detector = ColorDeviationDetector()
    
    # Load reference images
    detector.load_reference_images(min_path, std_path, max_path, output_dir=output_dir)
    
    # Get all images in test folder
    test_images = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff'):
        test_images.extend(glob.glob(os.path.join(test_folder, ext)))
    
    print(f"Found {len(test_images)} test images to process.")
    
    # Process each test image
    results = {}
    for img_path in tqdm(test_images):
        try:
            result = detector.process_test_image(img_path, output_dir)
            results[img_path] = result
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Generate summary report
    generate_summary_report(results, output_dir)
    
    return results

def generate_summary_report(results, output_dir):
    """
    Generate a summary report of all processed images.
    
    Args:
        results: Dictionary of results for each image
        output_dir: Directory to save the report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "summary_report.txt"), 'w') as f:
        f.write(f"Color Deviation Detection - Summary Report\n")
        f.write(f"=========================================\n\n")
        f.write(f"Total images processed: {len(results)}\n\n")
        
        # Sort images by overall deviation (descending)
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1]['overall_deviation_percentage'],
            reverse=True
        )
        
        f.write(f"Image Rankings by Deviation:\n")
        f.write(f"--------------------------\n")
        
        for i, (img_path, result) in enumerate(sorted_results):
            img_name = os.path.basename(img_path)
            f.write(f"{i+1}. {img_name}\n")
            f.write(f"   - Overall Deviation: {result['overall_deviation_percentage']:.2f}%\n")
            f.write(f"   - Average Delta E: {result['average_delta_e']:.2f}\n")
            f.write(f"   - Maximum Delta E: {result['max_delta_e']:.2f}\n\n")
        
        # Calculate statistics
        overall_deviations = [r['overall_deviation_percentage'] for _, r in results.items()]
        avg_deltas = [r['average_delta_e'] for _, r in results.items()]
        
        if overall_deviations:
            f.write(f"Batch Statistics:\n")
            f.write(f"----------------\n")
            f.write(f"Mean Overall Deviation: {np.mean(overall_deviations):.2f}%\n")
            f.write(f"Median Overall Deviation: {np.median(overall_deviations):.2f}%\n")
            f.write(f"Mean Delta E: {np.mean(avg_deltas):.2f}\n")

# Example usage
if __name__ == "__main__":
    # Replace these with actual paths
    min_reference_path = "path/to/min_reference.jpg"
    std_reference_path = "path/to/std_reference.jpg"
    max_reference_path = "path/to/max_reference.jpg"
    test_images_folder = "path/to/test_images/"
    output_directory = "path/to/output/"
    
    results = process_images(
        min_reference_path,
        std_reference_path,
        max_reference_path,
        test_images_folder,
        output_directory
    )
    
    print("Processing complete. Results saved to:", output_directory)