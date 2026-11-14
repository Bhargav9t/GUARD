# ==============================================================================
# File: src/preprocessing/face_aligner.py
# Description: Input Face Extraction & Bounding-Box Alignment Pipeline for GUARD.
#
# Methodology:
#   1. Detects facial bounding boxes and landmarks.
#   2. Calculates rotation angle between eyes to align faces canonically.
#   3. Expands bounding box by a configurable margin (e.g. 25%) to preserve
#      blending seams around the jaw, forehead, and hair.
#   4. Applies affine similarity transformation or smart aspect-ratio preserving
#      cropping to reach target spatial resolution (e.g. 380x380).
#   5. Gracefully falls back to high-quality center-crop/resize if no face is
#      detected (essential for non-face AI generation detection).
# ==============================================================================

import os
from typing import Optional, Tuple, Union, Dict, Any
import numpy as np
import cv2
from configs.config import FaceAlignmentConfig, default_config
from src.utils.logger import app_logger


class FaceAligner:
    """
    Modular Face Alignment and Preprocessing engine for the GUARD framework.
    Supports face extraction, eye-based canonical rotation, margin padding,
    and automatic fallback for general scene imagery.
    """

    def __init__(self, config: Optional[FaceAlignmentConfig] = None):
        """
        Initializes the FaceAligner with configuration parameters.

        Parameters:
            config (Optional[FaceAlignmentConfig]): Alignment configuration.
                If None, loads the project's default_config.face_alignment.
        """
        # Assign or fall back to default configuration
        self.config = config or default_config.face_alignment

        # Target dimensions (Height, Width) for the spatial network input
        self.target_size = self.config.target_face_size

        # Margin expansion factor (e.g. 0.25 adds 25% padding on all sides)
        self.crop_margin = self.config.crop_margin

        # Minimum confidence threshold for valid face detections
        self.confidence_threshold = self.config.confidence_threshold

        self.face_cascade = None
        self.eye_cascade = None
        self._load_cascades()

        app_logger.info(
            f"FaceAligner initialized: TargetSize={self.target_size}, "
            f"CropMargin={self.crop_margin}, Enabled={self.config.enable_face_crop}"
        )

    def _load_cascades(self) -> None:
        """Loads Haar cascade classifiers if available."""
        cascade_dir = self._find_cascade_dir()
        if cascade_dir is not None:
            face_xml = os.path.join(cascade_dir, "haarcascade_frontalface_default.xml")
            eye_xml = os.path.join(cascade_dir, "haarcascade_eye.xml")
            if os.path.exists(face_xml):
                self.face_cascade = cv2.CascadeClassifier(face_xml)
            if os.path.exists(eye_xml):
                self.eye_cascade = cv2.CascadeClassifier(eye_xml)

        if self.face_cascade is not None and not self.face_cascade.empty():
            app_logger.info(f"FaceAligner: Successfully loaded frontal face cascade from {cascade_dir}")
        else:
            app_logger.warning("FaceAligner: Cascade detector not found. Operating in automatic center-crop mode.")

    def __getstate__(self) -> Dict[str, Any]:
        """Custom pickling for multi-worker DataLoader support on Windows."""
        state = self.__dict__.copy()
        # CascadeClassifier is an unpicklable C++ pointer; exclude from serialization
        state["face_cascade"] = None
        state["eye_cascade"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Reconstruct state and re-initialize cascades in spawned worker process."""
        self.__dict__.update(state)
        self._load_cascades()

    @staticmethod
    def _find_cascade_dir() -> Optional[str]:
        """
        Locates the directory containing OpenCV Haar Cascades across different
        Python packaging environments (pip, Miniforge/Conda Windows, Linux).

        Returns:
            Optional[str]: Path to directory containing haarcascades, or None if not found.
        """
        import sys

        candidate_dirs = []

        # 1. Standard pip package cv2.data (if exposed)
        if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            candidate_dirs.append(cv2.data.haarcascades)

        # 2. Miniforge / Conda on Windows: sys.prefix/Library/etc/haarcascades
        candidate_dirs.append(os.path.join(sys.prefix, "Library", "etc", "haarcascades"))

        # 3. Conda / Unix: sys.prefix/share/opencv4/haarcascades or share/OpenCV/haarcascades
        candidate_dirs.append(os.path.join(sys.prefix, "share", "opencv4", "haarcascades"))
        candidate_dirs.append(os.path.join(sys.prefix, "share", "OpenCV", "haarcascades"))

        # 4. Under site-packages/cv2/data
        cv2_dir = os.path.dirname(cv2.__file__)
        candidate_dirs.append(os.path.join(cv2_dir, "data"))

        for path in candidate_dirs:
            if path and os.path.isdir(path):
                # Verify that haarcascade_frontalface_default.xml actually exists inside
                test_file = os.path.join(path, "haarcascade_frontalface_default.xml")
                if os.path.isfile(test_file):
                    return path

        return None

    @staticmethod
    def load_image(image_input: Union[str, np.ndarray]) -> np.ndarray:
        """
        Loads an image from a file path or validates an existing NumPy array.
        Always returns an RGB image array with shape (H, W, 3) and dtype uint8.

        Parameters:
            image_input (Union[str, np.ndarray]): Filepath string or NumPy array.

        Returns:
            np.ndarray: Loaded RGB image in uint8 format.

        Raises:
            FileNotFoundError: If image file path does not exist.
            ValueError: If input format is invalid or cannot be decoded.
        """
        # If input is a file path string
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Input image path does not exist: {image_input}")

            # cv2.imread loads images in BGR color space
            bgr_img = cv2.imread(image_input)
            if bgr_img is None:
                raise ValueError(f"Failed to read image from path: {image_input}")

            # Convert BGR to standard RGB color space
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            return rgb_img

        # If input is already a NumPy array
        elif isinstance(image_input, np.ndarray):
            # If image has 2 dimensions (grayscale), duplicate across 3 channels
            if image_input.ndim == 2:
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            # If image has 3 dimensions (H, W, C)
            elif image_input.ndim == 3:
                # If channel count is 4 (RGBA), drop alpha channel
                if image_input.shape[2] == 4:
                    return cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
                # If channel count is 3, return as is
                return image_input
            else:
                raise ValueError(f"Unsupported image array dimensions: {image_input.ndim}")

        else:
            raise TypeError(f"Expected str or np.ndarray, got {type(image_input)}")

    def detect_primary_face(
        self,
        rgb_image: np.ndarray
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Detects the primary (most prominent/largest area) face in the image.

        Parameters:
            rgb_image (np.ndarray): Input RGB image array.

        Returns:
            Optional[Tuple[int, int, int, int]]: (x, y, w, h) bounding box of the face,
                                                 or None if no face is found.
        """
        # If face cascade failed to load, return None to trigger safe center-crop fallback
        if self.face_cascade is None or self.face_cascade.empty():
            return None

        # Convert RGB image to grayscale for intensity-based detection
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

        # Detect candidate face bounding boxes
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # If no faces were detected, return None
        if len(faces) == 0:
            return None

        # Select the face with the largest area (width * height) as the primary subject
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        return tuple(largest_face)

    def detect_eyes(
        self,
        rgb_image: np.ndarray,
        face_bbox: Tuple[int, int, int, int]
    ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Detects left and right eye centers within the upper half of the face region
        to compute horizontal eye alignment.

        Parameters:
            rgb_image (np.ndarray): Input RGB image array.
            face_bbox (Tuple[int, int, int, int]): Face bounding box (x, y, w, h).

        Returns:
            Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
                Coordinates of (left_eye, right_eye) in absolute image space,
                or None if both eyes cannot be reliably resolved.
        """
        # If eye cascade is unavailable, return None
        if self.eye_cascade is None or self.eye_cascade.empty():
            return None
        fx, fy, fw, fh = face_bbox
        # Restrict eye search to the upper 60% of the face region
        eye_search_h = int(fh * 0.6)
        face_gray = cv2.cvtColor(
            rgb_image[fy : fy + eye_search_h, fx : fx + fw],
            cv2.COLOR_RGB2GRAY
        )

        eyes = self.eye_cascade.detectMultiScale(
            face_gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(20, 20)
        )

        # If fewer than 2 eyes are detected, eye-angle alignment cannot be accurately resolved
        if len(eyes) < 2:
            return None

        # Sort eyes by their horizontal x-coordinate (from left to right)
        sorted_eyes = sorted(eyes, key=lambda e: e[0])

        # Eye 1 (left relative to image viewer)
        e1 = sorted_eyes[0]
        # Eye 2 (right relative to image viewer)
        e2 = sorted_eyes[-1]

        # Calculate center coordinates in global image space
        left_eye_center = (fx + e1[0] + e1[2] / 2.0, fy + e1[1] + e1[3] / 2.0)
        right_eye_center = (fx + e2[0] + e2[2] / 2.0, fy + e2[1] + e2[3] / 2.0)

        return left_eye_center, right_eye_center

    def expand_bbox(
        self,
        bbox: Tuple[int, int, int, int],
        image_shape: Tuple[int, int, int],
        margin: float = 0.25
    ) -> Tuple[int, int, int, int]:
        """
        Expands the bounding box outward by a given margin ratio while ensuring
        coordinates remain within the boundary limits of the source image.

        Crucial for deepfake detection: Blending artifacts frequently appear
        along outer edges (forehead, hairline, jawline, ears).

        Parameters:
            bbox (Tuple[int, int, int, int]): Original bounding box (x, y, w, h).
            image_shape (Tuple[int, int, int]): Shape of source image (H, W, C).
            margin (float): Fraction of width/height to pad around each side.

        Returns:
            Tuple[int, int, int, int]: Expanded bounding box (x1, y1, x2, y2).
        """
        img_h, img_w = image_shape[:2]
        x, y, w, h = bbox

        # Compute pixel margin offsets
        margin_x = int(w * margin)
        margin_y = int(h * margin)

        # Calculate expanded coordinates bounded by image edges
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(img_w, x + w + margin_x)
        y2 = min(img_h, y + h + margin_y)

        return (x1, y1, x2, y2)

    def align_and_crop(
        self,
        rgb_image: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
        eye_centers: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
    ) -> np.ndarray:
        """
        Performs affine similarity transformation (rotation for horizontal eye alignment)
        followed by cropping and resizing to the target resolution.

        Parameters:
            rgb_image (np.ndarray): Source RGB image.
            face_bbox (Tuple[int, int, int, int]): Detected face bbox (x, y, w, h).
            eye_centers (Optional[Tuple[Tuple, Tuple]]): Left and right eye centers.

        Returns:
            np.ndarray: Cropped, aligned, and scaled image of size `target_size`.
        """
        img_h, img_w = rgb_image.shape[:2]
        target_h, target_w = self.target_size

        # If both eyes were successfully located, calculate rotation matrix
        if eye_centers is not None:
            left_eye, right_eye = eye_centers

            # Compute delta differences
            dx = right_eye[0] - left_eye[0]
            dy = right_eye[1] - left_eye[1]

            # Calculate rotation angle in degrees
            angle = float(np.degrees(np.arctan2(dy, dx)))

            # Limit extreme rotations to +/- 30 degrees to prevent inversion/distortion
            angle = float(np.clip(angle, -30.0, 30.0))

            # Midpoint between eyes serves as the rotation pivot center
            eyes_center = (
                float((left_eye[0] + right_eye[0]) / 2.0),
                float((left_eye[1] + right_eye[1]) / 2.0)
            )

            # Compute 2D Affine Rotation Matrix (scale = 1.0)
            rot_matrix = cv2.getRotationMatrix2D(eyes_center, angle, scale=1.0)

            # Warp the full image with border reflection to prevent dark edges
            rotated_img = cv2.warpAffine(
                rgb_image,
                rot_matrix,
                (img_w, img_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101
            )
        else:
            # If eyes were not resolved, proceed with unrotated image
            rotated_img = rgb_image

        # Expand the face bounding box with the configured margin
        x1, y1, x2, y2 = self.expand_bbox(face_bbox, rotated_img.shape, margin=self.crop_margin)

        # Extract the cropped face region
        face_crop = rotated_img[y1:y2, x1:x2]

        # Guard against zero-sized crops
        if face_crop.size == 0:
            return self.fallback_center_crop(rgb_image)

        # Resize the cropped face region to the target network dimensions using bicubic interpolation
        aligned_face = cv2.resize(face_crop, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

        return aligned_face

    def fallback_center_crop(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        Fallback mechanism when face detection is disabled or no face is found.
        Performs a center-square crop preserving aspect ratio, then resizes to `target_size`.

        Parameters:
            rgb_image (np.ndarray): Source RGB image.

        Returns:
            np.ndarray: Square center-cropped and resized image of size `target_size`.
        """
        img_h, img_w = rgb_image.shape[:2]
        target_h, target_w = self.target_size

        # Find the shorter dimension to create a square bounding region
        min_dim = min(img_h, img_w)

        # Calculate top-left offsets for centered crop
        start_y = (img_h - min_dim) // 2
        start_x = (img_w - min_dim) // 2

        # Extract square center crop
        square_crop = rgb_image[start_y : start_y + min_dim, start_x : start_x + min_dim]

        # Resize to target resolution
        resized = cv2.resize(square_crop, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

        return resized

    def process(
        self,
        image_input: Union[str, np.ndarray]
    ) -> Dict[str, Any]:
        """
        End-to-end preprocessing pipeline.
        Loads image, detects face, extracts aligned region, or gracefully falls back.

        Parameters:
            image_input (Union[str, np.ndarray]): Input image path or NumPy array.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - 'aligned_image' (np.ndarray): Processed RGB image ready for model streams (H, W, 3).
                - 'face_detected' (bool): Whether a human face was successfully located.
                - 'bbox' (Optional[Tuple[int, int, int, int]]): Face bbox (x, y, w, h) if detected.
                - 'eye_centers' (Optional[Tuple]): Detected eye positions if resolved.
                - 'original_shape' (Tuple[int, int, int]): Original dimensions of input image.
        """
        # 1. Load and standardize image into RGB uint8 array
        rgb_img = self.load_image(image_input)
        original_shape = rgb_img.shape

        # 2. Check if face alignment is explicitly enabled
        if not self.config.enable_face_crop:
            return {
                "aligned_image": self.fallback_center_crop(rgb_img),
                "face_detected": False,
                "bbox": None,
                "eye_centers": None,
                "original_shape": original_shape,
            }

        # 3. Detect primary face bounding box
        face_bbox = self.detect_primary_face(rgb_img)

        # 4. If no face is detected, fall back to center-square cropping
        if face_bbox is None:
            return {
                "aligned_image": self.fallback_center_crop(rgb_img),
                "face_detected": False,
                "bbox": None,
                "eye_centers": None,
                "original_shape": original_shape,
            }

        # 5. Detect eyes for canonical tilt correction
        eye_centers = self.detect_eyes(rgb_img, face_bbox)

        # 6. Perform alignment, margin expansion, cropping, and resizing
        aligned_face = self.align_and_crop(rgb_img, face_bbox, eye_centers)

        return {
            "aligned_image": aligned_face,
            "face_detected": True,
            "bbox": face_bbox,
            "eye_centers": eye_centers,
            "original_shape": original_shape,
        }

    def visualize_alignment(
        self,
        image_input: Union[str, np.ndarray],
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Utility for visual inspection and debugging.
        Renders a side-by-side comparison of the original image with bounding
        boxes/landmarks alongside the aligned output crop.

        Parameters:
            image_input (Union[str, np.ndarray]): Input image path or array.
            output_path (Optional[str]): Optional destination to save comparison image.

        Returns:
            np.ndarray: Side-by-side composite RGB image.
        """
        # Run standard alignment pipeline
        res = self.process(image_input)
        orig_img = self.load_image(image_input)
        aligned = res["aligned_image"]

        # Create an annotated copy of the original image
        annotated = orig_img.copy()

        # If a face was detected, draw the bounding box and landmarks
        if res["face_detected"] and res["bbox"] is not None:
            x, y, w, h = res["bbox"]
            # Draw green bounding box around detected face
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 3)

            # Draw cyan dots on eye centers if detected
            if res["eye_centers"] is not None:
                left_eye, right_eye = res["eye_centers"]
                cv2.circle(annotated, (int(left_eye[0]), int(left_eye[1])), 6, (0, 255, 255), -1)
                cv2.circle(annotated, (int(right_eye[0]), int(right_eye[1])), 6, (0, 255, 255), -1)

        # Resize both panels to standard visualization height (e.g. 400px)
        viz_height = 400
        aspect_orig = annotated.shape[1] / annotated.shape[0]
        resized_orig = cv2.resize(annotated, (int(viz_height * aspect_orig), viz_height))
        resized_aligned = cv2.resize(aligned, (viz_height, viz_height))

        # Stack panels horizontally (Original + Annotated | Aligned & Cropped)
        composite = np.hstack([resized_orig, resized_aligned])

        # If output file path is specified, save to disk
        if output_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            # Convert RGB back to BGR for OpenCV imwrite
            cv2.imwrite(output_path, cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
            app_logger.info(f"Saved alignment visualization to: {output_path}")

        return composite
