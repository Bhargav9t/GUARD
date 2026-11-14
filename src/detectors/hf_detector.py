"""
Hugging Face Pretrained Model Detector for GUARD.
Wraps Organika/sdxl-detector or umm-maybe/AI-image-detector via transformers.
Includes automatic RTX 3050 CUDA FP16 acceleration with CPU fallback,
class label normalization, and PyTorch-GradCAM spatial heatmap extraction.
"""

import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForImageClassification

from src.detectors.base import BaseDetector, DetectionResult


class HFModelGradCAMWrapper(torch.nn.Module):
    """
    Lightweight forward wrapper for Hugging Face image classification models.
    PyTorch-GradCAM expects model(x) to directly return a raw logits Tensor,
    whereas Hugging Face models return an ImageClassifierOutput object.
    """
    def __init__(self, hf_model: torch.nn.Module):
        super().__init__()
        self.hf_model = hf_model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.hf_model(pixel_values=pixel_values)
        if hasattr(out, "logits"):
            return out.logits
        return out


def swin_reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
    """
    Reshape transform required by PyTorch-GradCAM for Swin Transformer stages.
    Swin layers output (B, N, C) where N = H * W (e.g. 49 = 7x7).
    This converts (B, N, C) -> (B, C, H, W) for spatial convolutional Grad-CAM.
    """
    if len(tensor.shape) == 3:
        b, n, c = tensor.shape
        h = int(math.isqrt(n))
        w = h
        if h * w != n:
            # Non-square fallback
            h = int(math.sqrt(n))
            w = n // h
        reshaped = tensor.reshape(b, h, w, c)
        return reshaped.permute(0, 3, 1, 2)
    elif len(tensor.shape) == 4 and tensor.shape[1] != tensor.shape[3]:
        # Handle (B, H, W, C) -> (B, C, H, W)
        return tensor.permute(0, 3, 1, 2)
    return tensor


class HuggingFaceDetector(BaseDetector):
    """
    Inference engine wrapping Hugging Face image detectors.
    Configured for Organika/sdxl-detector or umm-maybe/AI-image-detector.
    """

    SUPPORTED_MODELS = [
        "Organika/sdxl-detector",
        "umm-maybe/AI-image-detector",
    ]

    def __init__(
        self,
        model_id: str = "Organika/sdxl-detector",
        force_cpu: bool = False,
        use_fp16_on_cuda: bool = True,
    ):
        """
        Initializes the Hugging Face detector.

        Parameters:
            model_id: Hugging Face model repository ID.
            force_cpu: If True, forces CPU execution even if CUDA is available.
            use_fp16_on_cuda: If True, runs in FP16 on CUDA (essential for 6GB RTX 3050).
        """
        self._model_id = model_id
        self._use_fp16_requested = use_fp16_on_cuda

        # 1. Hardware Resolution with RTX 3050 FP16 Optimization & CPU Fallback
        if not force_cpu and torch.cuda.is_available():
            self._device = torch.device("cuda:0")
            device_name = torch.cuda.get_device_name(0)
            # FP16 is fully supported on RTX 3050 (Ampere architecture, compute cap 8.6)
            self._is_fp16 = use_fp16_on_cuda
            self._dtype = torch.float16 if self._is_fp16 else torch.float32
            print(f"[GUARD HFDetector] Active Hardware: CUDA ({device_name}) | Precision: {self._dtype}")
        else:
            self._device = torch.device("cpu")
            self._is_fp16 = False
            self._dtype = torch.float32
            print("[GUARD HFDetector] Active Hardware: CPU Fallback | Precision: torch.float32")

        # 2. Load Pretrained Processor and Model
        print(f"[GUARD HFDetector] Loading weights for '{model_id}'...")
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_id)
            self.model = AutoModelForImageClassification.from_pretrained(
                model_id,
                torch_dtype=self._dtype if self._device.type == "cuda" else torch.float32,
            )
            self.model.to(self._device)
            self.model.eval()
        except Exception as e:
            if self._device.type == "cuda":
                print(f"[GUARD HFDetector] CUDA load failed ({e}), falling back to CPU...")
                self._device = torch.device("cpu")
                self._is_fp16 = False
                self._dtype = torch.float32
                self.processor = AutoImageProcessor.from_pretrained(model_id)
                self.model = AutoModelForImageClassification.from_pretrained(model_id)
                self.model.to(self._device)
                self.model.eval()
            else:
                raise e

        # 3. Resolve and Normalize Class Label Mappings
        self._setup_label_mapping()

        # 4. Initialize GradCAM Wrapper
        self._cam_wrapper = HFModelGradCAMWrapper(self.model)
        self._target_layers = self._resolve_target_layers()

    def _setup_label_mapping(self):
        """
        Normalizes various Hugging Face label formats into standard:
        Index corresponding to AI-generated / artificial vs Real / human.
        """
        raw_id2label = getattr(self.model.config, "id2label", {0: "artificial", 1: "human"})
        self.raw_id2label = {int(k): str(v).lower() for k, v in raw_id2label.items()}

        self.ai_class_idx = 0
        self.real_class_idx = 1

        for idx, lbl in self.raw_id2label.items():
            if any(term in lbl for term in ["artificial", "fake", "ai", "synthetic", "generated"]):
                self.ai_class_idx = idx
            elif any(term in lbl for term in ["human", "real", "authentic", "natural"]):
                self.real_class_idx = idx

        print(f"[GUARD HFDetector] Class normalization: AI={self.ai_class_idx} ({self.raw_id2label.get(self.ai_class_idx)}), REAL={self.real_class_idx} ({self.raw_id2label.get(self.real_class_idx)})")

    def _resolve_target_layers(self) -> List[torch.nn.Module]:
        """
        Finds the last spatial norm/layer in the backbone for Grad-CAM.
        Supports Swin Transformer, ViT, ConvNeXt, and ResNet architectures.
        """
        target_layers = []
        if hasattr(self.model, "swin"):
            # Swin Transformer
            if hasattr(self.model.swin, "layernorm"):
                target_layers.append(self.model.swin.layernorm)
            elif hasattr(self.model.swin, "encoder") and len(self.model.swin.encoder.layers) > 0:
                last_stage = self.model.swin.encoder.layers[-1]
                if hasattr(last_stage, "blocks") and len(last_stage.blocks) > 0:
                    last_block = last_stage.blocks[-1]
                    target_layers.append(getattr(last_block, "layernorm_after", last_block))
        elif hasattr(self.model, "vit"):
            # ViT
            if hasattr(self.model.vit, "layernorm"):
                target_layers.append(self.model.vit.layernorm)
        elif hasattr(self.model, "convnext"):
            # ConvNeXt
            if hasattr(self.model.convnext, "layernorm"):
                target_layers.append(self.model.convnext.layernorm)
        elif hasattr(self.model, "base_model"):
            target_layers.append(self.model.base_model)
        else:
            # Fallback to last child module before classifier
            children = list(self.model.children())
            if len(children) >= 2:
                target_layers.append(children[-2])
            elif len(children) > 0:
                target_layers.append(children[0])

        return target_layers

    def _preprocess(self, image: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        """Converts input image to model tensor on the active device and dtype."""
        if isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image)
        else:
            pil_img = image

        inputs = self.processor(images=pil_img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device=self._device)
        if self._is_fp16 and self._device.type == "cuda":
            pixel_values = pixel_values.half()
        return pixel_values

    def predict_image(
        self,
        image: Union[np.ndarray, Image.Image],
        threshold: float = 0.50,
    ) -> DetectionResult:
        """
        Runs single image forward pass and extracts uncalibrated softmax probabilities.
        """
        t0 = time.perf_counter()
        pixel_values = self._preprocess(image)

        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)
            logits = outputs.logits.float()  # Ensure float32 for softmax stability
            probs = F.softmax(logits, dim=-1).squeeze(0)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        ai_prob = float(probs[self.ai_class_idx].item())
        real_prob = float(probs[self.real_class_idx].item())
        is_ai = bool(ai_prob >= threshold)

        # Uncalibrated confidence metric: distance from decision uncertainty
        confidence = float(max(ai_prob, real_prob))

        predicted_label = "AI-Generated" if is_ai else "Real / Human"

        raw_scores = {
            self.raw_id2label.get(self.ai_class_idx, "artificial"): ai_prob,
            self.raw_id2label.get(self.real_class_idx, "human"): real_prob,
        }

        return DetectionResult(
            is_ai=is_ai,
            ai_probability=ai_prob,
            confidence=confidence,
            predicted_label=predicted_label,
            raw_scores=raw_scores,
            device=str(self._device),
            dtype="float16" if self._is_fp16 else "float32",
            model_name=self._model_id,
            inference_time_ms=latency_ms,
            metadata={
                "threshold": threshold,
                "ai_class_index": self.ai_class_idx,
                "real_class_index": self.real_class_idx,
                "raw_logits": logits.squeeze(0).cpu().tolist(),
                "calibration_status": "uncalibrated_softmax",
            },
        )

    def predict_batch(
        self,
        images: List[Union[np.ndarray, Image.Image]],
        threshold: float = 0.50,
    ) -> List[DetectionResult]:
        """Runs batched inference for video frames."""
        if not images:
            return []

        pil_images = [
            Image.fromarray(img) if isinstance(img, np.ndarray) else img
            for img in images
        ]

        t0 = time.perf_counter()
        inputs = self.processor(images=pil_images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device=self._device)
        if self._is_fp16 and self._device.type == "cuda":
            pixel_values = pixel_values.half()

        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)
            logits = outputs.logits.float()
            probs = F.softmax(logits, dim=-1)

        batch_latency = (time.perf_counter() - t0) * 1000.0
        per_item_latency = batch_latency / len(images)

        results = []
        for i in range(len(images)):
            p = probs[i]
            ai_prob = float(p[self.ai_class_idx].item())
            real_prob = float(p[self.real_class_idx].item())
            is_ai = bool(ai_prob >= threshold)
            confidence = float(max(ai_prob, real_prob))
            predicted_label = "AI-Generated" if is_ai else "Real / Human"

            raw_scores = {
                self.raw_id2label.get(self.ai_class_idx, "artificial"): ai_prob,
                self.raw_id2label.get(self.real_class_idx, "human"): real_prob,
            }

            results.append(
                DetectionResult(
                    is_ai=is_ai,
                    ai_probability=ai_prob,
                    confidence=confidence,
                    predicted_label=predicted_label,
                    raw_scores=raw_scores,
                    device=str(self._device),
                    dtype="float16" if self._is_fp16 else "float32",
                    model_name=self._model_id,
                    inference_time_ms=per_item_latency,
                    metadata={"batch_index": i},
                )
            )

        return results

    def get_spatial_heatmap(
        self,
        image: Union[np.ndarray, Image.Image],
        target_class_idx: Optional[int] = None,
        method: str = "gradcam",
    ) -> np.ndarray:
        """
        Generates 2D spatial heatmap in [0.0, 1.0].
        Supports:
          1. PyTorch-GradCAM with Swin reshape transform.
          2. Activation Norm fallback: L2 norm across feature channels of final layer.
        """
        if target_class_idx is None:
            target_class_idx = self.ai_class_idx

        # Target size for output heatmap
        if isinstance(image, np.ndarray):
            h_orig, w_orig = image.shape[:2]
        else:
            w_orig, h_orig = image.size

        # Method 1: Try PyTorch-GradCAM
        if method.lower() == "gradcam" and len(self._target_layers) > 0:
            try:
                from pytorch_grad_cam import GradCAM
                from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

                pixel_values = self._preprocess(image)
                # GradCAM requires float32 gradients
                pixel_values = pixel_values.float().requires_grad_(True)
                fp32_model = self.model.float()
                cam_wrapper = HFModelGradCAMWrapper(fp32_model)

                cam = GradCAM(
                    model=cam_wrapper,
                    target_layers=self._target_layers,
                    reshape_transform=swin_reshape_transform if hasattr(self.model, "swin") else None,
                )

                targets = [ClassifierOutputTarget(target_class_idx)]
                grayscale_cam = cam(input_tensor=pixel_values, targets=targets)[0]

                # Check if GradCAM returned non-trivial heatmap
                if grayscale_cam.max() > 0.05:
                    import cv2
                    resized_cam = cv2.resize(grayscale_cam, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
                    return np.clip(resized_cam, 0.0, 1.0)
            except Exception as e:
                print(f"[GUARD HFDetector] GradCAM encountered exception: {e}. Falling back to spatial activation norms.")

        # Method 2: Spatial Activation Norm Fallback (Differentiable & Robust)
        return self._extract_activation_norm_heatmap(image, h_orig, w_orig)

    def _extract_activation_norm_heatmap(
        self,
        image: Union[np.ndarray, Image.Image],
        target_h: int,
        target_w: int,
    ) -> np.ndarray:
        """
        Extracts spatial activation norm map from the last feature stage.
        Guaranteed to produce a continuous spatial attention map even when gradients are zero.
        """
        import cv2

        pixel_values = self._preprocess(image)
        features_cache = []

        def capture_hook(module, inp, outp):
            features_cache.append(outp)

        target_layer = self._target_layers[0] if self._target_layers else self.model
        handle = target_layer.register_forward_hook(capture_hook)

        with torch.no_grad():
            _ = self.model(pixel_values=pixel_values)

        handle.remove()

        if not features_cache:
            # Fallback uniform
            return np.ones((target_h, target_w), dtype=np.float32) * 0.5

        feat = features_cache[0]
        if hasattr(feat, "last_hidden_state"):
            feat = feat.last_hidden_state

        if len(feat.shape) == 3:
            # (B, N, C) -> (B, C, H, W)
            feat = swin_reshape_transform(feat)

        if len(feat.shape) == 4:
            # L2 norm across channel dimension C
            norm_map = torch.norm(feat.float(), p=2, dim=1).squeeze(0).cpu().numpy()
            norm_map = (norm_map - norm_map.min()) / (norm_map.max() - norm_map.min() + 1e-8)
            resized = cv2.resize(norm_map, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            return np.clip(resized, 0.0, 1.0)

        return np.ones((target_h, target_w), dtype=np.float32) * 0.5

    @property
    def model_name(self) -> str:
        return self._model_id

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def is_fp16(self) -> bool:
        return self._is_fp16
