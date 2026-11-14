"""
Video Processing & Temporal Flicker Analysis Module for GUARD.
Extracts video frames at 1-2 FPS via OpenCV, runs per-frame AI detection,
computes temporal flicker metrics (first-order AI confidence variance combined
with high-frequency Laplacian pixel variance), and aggregates video-level verdicts.
"""

from dataclasses import dataclass, field
import os
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from src.detectors.base import BaseDetector, DetectionResult
from src.explainability.gradcam import create_heatmap_overlay
from src.preprocessing.face_aligner import FaceAligner


@dataclass
class VideoFrameData:
    frame_idx: int
    timestamp_sec: float
    frame_rgb: np.ndarray
    ai_probability: float
    confidence: float
    is_ai: bool
    predicted_label: str
    laplacian_variance: float
    first_order_ai_diff: float          # |s_t - s_{t-1}|
    temporal_flicker_score: float       # Combined flicker metric (scene-cut filtered)
    is_anomaly: bool                    # Flagged if prob > threshold or continuous-shot flicker
    is_scene_cut: bool = False          # True if normal camera/editing jump cut
    face_detected: bool = False         # True if face extracted for targeted inspection
    heatmap_overlay_rgb: Optional[np.ndarray] = None


@dataclass
class VideoAnalysisResult:
    video_filename: str
    duration_sec: float
    total_frames_analyzed: int
    sampling_fps: float
    video_verdict: str
    is_ai: bool
    mean_ai_probability: float
    ai_frame_percentage: float
    mean_confidence: float
    peak_flicker_score: float
    mean_flicker_score: float
    frames: List[VideoFrameData]
    top_flagged_frames: List[VideoFrameData]
    anomalous_timestamp_ranges: List[Tuple[float, float]]
    processing_time_sec: float
    device: str
    model_name: str
    scene_cuts_detected: int = 0
    is_social_media_stream: bool = False
    top_k_ai_probability: float = 0.0
    faces_detected_count: int = 0


def is_scene_cut(prev_frame_rgb: np.ndarray, curr_frame_rgb: np.ndarray) -> Tuple[bool, float, float]:
    """
    Detects whether the transition between consecutive frames is a hard scene cut
    or jump cut (common in YouTube Shorts, Reels, and TikToks) rather than generative
    AI temporal morphing/flicker.

    Returns:
        Tuple:
            - is_cut (bool): True if classified as normal camera/editing cut.
            - l1_diff (float): Normalized pixel delta [0.0, 1.0].
            - hist_corr (float): HSV color histogram correlation [-1.0, 1.0].
    """
    # Downsample to 128x128 for real-time comparative processing
    p_small = cv2.resize(prev_frame_rgb, (128, 128))
    c_small = cv2.resize(curr_frame_rgb, (128, 128))

    # Pixel L1 absolute mean difference
    l1_diff = float(np.mean(np.abs(c_small.astype(np.float32) - p_small.astype(np.float32))) / 255.0)

    # HSV 2D color histogram correlation
    p_hsv = cv2.cvtColor(p_small, cv2.COLOR_RGB2HSV)
    c_hsv = cv2.cvtColor(c_small, cv2.COLOR_RGB2HSV)

    h_p = cv2.calcHist([p_hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    h_c = cv2.calcHist([c_hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(h_p, h_p, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(h_c, h_c, 0, 1, cv2.NORM_MINMAX)

    hist_corr = float(cv2.compareHist(h_p, h_c, cv2.HISTCMP_CORREL))

    # Scene cut condition: Major color histogram shift or large L1 difference
    is_cut = (l1_diff >= 0.38) or (hist_corr <= 0.45)
    return is_cut, l1_diff, hist_corr


def compute_laplacian_variance(frame_rgb: np.ndarray) -> float:
    """Computes high-frequency edge/texture variance using the Laplacian operator."""
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def extract_frames_from_video(
    video_source: Union[str, bytes],
    target_fps: float = 1.5,
    max_frames: int = 60,
) -> Tuple[List[np.ndarray], List[float], float, float]:
    """
    Extracts frames from video file or byte buffer at the specified sampling rate (1-2 FPS).

    Returns:
        Tuple:
            - frames_rgb: List of RGB numpy frames.
            - timestamps: List of timestamp floats in seconds.
            - original_fps: Native video FPS.
            - duration_sec: Total duration in seconds.
    """
    temp_file_path = None
    if isinstance(video_source, bytes):
        suffix = ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(video_source)
            temp_file_path = tmp.name
        video_path = temp_file_path
    else:
        video_path = video_source

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise ValueError(f"Could not open video file: {video_path}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_raw_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = total_raw_frames / orig_fps if orig_fps > 0 else 0.0

    step = max(1, int(round(orig_fps / target_fps)))

    frames_rgb = []
    timestamps = []
    raw_idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if raw_idx % step == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames_rgb.append(frame_rgb)
            ts = raw_idx / orig_fps
            timestamps.append(round(ts, 2))

            if len(frames_rgb) >= max_frames:
                break

        raw_idx += 1

    cap.release()
    if temp_file_path and os.path.exists(temp_file_path):
        os.remove(temp_file_path)

    return frames_rgb, timestamps, orig_fps, duration_sec


def process_video_pipeline(
    video_source: Union[str, bytes],
    detector: BaseDetector,
    video_filename: str = "uploaded_video.mp4",
    target_fps: float = 1.5,
    max_frames: int = 60,
    threshold: float = 0.50,
    is_social_media_stream: bool = False,
    progress_callback: Optional[Any] = None,
) -> VideoAnalysisResult:
    """
    Executes full video analysis pipeline:
      1. Frame extraction at 1-2 FPS.
      2. Batched AI detection.
      3. Scene-Cut Aware Temporal Flicker & Laplacian variance analysis.
         (Jump cuts and transitions in Reels/Shorts are filtered out from AI flicker score).
      4. Aggregated video-level verdict.
      5. Top 3 flagged frames with spatial heatmaps.
    """
    t_start = time.perf_counter()

    if progress_callback:
        progress_callback(0.1, "Extracting frames at 1-2 FPS using OpenCV...")

    frames_rgb, timestamps, orig_fps, duration_sec = extract_frames_from_video(
        video_source=video_source,
        target_fps=target_fps,
        max_frames=max_frames,
    )

    if not frames_rgb:
        raise ValueError("No valid video frames could be extracted.")

    if progress_callback:
        progress_callback(0.25, f"Detecting facial regions & preprocessing {len(frames_rgb)} frames...")

    # 1. Face alignment and extraction per frame (critical for facial deepfakes)
    aligner = FaceAligner()
    evaluation_images = []
    face_flags = []
    faces_detected_count = 0

    for f_rgb in frames_rgb:
        bbox = aligner.detect_primary_face(f_rgb)
        if bbox is not None:
            # Crop the face region with 25% margin padding to focus on manipulation artifacts
            x1, y1, x2, y2 = aligner.expand_bbox(bbox, f_rgb.shape, margin=0.25)
            face_crop = f_rgb[y1:y2, x1:x2]
            if face_crop.size > 0:
                face_crop_resized = cv2.resize(face_crop, (380, 380), interpolation=cv2.INTER_AREA)
                evaluation_images.append(face_crop_resized)
                face_flags.append(True)
                faces_detected_count += 1
                continue
        # Fallback to whole frame if no face is detected (e.g. generative AI scenes)
        evaluation_images.append(f_rgb)
        face_flags.append(False)

    if progress_callback:
        progress_callback(0.40, f"Executing accelerated neural forensic inference on {len(evaluation_images)} frames...")

    # Run high-throughput batched inference on all extracted regions
    detection_results = detector.predict_batch(evaluation_images, threshold=threshold)

    if progress_callback:
        progress_callback(0.65, "Computing scene-cut filtered temporal flicker & Laplacian variance...")

    # Compute Laplacian variances
    lap_vars = [compute_laplacian_variance(f) for f in frames_rgb]
    max_lap = max(lap_vars) if max(lap_vars) > 0 else 1.0

    frame_data_list: List[VideoFrameData] = []
    prev_prob = detection_results[0].ai_probability
    scene_cuts_count = 0

    for i, (f_rgb, ts, det, lap_v, has_face) in enumerate(zip(frames_rgb, timestamps, detection_results, lap_vars, face_flags)):
        curr_prob = det.ai_probability

        # Check for scene cut / editing jump cut between consecutive frames
        is_cut = False
        if i > 0:
            is_cut, l1_d, h_corr = is_scene_cut(frames_rgb[i - 1], f_rgb)
            if is_cut:
                scene_cuts_count += 1

        # 1. First-order confidence difference: |s_t - s_{t-1}|
        first_order_diff = abs(curr_prob - prev_prob) if i > 0 else 0.0

        # 2. Normalized Laplacian variance difference: |\Delta Lap_t| / max(Lap)
        prev_lap = lap_vars[i - 1] if i > 0 else lap_v
        lap_diff_norm = abs(lap_v - prev_lap) / (max_lap + 1e-6)

        # 3. Scene-Cut Aware Temporal Flicker Metric:
        if is_cut:
            # Normal video editing jump cut: exclude from AI morphing anomaly score!
            flicker_score = 0.0
            is_anomaly = bool(det.is_ai)
        else:
            # Continuous shot: frame-to-frame difference measures synthetic generative morphing/flicker
            flicker_score = first_order_diff + (0.4 * lap_diff_norm)
            is_anomaly = bool(det.is_ai or flicker_score >= 0.25)

        frame_data_list.append(
            VideoFrameData(
                frame_idx=i,
                timestamp_sec=ts,
                frame_rgb=f_rgb,
                ai_probability=curr_prob,
                confidence=det.confidence,
                is_ai=det.is_ai,
                predicted_label=det.predicted_label,
                laplacian_variance=lap_v,
                first_order_ai_diff=first_order_diff,
                temporal_flicker_score=flicker_score,
                is_anomaly=is_anomaly,
                is_scene_cut=is_cut,
                face_detected=has_face,
                heatmap_overlay_rgb=None,
            )
        )
        prev_prob = curr_prob

    if progress_callback:
        progress_callback(0.85, "Aggregating video-level metrics with forensic anomaly pooling...")

    # Aggregate video metrics
    ai_probs = [fd.ai_probability for fd in frame_data_list]
    flicker_scores = [fd.temporal_flicker_score for fd in frame_data_list]
    ai_frames_count = sum(1 for fd in frame_data_list if fd.is_ai)

    mean_prob = float(np.mean(ai_probs))
    ai_frame_pct = (ai_frames_count / len(frame_data_list)) * 100.0
    mean_conf = float(np.mean([fd.confidence for fd in frame_data_list]))
    peak_flicker = float(np.max(flicker_scores)) if flicker_scores else 0.0
    mean_flicker = float(np.mean(flicker_scores)) if flicker_scores else 0.0

    # Top-K Anomaly Pooling: Evaluates top 20% most suspicious frames
    sorted_probs = sorted(ai_probs, reverse=True)
    k_count = max(1, int(np.ceil(len(sorted_probs) * 0.20)))
    top_k_mean = float(np.mean(sorted_probs[:k_count]))

    # Video-level verdict:
    # 1. Top-k anomaly mean exceeds threshold (catches subtle or partial deepfakes)
    # 2. OR global mean exceeds threshold
    # 3. OR over 25% of frames are classified as AI
    # 4. OR high temporal flicker on continuous shots with elevated top-k probability
    is_video_ai = bool(
        (top_k_mean >= threshold) or
        (mean_prob >= threshold) or
        (ai_frame_pct >= 25.0) or
        (peak_flicker >= 0.28 and top_k_mean >= 0.38)
    )
    video_verdict = "AI-Generated / Deepfake Video" if is_video_ai else "Authentic / Real Video"

    # Identify anomalous timestamp ranges (clusters of anomalies)
    anomalous_ranges: List[Tuple[float, float]] = []
    in_range = False
    start_ts = 0.0
    last_ts = 0.0

    for fd in frame_data_list:
        if fd.is_anomaly:
            if not in_range:
                in_range = True
                start_ts = fd.timestamp_sec
            last_ts = fd.timestamp_sec
        else:
            if in_range:
                anomalous_ranges.append((start_ts, last_ts))
                in_range = False
    if in_range:
        anomalous_ranges.append((start_ts, last_ts))

    if progress_callback:
        progress_callback(0.9, "Generating spatial heatmaps for top flagged frames...")

    # Select top 3 flagged frames: prioritizing highest AI probability, then flicker
    sorted_frames = sorted(
        frame_data_list,
        key=lambda x: (x.ai_probability, x.temporal_flicker_score),
        reverse=True,
    )
    top_3_frames = sorted_frames[:3]

    # Generate spatial heatmaps for the top 3 frames
    for top_f in top_3_frames:
        try:
            hm = detector.get_spatial_heatmap(top_f.frame_rgb, method="gradcam")
            top_f.heatmap_overlay_rgb = create_heatmap_overlay(top_f.frame_rgb, hm, alpha=0.50)
        except Exception as e:
            print(f"[GUARD Video] Heatmap extraction failed for frame {top_f.frame_idx}: {e}")
            top_f.heatmap_overlay_rgb = top_f.frame_rgb

    total_proc_time = time.perf_counter() - t_start

    if progress_callback:
        progress_callback(1.0, "Video analysis complete!")

    return VideoAnalysisResult(
        video_filename=video_filename,
        duration_sec=duration_sec,
        total_frames_analyzed=len(frame_data_list),
        sampling_fps=target_fps,
        video_verdict=video_verdict,
        is_ai=is_video_ai,
        mean_ai_probability=mean_prob,
        ai_frame_percentage=ai_frame_pct,
        mean_confidence=mean_conf,
        peak_flicker_score=peak_flicker,
        mean_flicker_score=mean_flicker,
        frames=frame_data_list,
        top_flagged_frames=top_3_frames,
        anomalous_timestamp_ranges=anomalous_ranges,
        processing_time_sec=total_proc_time,
        device=str(detector.device),
        model_name=detector.model_name,
        scene_cuts_detected=scene_cuts_count,
        is_social_media_stream=is_social_media_stream,
        top_k_ai_probability=top_k_mean,
        faces_detected_count=faces_detected_count,
    )
