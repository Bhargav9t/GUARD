"""
Video package exports for GUARD.
"""

from src.video.processor import (
    VideoFrameData,
    VideoAnalysisResult,
    extract_frames_from_video,
    process_video_pipeline,
    compute_laplacian_variance,
)
from src.video.timeline import (
    create_plotly_video_timeline,
    create_matplotlib_video_timeline,
)
from src.video.url_downloader import (
    download_video_from_url,
    VideoDownloadError,
    VideoDurationExceededError,
    identify_platform,
)

__all__ = [
    "VideoFrameData",
    "VideoAnalysisResult",
    "extract_frames_from_video",
    "process_video_pipeline",
    "compute_laplacian_variance",
    "create_plotly_video_timeline",
    "create_matplotlib_video_timeline",
    "download_video_from_url",
    "VideoDownloadError",
    "VideoDurationExceededError",
    "identify_platform",
]
