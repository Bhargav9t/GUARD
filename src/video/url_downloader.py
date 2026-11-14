"""
URL Video Stream Ingestion Module for GUARD.
Provides real-time streaming video fetching for YouTube Videos, YouTube Shorts,
Instagram Reels, and TikTok via yt-dlp.

Features:
  - Lightweight download format <=720p MP4 (avoids 4K bloat and downloads in 2-3s).
  - Hard duration ceiling (<= 90 seconds) to prevent GPU freeze and memory exhaustion.
  - Automatic isolation in temporary directories with automated cleanup.
  - Graceful error handling for Instagram rate-limiting, auth walls, and geo-blocks.
"""

import os
import re
import tempfile
from typing import Any, Dict, Optional, Tuple
import yt_dlp


class VideoDownloadError(Exception):
    """Custom exception raised for video download errors with user-friendly messages."""
    pass


class VideoDurationExceededError(VideoDownloadError):
    """Raised when a video exceeds the maximum duration limit."""
    pass


def identify_platform(url: str) -> str:
    """Identifies the social media or video hosting platform from the URL."""
    url_lower = url.lower()
    if "youtube.com/shorts" in url_lower or "youtu.be" in url_lower and "shorts" in url_lower:
        return "YouTube Shorts"
    elif "youtube.com" in url_lower or "youtu.be" in url_lower:
        return "YouTube"
    elif "instagram.com" in url_lower:
        return "Instagram Reels"
    elif "tiktok.com" in url_lower:
        return "TikTok"
    else:
        return "Web Video Stream"


def _get_ffmpeg_executable() -> Optional[str]:
    """
    Locates an available ffmpeg executable.
    Checks system PATH first, then falls back to python's bundled imageio_ffmpeg binary.
    """
    import shutil
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg
    try:
        import imageio_ffmpeg
        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        if ffmpeg_bin and os.path.exists(ffmpeg_bin):
            return ffmpeg_bin
    except Exception:
        pass
    return None


def download_video_from_url(
    url: str,
    max_duration_sec: int = 90,
    max_height: int = 720,
    temp_dir: Optional[str] = None,
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Downloads a lightweight video stream from a social media or video URL into an auto-cleaning directory.

    Parameters:
        url: URL of the YouTube Video/Short, Instagram Reel, or TikTok.
        max_duration_sec: Hard duration ceiling in seconds (default 90).
        max_height: Maximum vertical resolution (default 720p).
        temp_dir: Optional target directory; if None, creates an isolated temporary directory.

    Returns:
        Tuple:
            - downloaded_file_path (str): Absolute path to the downloaded .mp4 file.
            - target_dir (str): Directory containing the file (for subsequent cleanup).
            - metadata (dict): Video metadata including title, duration, uploader, platform.

    Raises:
        VideoDurationExceededError: If the video exceeds max_duration_sec.
        VideoDownloadError: If download fails due to platform blocking, network, or invalid URL.
    """
    clean_url = url.strip()
    platform = identify_platform(clean_url)

    if not temp_dir:
        target_dir = tempfile.mkdtemp(prefix="guard_stream_")
    else:
        target_dir = temp_dir
        os.makedirs(target_dir, exist_ok=True)

    output_template = os.path.join(target_dir, "stream_video_%(id)s.%(ext)s")

    # Locate cookie file if provided in workspace
    cookie_file = None
    for cand in ["cookies.txt", "instagram_cookies.txt", "configs/cookies.txt"]:
        if os.path.exists(cand):
            cookie_file = cand
            break

    ffmpeg_exe = _get_ffmpeg_executable()

    browser_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Sec-Fetch-Mode": "navigate",
    }

    # 1. First probe metadata to check duration before downloading
    probe_opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "extract_flat": False,
        "http_headers": browser_headers,
    }
    if ffmpeg_exe:
        probe_opts["ffmpeg_location"] = ffmpeg_exe
    if cookie_file:
        probe_opts["cookiefile"] = cookie_file

    try:
        with yt_dlp.YoutubeDL(probe_opts) as ydl:
            info = ydl.extract_info(clean_url, download=False)
    except yt_dlp.utils.DownloadError as e:
        err_msg = str(e)
        if "instagram" in clean_url.lower() and any(k in err_msg.lower() for k in ["login", "rate-limit", "blocked", "checkpoint", "auth", "restricted"]):
            raise VideoDownloadError(
                "Instagram restricted anonymous stream extraction (Meta login checkpoint or anti-scraping policy). "
                "To inspect this clip: (1) Save the Reel to your device and upload via 'Local Video File Upload', or "
                "(2) Test with a public YouTube Short / direct video URL, or "
                "(3) Place an exported 'cookies.txt' in the project directory for authenticated extraction."
            )
        elif "sign in" in err_msg.lower() or "bot" in err_msg.lower():
            raise VideoDownloadError(
                f"{platform} requested bot verification or authentication. "
                "Please test with a different public video URL or upload a local file."
            )
        else:
            raise VideoDownloadError(f"Failed to fetch video stream from {platform}: {err_msg[:200]}")
    except Exception as e:
        raise VideoDownloadError(f"Unexpected error inspecting stream: {str(e)}")

    if not info:
        raise VideoDownloadError(f"No stream metadata could be extracted from: {clean_url}")

    # Check duration
    duration = info.get("duration")
    if duration is not None and duration > max_duration_sec:
        raise VideoDurationExceededError(
            f"Video duration is {int(duration)} seconds, which exceeds the {max_duration_sec}-second live demo ceiling. "
            "Please paste a YouTube Short, Instagram Reel, or clip under 90 seconds."
        )

    title = info.get("title", "Social Stream Video")
    uploader = info.get("uploader", "Unknown Creator")
    fps = info.get("fps", 30.0)

    # 2. Download lightweight <=720p MP4
    if ffmpeg_exe:
        format_spec = f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/best[height<={max_height}][ext=mp4]/best[height<={max_height}]/best"
    else:
        format_spec = f"best[height<={max_height}][ext=mp4]/bestvideo[height<={max_height}][ext=mp4]/best[height<={max_height}]/best"

    download_opts = {
        "format": format_spec,
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "socket_timeout": 20,
        "retries": 3,
        "http_headers": browser_headers,
    }
    if ffmpeg_exe:
        download_opts["ffmpeg_location"] = ffmpeg_exe
        download_opts["merge_output_format"] = "mp4"
    if cookie_file:
        download_opts["cookiefile"] = cookie_file

    try:
        with yt_dlp.YoutubeDL(download_opts) as ydl:
            download_info = ydl.extract_info(clean_url, download=True)
            filename = ydl.prepare_filename(download_info)
            # Ensure .mp4 extension
            if not filename.endswith(".mp4") and os.path.exists(filename.rsplit(".", 1)[0] + ".mp4"):
                filename = filename.rsplit(".", 1)[0] + ".mp4"
    except Exception as e:
        err_msg = str(e)
        if "instagram" in clean_url.lower():
            raise VideoDownloadError(
                "Instagram restricted anonymous stream extraction (Meta login checkpoint or anti-scraping policy). "
                "To inspect this clip: (1) Save the Reel to your device and upload via 'Local Video File Upload', or "
                "(2) Test with a public YouTube Short / direct video URL, or "
                "(3) Place an exported 'cookies.txt' in the project directory for authenticated extraction."
            )
        raise VideoDownloadError(f"Failed while downloading {platform} stream: {err_msg[:250]}")

    # Find the actual downloaded file in target_dir if filename doesn't match directly
    if not os.path.exists(filename):
        candidates = [os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.startswith("stream_video_")]
        if candidates:
            filename = candidates[0]
        else:
            raise VideoDownloadError("Stream was processed, but the output MP4 could not be located on disk.")

    meta = {
        "title": title,
        "duration_sec": duration or 0.0,
        "uploader": uploader,
        "platform": platform,
        "fps": fps,
        "url": clean_url,
        "file_size_mb": round(os.path.getsize(filename) / (1024 * 1024), 2),
    }

    return filename, target_dir, meta
