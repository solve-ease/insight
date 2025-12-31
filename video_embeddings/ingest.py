import os
from pathlib import Path
import logging
import asyncio
from PIL import Image

try:
    import cv2
except ImportError:
    print("OpenCV not installed. Installing opencv-python...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'opencv-python'])
    import cv2

logger = logging.getLogger(__name__)

sample_rate = int(os.getenv("VIDEO_SAMPLE_RATE", "1"))

def ingest(folder_path: str):
    try:
        logger.info(f"Starting ingestion of videos from folder: {folder_path}")
        folder = Path(folder_path)

        sampled_videos = []
        
        if not folder.exists():
            raise ValueError(f"Folder path {folder_path} does not exist.")
        
        for video in folder.rglob("*.*"):
            if video.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
                logger.info(f"Processing video: {video}")
                frames = asyncio.run(sample_video(video))
                sampled_videos.append((video, frames))

            else:
                logger.info(f"Skipping non-video file: {video}")
        
        return sampled_videos
    
    except Exception as e:
        logger.error(f"Error ingesting videos from folder {folder_path}: {e}")

async def sample_video(path: Path):
    try:
        frames = []

        path_str = str(path)
        logger.info(f"Sampling video at path: {path_str}")

        vid = cv2.VideoCapture(path_str)
        if not vid.isOpened():
            logger.error(f"Failed to open video file: {path_str}")
            return []

        frame_rate = vid.get(cv2.CAP_PROP_FPS)

        rate = int(frame_rate/sample_rate)
        
        # Ensure rate is at least 1
        if rate < 1:
            rate = 1

        await asyncio.sleep(0)  # Yield control to the event loop

        # here we can also add temporal data also in the frames, in the future for better embeddings

        frame_count = 0
        while True:
            ret, frame = vid.read()
            if not ret:
                break

            current_frame = int(vid.get(cv2.CAP_PROP_POS_FRAMES))

            if current_frame % rate == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                logger.info(f"Sampled frame {frame_count} at position {current_frame} from video {path_str}")
                frames.append(frame)
                frame_count += 1
        
        vid.release()
        logger.info(f"Sampled {len(frames)} frames from {path_str}")
        return frames
    
    except Exception as e:
        logger.error(f"Error sampling video {path_str}: {e}")
        return []    