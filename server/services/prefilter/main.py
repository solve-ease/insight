from server.services.fileDB.main import fileDB
# from server.services.fileDB.main import check_files
from pathlib import Path
import os
from PIL import Image
import logging
import asyncio
logger = logging.getLogger(__name__)

folder_path = os.getenv("FOLDER PATH", "/home/rogue/Downloads")

db = fileDB()

def get_files():
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif', '.webp', '.ico', ".mp4", ".avi", ".mov", ".mkv"}
    image_files = []

    validCnt = 0
    invalidCnt = 0

    # Recursively walk through all subdirectories
    for file in Path(folder_path).rglob('*'):
        if file.is_file() and file.suffix.lower() in image_extensions:
            # Verify it's actually an image by trying to open it
            try:
                # with Image.open(file) as img:
                #     img.verify()
                validCnt += 1
                image_files.append(str(file))

            except Exception:
                # Skip files that can't be opened as images
                logger.debug(f"Skipping invalid media file: {file.name}")
                invalidCnt += 1
    
    return (image_files, validCnt, invalidCnt)


async def prefilter(rescan_start_time):
    logger.info("Running Prefilter")

    await db.connect()

    image_files , validCnt, invalidCnt = get_files()

    logger.info(f"Read {validCnt} media files")

    file_list = list()

    # store the mtime and size of the files in a dictionary
    for idx , i in enumerate(image_files):
        stat = os.stat(i)
        mtime = int(stat.st_mtime)
        size = stat.st_size

        file_list.append((i,mtime , size))
    
    prefilter_res = await db.check_files(file_list, rescan_start_time)

    logger.info(f"Prefilter Eliminated {len(image_files) - len(prefilter_res)}")

    return prefilter_res, image_files