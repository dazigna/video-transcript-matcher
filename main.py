from typing import Any


from cv2 import VideoCapture


from pathlib import Path
import shutil
from openai import OpenAI
import httpx
import cv2
import re
import tempfile
import rich.progress
from loguru import logger

GDRIVE_URL_FORMAT: str = "https://drive.google.com/uc?export=download&id={video_id}"


# sequential single thread video download
def download_assets(video_ids: list[str], output_dir: Path) -> list[Path]:
    logger.debug(f"{output_dir=}")
    video_paths: list[Path] = []

    for id in video_ids:
        output_path: Path = output_dir / f"{id}.mp4"
        video_paths.append(output_path)
        # Check if video is already on disk otherwise download:
        if output_path.exists():
            logger.debug(f"Video {id} already exists on disk - skipping")
            continue

        url: str = GDRIVE_URL_FORMAT.format(video_id=id)
        logger.debug(f"{url=}")
        download_video(url, output_path)
    return video_paths


def download_video(url: str, output_path: Path):
    with httpx.Client(follow_redirects=True) as client:
        response = client.get(url, follow_redirects=False)

        if response.status_code == 303:
            logger.debug("Redirect detected")
            # Follow the redirect to the confirmation page
            confirm_url = response.headers["Location"]
            confirm_response = client.get(confirm_url)
            logger.debug(f"{confirm_url=}")

            # Do not extract as json as we don't know the encoding of the response and we'd have to deal with this.
            # Extract the confirmation token (if needed)
            confirm_token = None
            if "confirm=" in confirm_response.text:
                confirm_token = re.search(
                    r"confirm=([^&]+)", confirm_response.text
                ).group(1)
                download_url = f"{url}&confirm={confirm_token}"
            else:
                download_url = confirm_url

            logger.debug(f"{download_url=}")
            # Assume filename is just the file ID for now and that the video files are .mp4, but we can extract it from another API call or the SDK call if needed.
            # Step 2: Stream the file to a temporary file
            with tempfile.NamedTemporaryFile() as tmp_file:
                with client.stream("GET", download_url) as file_response:
                    total = int(file_response.headers["Content-Length"])
                    with rich.progress.Progress(
                        "[progress.percentage]{task.percentage:>3.0f}%",
                        rich.progress.BarColumn(bar_width=None),
                        rich.progress.DownloadColumn(),
                        rich.progress.TransferSpeedColumn(),
                    ) as progress:
                        download_task = progress.add_task("Download", total=total)
                        for chunk in file_response.iter_bytes():
                            tmp_file.write(chunk)
                            progress.update(download_task, advance=len(chunk))

                # Step 3: Flush and move the temporary file to the destination
                tmp_file.flush()
                shutil.move(tmp_file.name, output_path.as_posix())
                logger.debug(f"Downloaded {tmp_file.name} to {output_dir}")
        else:
            print(f"Unexpected status code: {response.status_code}")


def extract_video_frames(video_path: Path, output_path: Path, extraction_rate: int = 1):
    # OpenCV
    logger.debug(f"Extracting frames from video: {video_path}")
    logger.debug(f"Writing frames to: {output_path}")
    video_capture: VideoCapture = cv2.VideoCapture(video_path.as_posix())
    if not video_capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    try:
        frame_rate: float = video_capture.get(cv2.CAP_PROP_FPS)
        frame_interval: int = int(round(frame_rate / extraction_rate))
        logger.debug(f"Frame rate: {frame_rate}, Frame interval: {frame_interval}")
        count = 0
        frame_count = 0  # Counter for unique frame filenames
        success, image = video_capture.read()
        logger.debug(f"success: {success}")
        while success:
            if count % frame_interval == 0:
                frame_time = count / frame_rate
                # Use frame_count to avoid filename collisions
                frame_path = (
                    output_path / f"frame_{frame_count:04d}_{frame_time:.2f}.jpg"
                )
                cv2.imwrite(frame_path, image)
                frame_count += 1

            success, image = video_capture.read()
            count += 1
    finally:
        video_capture.release()


def generate_transcript(video_path: Path, api_key: str):
    client = OpenAI(api_key=api_key)


def main():
    download_sub_dir = "videos"
    frames_sub_dir = "frames"
    transcript_sub_dir = "transcripts"

    video_ids = [
        "1rbZ5j06sz21n8CzfFITqAbnRLPQOXKrH",
        # "1ifluItlqtYmL1d619M26PTKp6aOoKzkO",
        # "13VGCH9xA6YmaMCLyhp3Yxqq7xELcyZMF",
        # "1mSIV1Y0vhXnntT53NLBFyTkJlwoygdnG",
    ]

    download_dir: Path = Path.cwd() / download_sub_dir
    download_dir.mkdir(exist_ok=True)

    frames_dir: Path = Path.cwd() / frames_sub_dir
    frames_dir.mkdir(exist_ok=True)

    transcript_dir: Path = Path.cwd() / transcript_sub_dir
    transcript_dir.mkdir(exist_ok=True)

    downloaded_videos_paths: list[Path] = download_assets(
        video_ids=video_ids, output_dir=download_dir
    )

    for video_path in downloaded_videos_paths:
        extraction_dir = frames_dir / video_path.stem
        extraction_dir.mkdir(exist_ok=True)
        extract_video_frames(video_path, extraction_dir)
        generate_transcript(video_path=video_path)
    # 3. Extract transcript using openAI API
    # 4. Match transcript with video frames
    # 5. Save frames and transcript to local storage
    # 6. Handle large scale processing and long videos in the future
    # 7. Add error handling and logging
    # 8. Add unit tests
    # 9. Add documentation


if __name__ == "__main__":
    logger.info("Running video transcript matcher")
    main()
    logger.info(("Stopping video transcript matcher"))
