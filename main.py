import json
from dataclasses import dataclass, asdict
import argparse
from typing import Any
from openai.types.audio.transcription_verbose import TranscriptionVerbose
from pathlib import Path
import shutil
import tempfile
import re

import rich.progress
from openai.types.audio.transcription_verbose import TranscriptionVerbose
from cv2 import VideoCapture
from openai import OpenAI
import httpx
import cv2
from pydub import AudioSegment
from loguru import logger

GDRIVE_URL_FORMAT: str = "https://drive.google.com/uc?export=download&id={video_id}"


@dataclass
class FrameTranscriptMatch:
    frame_id: str
    text: str = ""


def download_video(url: str, output_path: Path):
    # Since Google Drive requires a confirmation step for large files, we need to handle this case.
    # We can do this by following the redirect and extracting the confirmation token from the response.
    with httpx.Client(follow_redirects=True) as client:
        response = client.get(url, follow_redirects=False)

        if response.status_code == 303:
            logger.info("Downloading video - Google Drive redirect detected ")
            # Follow the redirect to the confirmation page
            confirm_url = response.headers["Location"]
            confirm_response = client.get(confirm_url)
            logger.info(
                f"Downloading video - Google Drive confirmation url {confirm_url}"
            )

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

            logger.info(f"Downloading video - Google Drive download url {download_url}")

            # Assume filename is just the file ID for now and that the video files are .mp4, but we can extract it from another API call or the SDK call if needed.
            with tempfile.NamedTemporaryFile() as tmp_file:
                with client.stream("GET", download_url) as file_response:
                    # Simple progress bar
                    total = int(file_response.headers["Content-Length"])
                    with rich.progress.Progress(
                        "[progress.percentage]{task.percentage:>3.0f}%",
                        rich.progress.BarColumn(bar_width=None),
                        rich.progress.DownloadColumn(),
                        rich.progress.TransferSpeedColumn(),
                    ) as progress:
                        download_task = progress.add_task("Download", total=total)
                        for chunk in file_response.iter_bytes():
                            _ = tmp_file.write(chunk)
                            progress.update(download_task, advance=len(chunk))

                # Flush and move the temporary file to the destination
                tmp_file.flush()
                _ = shutil.move(tmp_file.name, output_path.as_posix())
                logger.info(
                    f"Downloading video - saved temp file {tmp_file.name} to {output_path}"
                )
        else:
            print(f"Downloading video - Unexpected status code: {response.status_code}")


def download_assets(video_ids: list[str], output_dir: Path) -> list[Path]:
    logger.info(f"Downloading videos from Google Drive to {output_dir}")
    video_paths: list[Path] = []

    for id in video_ids:
        output_path: Path = output_dir / f"{id}.mp4"
        video_paths.append(output_path)
        # Check if video is already on disk otherwise download:
        if output_path.exists():
            logger.info(f"Video {id} already exists on disk - skipping")
            continue

        url: str = GDRIVE_URL_FORMAT.format(video_id=id)
        logger.info(f"Video URL: {url}")
        download_video(url, output_path)
    return video_paths


def extract_video_frames(
    video_path: Path, output_path: Path, extraction_rate_in_sec: int = 1
) -> list[str]:
    logger.info(f"Extracting frames from video: {video_path} to {output_path}")

    video_capture: VideoCapture = cv2.VideoCapture(video_path.as_posix())
    if not video_capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frames_filename_output: list[str] = []
    try:
        frame_rate: float = video_capture.get(cv2.CAP_PROP_FPS)
        frame_interval: int = int(round(frame_rate / extraction_rate_in_sec))
        count: int = 0
        frame_count: int = 0

        success, image = video_capture.read()
        while success:
            if count % frame_interval == 0:
                frame_time_in_millisec: float = count / frame_rate * 1000
                file_output_name: str = f"frame_{frame_time_in_millisec}.jpg"
                frame_path: Path = output_path / file_output_name

                cv2.imwrite(frame_path, image)

                frames_filename_output.append(file_output_name)
                frame_count += 1

            success, image = video_capture.read()
            count += 1
    finally:
        video_capture.release()
        return frames_filename_output


def generate_transcript(
    audio_path: Path, output_path: Path, api_key: str
) -> TranscriptionVerbose:
    filename = "transcript.json"
    output_file_path = output_path / filename
    logger.info(
        f"Generating transcript for audio file {audio_path} to {output_file_path}"
    )

    client = OpenAI(api_key=api_key)
    with open(audio_path, "rb") as audio_file:
        transcript: TranscriptionVerbose = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )
        # Write transcript to file
        with open(output_file_path, "w") as f:
            json.dump(transcript.model_dump_json(), f, indent=4)

        logger.info("Generating transcript - done")
        return transcript


def extract_audio_from_video(video_path: Path, output_dir: Path) -> Path:
    output_path = output_dir / "audio.mp3"
    logger.info(f"Extracting audio from video file {video_path} to {output_path}")

    if output_path.exists():
        logger.info("Audio already exists - skipping extraction")
        return output_path

    mp4_audio = AudioSegment.from_file(video_path, "mp4")
    mp4_audio.export(output_path.as_posix(), format="mp3")
    logger.info("Extraction audio - done")
    return output_path


def match_frames_with_transcript(
    frame_timings_in_millisec: list[str],
    transcript: TranscriptionVerbose,
    output_dir: Path,
) -> Path:
    """
    Matches frames with transcript segments based on timestamps. For each frame, find the closest transcript segment.
    Format the transcript object : https://platform.openai.com/docs/api-reference/audio/createTranscription
    """

    output_file_path = output_dir / "matches.json"
    logger.info(f"Matching frames and transcript to {output_file_path}")
    if not transcript.segments:
        logger.error("No segments in transcripts, skipping matching phase")
        return []
    if len(frame_timings_in_millisec) == 0:
        logger.error("No frames in video, skipping matching phase")
        return []

    matches: list[FrameTranscriptMatch] = []

    previous_frame_timestamp_in_sec: float = 0.0

    # Find the closest transcript segment - we are computing the overlap, so segments can belong to multiple frames
    for frame in frame_timings_in_millisec:
        match: FrameTranscriptMatch = FrameTranscriptMatch(frame_id=frame)

        frame_timestamp_in_sec = int(frame.split("_")[1].split(".")[0]) / 1000
        for segment in transcript.segments:
            # Check if the segment overlaps with the current frame's time window
            if (
                segment.start < frame_timestamp_in_sec
                and segment.end > previous_frame_timestamp_in_sec
            ):
                match.text += segment.text

        matches.append(match)
        previous_frame_timestamp_in_sec = frame_timestamp_in_sec

    logger.info(f"Matched {len(matches)} frames")

    # write matches to disc
    matches_dict: list[dict[str, Any]] = [asdict(match) for match in matches]
    with open(output_file_path, "w") as f:
        json.dump(matches_dict, f, indent=4)
    logger.info("Matching - done")
    return output_file_path


def main(api_key: str):
    download_sub_dir = "videos"
    frames_sub_dir = "frames"
    transcript_sub_dir = "transcripts"
    audio_sub_dir = "audio"
    result_sub_dir = "results"

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

    audio_dir = Path.cwd() / audio_sub_dir
    audio_dir.mkdir(exist_ok=True)

    result_dir = Path.cwd() / result_sub_dir
    result_dir.mkdir(exist_ok=True)

    downloaded_videos_paths: list[Path] = download_assets(
        video_ids=video_ids, output_dir=download_dir
    )

    for video_path in downloaded_videos_paths:
        extraction_dir: Path = frames_dir / video_path.stem
        extraction_dir.mkdir(exist_ok=True)

        transcript_output_dir: Path = transcript_dir / video_path.stem
        transcript_output_dir.mkdir(exist_ok=True)
        audio_output_dir: Path = audio_dir / video_path.stem
        audio_output_dir.mkdir(exist_ok=True)

        result_output_dir: Path = result_dir / video_path.stem
        result_output_dir.mkdir(exist_ok=True)

        frames_filename_output: list[str] = extract_video_frames(
            video_path, extraction_dir
        )
        audio_path = extract_audio_from_video(video_path, audio_output_dir)
        transcript = generate_transcript(
            audio_path=audio_path, output_path=transcript_output_dir, api_key=api_key
        )
        output_file: Path = match_frames_with_transcript(
            frames_filename_output, transcript, result_output_dir
        )


if __name__ == "__main__":
    logger.info("Running video transcript matcher")

    parser = argparse.ArgumentParser(description="Run a script with an API key.")
    parser.add_argument(
        "--api-key", type=str, required=True, help="Your OpenAI API key"
    )
    args = parser.parse_args()

    main(args.api_key)

    logger.info(("Stopping video transcript matcher"))
