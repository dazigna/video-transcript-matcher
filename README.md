
# Video Transcript Matcher

A Python utility to download videos, generate transcripts using OpenAI's Whisper, and match transcript segments to corresponding video frames.

## Features

- **Video Downloading**: Downloads videos from Google Drive URLs.
- **Audio Extraction**: Extracts audio from video files in MP3 format.
- **Frame Extraction**: Extracts video frames at a specified rate (e.g., one frame per second).
- **Transcription**: Generates detailed transcripts with timestamps using OpenAI's Whisper API.
- **Frame-Transcript Matching**: Matches transcribed text segments to the corresponding video frames based on timestamps.
- **Organized Output**: Saves downloaded videos, extracted frames, audio files, transcripts, and match results in separate, organized directories.


## Requirements

- Python 3.13+
- [FFmpeg](https://ffmpeg.org/download.html) installed on your local machine.
- An active OpenAI API key.

This project uses the following Python libraries:
- `httpx`
- `loguru`
- `openai`
- `opencv-python`
- `pydub`
- `rich`


## Installation

1. **Install uv**
   If you don't have uv installed, run:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Ensure FFmpeg is installed**:
   - **Linux**: `sudo apt install ffmpeg`
   - **macOS**: `brew install ffmpeg`
   - **Windows**: Download from [FFmpeg's official site](https://ffmpeg.org/download.html).

3. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/video-transcript-matcher.git
   cd video-transcript-matcher
   ```

4. **Install the required Python packages** (preferably in a virtual environment):
   ```bash
   uv sync
   ```

## Usage

Run the main script from the command line, providing your OpenAI API key as an argument:
```bash
uv run main.py --api-key YOUR_OPENAI_API_KEY
```

The script will perform the following steps:

1. Download the videos specified in `main.py`.

2. Extract audio and frames from each video using opencv and pydub. It's not using ffmpeg directly for the sake of simplicity, but if we wanted more performance we should use ffmpeg directly and tweak our decoder accordingly.

3. Generate a transcript for each video's audio using whisper-1 instead of gpt-4o-transcribe, as we need the segments timestamps and the other model doesn't provide them, but we could use the whisper-1 model with the verbose mode to get the segments and then use gpt-4o to refine the transcript. See documentation: [https://platform.openai.com/docs/guides/speech-to-text#transcriptions](https://platform.openai.com/docs/guides/speech-to-text#transcriptions)

4. Match the transcript segments with the video frames by operating an overlap computation. Practically speaking it means that segments will be shared between frames.

5. Save the results as a json file.

**Note**: The directory structure has been made in a way that each step owns its assets and hierarchy so that we can easily parallelize the processing of each video and each step in the future.

## Future Improvements

The current implementation provides a solid foundation. Future development could focus on the following areas to enhance scalability and robustness:

### System Design for Scale

- **Parallel Processing**: Implement a task queue (e.g., Celery, RabbitMQ) or a DAG orchestrator (e.g., Flyte, Prefect, Temporal) to process multiple videos in parallel.
- **Storage and cloud Storage**: Integrate with a cloud storage solution like AWS S3 or Google Cloud Storage for storing videos, frames, and transcripts, which is more scalable than local storage. Refactor the directory handling to make it more robust and less verbose.
- **Batch Processing**: For large numbers of videos, switch to a batch processing model to optimize resource usage and API calls.
- **Monitoring**: Enhance logging with a dedicated monitoring solution (e.g., Prometheus and Grafana) to track processing times, failures, and resource usage.

### Handling Large Videos

- **Video Chunking**: For very long videos, pre-process them by splitting them into smaller chunks to enable parallel processing and reduce the impact of failures.
- **Streaming**: Utilize streaming for downloads and processing to minimize memory footprint.

### General Enhancements

- **Comprehensive Error Handling**: Implement more robust error handling, including retries for network requests and API calls.
- **Testing**: Add a suite of unit and integration tests to ensure code quality and reliability.
- **Configuration**: Move hardcoded values (like video IDs) into a separate configuration file (.env or .yaml)
- **Security**: Store any secrets in a vault/secret manager and add git hooks to scan for secrets at push time.
- **Containerization**: Provide a `Dockerfile` to simplify setup and ensure a consistent environment.