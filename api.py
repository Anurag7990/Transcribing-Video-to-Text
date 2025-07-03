from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
import subprocess
import re
import yt_dlp
import glob

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://video-to-subtitle.vercel.app/"
    ],  # Your Next.js URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the paths relative to the transcription project directory
OUTPUT_PATH = "output"
DOWNLOADS_PATH = "downloads"

class YouTubeRequest(BaseModel):
    url: str
    method: str = "whisper"

class CleanupRequest(BaseModel):
    videoId: str = None
    filename: str = None
    action: str = "delete_after_copy"

@app.get("/test/")
async def test():
    return {"message": "FastAPI is working!", "transcription": "This is a test transcription"}

@app.post("/cleanup-downloads/")
async def cleanup_downloads(request: CleanupRequest):
    """
    Clean up downloaded videos and temporary files
    """
    try:
        deleted_files = []
        freed_space = 0
        
        # Ensure downloads directory exists
        if not os.path.exists(DOWNLOADS_PATH):
            return {
                "success": True,
                "message": "Downloads directory does not exist",
                "deletedFiles": [],
                "freedSpace": 0
            }
        
        # Delete specific file if filename is provided
        if request.filename:
            file_path = os.path.join(DOWNLOADS_PATH, request.filename)
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                os.remove(file_path)
                deleted_files.append(request.filename)
                freed_space += file_size
                print(f"Deleted specific file: {request.filename}")
        
        # Delete all video files in downloads directory
        video_extensions = ['*.mp4', '*.mkv', '*.avi', '*.mov', '*.webm']
        for ext in video_extensions:
            pattern = os.path.join(DOWNLOADS_PATH, ext)
            for file_path in glob.glob(pattern):
                try:
                    file_size = os.path.getsize(file_path)
                    filename = os.path.basename(file_path)
                    os.remove(file_path)
                    deleted_files.append(filename)
                    freed_space += file_size
                    print(f"Deleted video file: {filename}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
        
        # Also clean up any temporary files
        temp_extensions = ['*.tmp', '*.temp', '*.part']
        for ext in temp_extensions:
            pattern = os.path.join(DOWNLOADS_PATH, ext)
            for file_path in glob.glob(pattern):
                try:
                    file_size = os.path.getsize(file_path)
                    filename = os.path.basename(file_path)
                    os.remove(file_path)
                    deleted_files.append(filename)
                    freed_space += file_size
                    print(f"Deleted temp file: {filename}")
                except Exception as e:
                    print(f"Error deleting temp file {file_path}: {e}")
        
        return {
            "success": True,
            "message": f"Cleanup completed. Deleted {len(deleted_files)} files.",
            "deletedFiles": deleted_files,
            "freedSpace": freed_space
        }
        
    except Exception as e:
        print(f"Cleanup error: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/transcribe/")
async def transcribe(
    file: UploadFile = File(...),
    method: str = Form("whisper")  # whisper ya wav2vec
):
    print(f"Received file: {file.filename}, method: {method}")
    
    # Upload folder ensure karo
    os.makedirs("uploads", exist_ok=True)

    # File save karo temporary folder me
    input_path = os.path.join("uploads", file.filename)
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # transcription method check karo
    if method not in ["whisper", "wav2vec"]:
        return {"error": "Invalid method. Use 'whisper' or 'wav2vec'."}

    try:
        # Step-by-step pipeline like your bash script
        subprocess.run(["python", "extract_audio.py", input_path], check=True)
        subprocess.run(["python", "preprocess_audio.py"], check=True)
        subprocess.run(["python", "noise_removal.py"], check=True)
        subprocess.run(["python", "main.py", method], check=True)  # your original main.py

        # Output padho from the correct location
        output_path = os.path.join(OUTPUT_PATH, f"{method}_transcript.txt")
        if not os.path.exists(output_path):
            return {"error": f"Transcription output not found at {output_path}"}

        with open(output_path, "r") as f:
            transcription = f.read()

        print(f"Transcription result: {transcription[:100]}...")  # Print first 100 chars

        result = {
            "transcription": transcription,
            "method": method,
            "filename": file.filename
        }
        
        print(f"Returning result: {result}")
        return result

    except subprocess.CalledProcessError as e:
        print(f"Subprocess error: {e}")
        return {"error": f"Processing failed: {e}"}
    except Exception as e:
        print(f"Unexpected error: {e}")
        return {"error": f"Unexpected error: {e}"}

@app.post("/transcribe-youtube/")
async def transcribe_youtube(request: YouTubeRequest):
    # Validate YouTube URL
    youtube_regex = r'^(https?:\/\/)?(www\.)?(youtube\.com|youtu\.be)\/.+'
    if not re.match(youtube_regex, request.url):
        return {"error": "Invalid YouTube URL"}

    # transcription method check karo
    if request.method not in ["whisper", "wav2vec"]:
        return {"error": "Invalid method. Use 'whisper' or 'wav2vec'."}

    try:
        # Create downloads directory
        os.makedirs(DOWNLOADS_PATH, exist_ok=True)
        
        # YouTube download configuration - LOW QUALITY for faster processing
        ydl_opts = {
            'outtmpl': f'./{DOWNLOADS_PATH}/%(title)s.%(ext)s',
            'format': 'worst[height<=480]/worst',  # Download lowest quality available, max 480p
            'postprocessors': [{
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4',
            }],
        }
        
        print(f"Analysing Youtube Url: {request.url}")
        
        # Download the video
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Get video info first
            info = ydl.extract_info(request.url, download=False)
            video_title = info.get('title', 'unknown_title')
            video_ext = info.get('ext', 'mp4')
            
            # Download the video
            ydl.download([request.url])
        
        # Find the downloaded file
        downloaded_file = None
        original_filename = None
        for filename in os.listdir(DOWNLOADS_PATH):
            if filename.endswith(video_ext) and video_title.lower() in filename.lower():
                downloaded_file = os.path.join(DOWNLOADS_PATH, filename)
                original_filename = filename
                break
        
        if not downloaded_file:
            # Fallback: find any video file in downloads
            for filename in os.listdir(DOWNLOADS_PATH):
                if filename.endswith(('.mp4', '.mkv', '.avi', '.mov')):
                    downloaded_file = os.path.join(DOWNLOADS_PATH, filename)
                    original_filename = filename
                    break
        
        if not downloaded_file:
            return {"error": "Failed to find downloaded video file"}
        
        print(f"Downloaded video: {downloaded_file}")
        
        # Copy to uploads directory for processing
        uploads_dir = "uploads"
        os.makedirs(uploads_dir, exist_ok=True)
        
        # Create a clean filename
        import time

        timestamp = int(time.time())
        clean_filename = f"youtube_{video_title[:40]}_{timestamp}.{video_ext}".replace(' ', '_').replace('/', '_')
        input_path = os.path.join(uploads_dir, clean_filename)
        
        shutil.copy2(downloaded_file, input_path)
        print(f"Copied to: {input_path}")
        
        # Step-by-step pipeline with progress updates
        print("Starting transcription pipeline...")
        
        # Step 1: Extract audio
        print("Step 1: Extracting audio from video...")
        subprocess.run(["python", "extract_audio.py", input_path], check=True)
        
        # Step 2: Preprocess audio
        print("Step 2: Preprocessing audio (noise reduction, normalization)...")
        subprocess.run(["python", "preprocess_audio.py"], check=True)
        
        # Step 3: Noise removal
        print("Step 3: Advanced noise cancellation...")
        subprocess.run(["python", "noise_removal.py"], check=True)
        
        # Step 4: Transcription
        print(f"Step 4: Transcribing using {request.method} model...")
        subprocess.run(["python", "main.py", request.method], check=True)

        # Output padho from the correct location
        output_path = os.path.join(OUTPUT_PATH, f"{request.method}_transcript.txt")
        if not os.path.exists(output_path):
            return {"error": f"Transcription output not found at {output_path}"}

        with open(output_path, "r") as f:
            transcription = f.read()

        print(f"YouTube transcription result: {transcription[:100]}...")

        # Clean up downloaded video after successful processing
        cleanup_result = None
        try:
            if original_filename:
                cleanup_response = await cleanup_downloads(CleanupRequest(
                    filename=original_filename,
                    action="delete_after_copy"
                ))
                cleanup_result = cleanup_response
                print(f"Cleanup result: {cleanup_result}")
        except Exception as cleanup_error:
            print(f"Cleanup failed: {cleanup_error}")
            cleanup_result = {"success": False, "error": str(cleanup_error)}

        result = {
            "transcription": transcription,
            "method": request.method,
            "url": request.url,
            "video_title": video_title,
            "filename": clean_filename,
            "original_filename": original_filename,
            "cleanup_result": cleanup_result
        }
        
        print(f"Returning YouTube result: {result}")
        return result

    except Exception as e:
        print(f"YouTube transcription error: {e}")
        return {"error": f"Failed to transcribe YouTube video: {str(e)}"} 