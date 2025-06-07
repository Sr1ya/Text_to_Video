import os
import glob
import argparse
import shutil
import concurrent.futures
import multiprocessing
from typing import List, Tuple, Optional
import time
import logging
from dataclasses import dataclass
import json
from moviepy.editor import VideoFileClip, ImageSequenceClip, ColorClip, CompositeVideoClip, AudioFileClip, TextClip
from moviepy.video.VideoClip import VideoClip
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
@dataclass
class Config:
    GROQ_API_KEY: str = "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"  # Replace with your Groq API key
    PEXELS_API_KEY: str = "AhOk2xRCnRelEl9nG6GIqmnWnFdEsl6eBtBUl2KGr5xgGLPmPh0DOEKU"
    PIXABAY_API_KEY: str = "50585148-600c994402cc6e3c7951ef55b"
    VIDEO_WIDTH: int = 1920
    VIDEO_HEIGHT: int = 1080
    FPS: int = 30
    MAX_WORKERS: int = multiprocessing.cpu_count()
    TEMP_DIR: str = "temp_media"
    OUTPUT_DIR: str = "output_videos"
    SCRAPED_DATA_DIR: str = "scraped_data"
    CACHE_FILE: str = "media_cache.json"
    GROQ_MODEL: str = "meta-llama/llama-4-scout-17b-16e-instruct"  # Can also use "llama2-70b-4096"
    
config = Config()

# Ensure directories exist
os.makedirs(config.TEMP_DIR, exist_ok=True)
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
os.makedirs(config.SCRAPED_DATA_DIR, exist_ok=True)

# --- UTILITIES ---
def timeit(func):
    """Decorator to time functions"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.info(f"{func.__name__} executed in {end-start:.2f} seconds")
        return result
    return wrapper

def clean_temp_files():
    """Clean up temporary files"""
    if os.path.exists(config.TEMP_DIR):
        shutil.rmtree(config.TEMP_DIR)
        os.makedirs(config.TEMP_DIR)

# --- SCRIPT GENERATION WITH GROQ ---
def generate_script(topic: str, style: str = "explainer") -> str:
    """
    Generate a high-quality video script using Groq's ultra-fast LLM API
    """
    if not config.GROQ_API_KEY:
        logger.warning("Groq API key not set - using placeholder script")
        return f"Sample script about {topic}.\n\nPoint 1: First key point.\nPoint 2: Second key point.\nConclusion: Summary of main points."

    try:
        from groq import Groq
        client = Groq(api_key=config.GROQ_API_KEY)
        
        styles = {
            "explainer": "Create a concise, engaging explainer video script (150-250 words) with clear sections.",
            "documentary": "Write a documentary-style narration script (200-300 words) with factual information.",
            "promotional": "Develop a promotional video script (120-200 words) with persuasive language and calls to action.",
            "tutorial": "Compose a step-by-step tutorial script (200-300 words) with clear instructions."
        }
        
        prompt = f"""
        {styles.get(style, styles['explainer'])}
        Topic: {topic}
        
        Requirements:
        - Divide into clear sections: Introduction, 3-5 main points, Conclusion
        - Each main point should be 2-3 sentences
        - Include natural transitions between points
        - Use simple, conversational language
        - Include timing suggestions for visuals (e.g., [0:00-0:10] Introduction scene)
        - Add relevant keywords for stock footage in brackets (e.g., [keyword1, keyword2])
        - Output should be in markdown format with clear section headers
        
        Output format:
        # [Title]
        ## [Introduction]
        ## [Main Point 1] 
        ## [Main Point 2]
        ## [Main Point 3]
        ## [Conclusion]
        """
        
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional video script writer. Create engaging, well-structured video scripts."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=config.GROQ_MODEL,
            temperature=0.7,
            max_tokens=1000,
            stop=None,
        )
        
        script = chat_completion.choices[0].message.content
        logger.info("Script generated successfully with Groq API")
        return script
    except Exception as e:
        logger.error(f"Error generating script with Groq: {e}")
        return f"Sample script about {topic}.\n\nPoint 1: First key point.\nPoint 2: Second key point.\nConclusion: Summary of main points."

# --- MEDIA SEARCH & DOWNLOAD ---
class MediaCache:
    """Cache for media URLs to avoid duplicate API calls"""
    def __init__(self):
        self.cache_file = config.CACHE_FILE
        self.cache = self._load_cache()
        
    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
        
    def save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)
            
    def get(self, key):
        return self.cache.get(key, None)
        
    def set(self, key, value):
        self.cache[key] = value
        
media_cache = MediaCache()

def search_media_parallel(keywords: List[str], media_type: str = "image", max_results: int = 10) -> List[str]:
    """
    Parallel media search across multiple APIs and local files
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
        # Create futures for all search tasks
        futures = [
            executor.submit(search_pexels, keywords, media_type, max_results),
            executor.submit(search_pixabay, keywords, media_type, max_results),
            executor.submit(search_local_files, keywords, media_type, max_results)
        ]
        
        results = set()
        for future in concurrent.futures.as_completed(futures):
            try:
                media_urls = future.result()
                results.update(media_urls)
            except Exception as e:
                logger.error(f"Media search error: {e}")
                
        return list(results)[:max_results]

def search_pexels(keywords: List[str], media_type: str, max_results: int) -> List[str]:
    """Search Pexels API for media"""
    if not config.PEXELS_API_KEY:
        return []
        
    cache_key = f"pexels_{media_type}_{'_'.join(keywords)}"
    cached = media_cache.get(cache_key)
    if cached:
        return cached
        
    try:
        import requests
        headers = {"Authorization": config.PEXELS_API_KEY}
        query = " ".join(keywords)
        url = f"https://api.pexels.com/v1/search" if media_type == "image" else f"https://api.pexels.com/videos/search"
        params = {"query": query, "per_page": max_results, "size": "large"}
        
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        
        results = []
        if media_type == "image":
            for photo in data.get("photos", []):
                results.append(photo["src"]["original"])
        else:
            for video in data.get("videos", []):
                # Get the highest quality video file
                files = video.get("video_files", [])
                if files:
                    best = max(files, key=lambda f: f.get("width", 0))
                    results.append(best["link"])
                    
        media_cache.set(cache_key, results)
        return results
    except Exception as e:
        logger.error(f"Pexels search failed: {e}")
        return []

def search_pixabay(keywords: List[str], media_type: str, max_results: int) -> List[str]:
    """Search Pixabay API for media"""
    if not config.PIXABAY_API_KEY:
        return []
        
    cache_key = f"pixabay_{media_type}_{'_'.join(keywords)}"
    cached = media_cache.get(cache_key)
    if cached:
        return cached
        
    try:
        import requests
        query = "+".join(keywords)
        if media_type == "image":
            url = f"https://pixabay.com/api/?key={config.PIXABAY_API_KEY}&q={query}&image_type=photo&per_page={max_results}"
        else:
            url = f"https://pixabay.com/api/videos/?key={config.PIXABAY_API_KEY}&q={query}&per_page={max_results}"
            
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        results = []
        if media_type == "image":
            for hit in data.get("hits", []):
                results.append(hit["largeImageURL"])
        else:
            for hit in data.get("hits", []):
                videos = hit.get("videos", {})
                if "large" in videos:
                    results.append(videos["large"]["url"])
                elif "medium" in videos:
                    results.append(videos["medium"]["url"])
                    
        media_cache.set(cache_key, results)
        return results
    except Exception as e:
        logger.error(f"Pixabay search failed: {e}")
        return []

def search_local_files(keywords: List[str], media_type: str, max_results: int) -> List[str]:
    """Search local files matching keywords"""
    extensions = [".jpg", ".jpeg", ".png", ".webp"] if media_type == "image" else [".mp4", ".mov", ".avi"]
    search_dir = os.path.join(config.SCRAPED_DATA_DIR, "images" if media_type == "image" else "videos")
    
    if not os.path.exists(search_dir):
        return []
        
    matches = []
    for root, _, files in os.walk(search_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                if any(kw.lower() in file.lower() for kw in keywords):
                    matches.append(os.path.join(root, file))
                    if len(matches) >= max_results:
                        return matches
    return matches

def download_media_parallel(urls: List[str], media_type: str) -> List[str]:
    """
    Download media files in parallel
    Returns list of local file paths
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
        futures = []
        for url in urls:
            futures.append(executor.submit(download_single_media, url, media_type))
            
        results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Download failed: {e}")
                
        return results

def download_single_media(url: str, media_type: str) -> Optional[str]:
    """Download a single media file"""
    try:
        import requests
        from urllib.parse import urlparse
        
        # Create filename from URL
        parsed = urlparse(url)
        filename = os.path.basename(parsed.path)
        if not filename:
            filename = f"{int(time.time())}.{'jpg' if media_type == 'image' else 'mp4'}"
            
        local_path = os.path.join(config.TEMP_DIR, filename)
        
        # Skip if already downloaded
        if os.path.exists(local_path):
            return local_path
            
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        with requests.get(url, headers=headers, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
        logger.debug(f"Downloaded {media_type}: {url}")
        return local_path
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return None

# --- VOICEOVER GENERATION ---
def generate_voiceover(script: str, output_path: str = "voiceover.mp3") -> Optional[str]:
    """
    Generate high-quality voiceover with options for different voices
    """
    try:
        # Using gTTS as default since Groq doesn't have TTS (yet)
        from gtts import gTTS
        tts = gTTS(script, lang='en', slow=False)
        tts.save(output_path)
        return output_path
    except Exception as e:
        logger.error(f"Voiceover generation failed: {e}")
        return None

# --- VIDEO ASSEMBLY ---
def process_media_clip(media_path: str, duration: float, start_time: float) -> VideoClip:
    """
    Process a single media clip for the video with updated MoviePy API calls
    """
    try:
        if media_path.endswith(('.mp4', '.mov', '.avi')):
            clip = VideoFileClip(media_path)
            # Resize using fx method for videos
            clip = clip.resize(height=config.VIDEO_HEIGHT)
            if clip.duration > duration:
                clip = clip.subclip(0, duration)
            return clip.set_start(start_time)
        else:
            # Image with fade in/out
            clip = ImageSequenceClip([media_path], durations=[duration])
            clip = clip.resize(height=config.VIDEO_HEIGHT)
            return clip.set_start(start_time).crossfadein(0.5).crossfadeout(0.5)
    except Exception as e:
        logger.error(f"Failed to process {media_path}: {e}")
        # Return blank clip if media fails to load
        blank_clip = ColorClip(
            size=(config.VIDEO_WIDTH, config.VIDEO_HEIGHT),
            color=(0, 0, 0),
            duration=duration
        )
        blank_clip = blank_clip.set_duration(duration)
        return blank_clip.set_start(start_time)
    """
    Process a single media clip for the video
    """
    from moviepy import VideoFileClip, ImageClip, ColorClip, CompositeVideoClip, TextClip
    
    try:
        if media_path.endswith(('.mp4', '.mov', '.avi')):
            clip = VideoFileClip(media_path)
            # Resize and trim to fit duration
            clip = clip.resize((config.VIDEO_WIDTH, config.VIDEO_HEIGHT))
            if clip.duration > duration:
                clip = clip.subclip(0, duration)
            return clip.set_start(start_time)
        else:
            # Image with fade in/out
            clip = ImageClip(media_path, duration=duration)
            clip = clip.resize((config.VIDEO_WIDTH, config.VIDEO_HEIGHT))
            return clip.set_start(start_time).crossfadein(0.5).crossfadeout(0.5)
    except Exception as e:
        logger.error(f"Failed to process {media_path}: {e}")
        # Return blank clip if media fails to load
        return ColorClip(
            size=(config.VIDEO_WIDTH, config.VIDEO_HEIGHT),
            color=(0, 0, 0),
            duration=duration
        ).set_start(start_time)

def assemble_video_parallel(
    script: str,
    media_paths: List[str],
    voiceover_path: str,
    output_path: str = "output.mp4",
    style: str = "modern"
) -> bool:
    """
    Assemble video with parallel processing of clips and proper error handling
    """
    try:
        audio = AudioFileClip(voiceover_path)
        total_duration = audio.duration
        
        # Handle case where no media paths are provided
        if not media_paths:
            logger.warning("No media paths provided - creating blank video with audio")
            blank_clip = ColorClip(
                size=(config.VIDEO_WIDTH, config.VIDEO_HEIGHT),
                color=(0, 0, 0),
                duration=total_duration
            ).set_audio(audio)
            blank_clip.write_videofile(
                output_path,
                fps=config.FPS,
                codec="libx264",
                audio_codec="aac",
                threads=config.MAX_WORKERS
            )
            return True
            
        clip_durations = calculate_clip_durations(total_duration, len(media_paths))
        
        # Process clips in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
            futures = []
            start_time = 0
            for i, (media_path, duration) in enumerate(zip(media_paths, clip_durations)):
                futures.append(executor.submit(
                    process_media_clip,
                    media_path,
                    duration,
                    start_time
                ))
                start_time += duration
                
            clips: List[VideoClip] = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    clip = future.result()
                    if clip:
                        clips.append(clip)
                except Exception as e:
                    logger.error(f"Clip processing failed: {e}")
        
        # Verify we have clips to work with
        if not clips:
            logger.error("No valid clips were created - cannot assemble video")
            return False
        
        # Create final video
        final_clip = CompositeVideoClip(clips, size=(config.VIDEO_WIDTH, config.VIDEO_HEIGHT))
        final_clip = final_clip.set_audio(audio)
        
        # Add captions if requested
        if style in ["modern", "documentary"]:
            final_clip = add_captions(final_clip, script)
        
        # Write output
        final_clip.write_videofile(
            output_path,
            fps=config.FPS,
            codec="libx264",
            audio_codec="aac",
            threads=config.MAX_WORKERS,
            preset='fast',
            bitrate="8000k"
        )
        return True
        
    except Exception as e:
        logger.error(f"Video assembly failed: {e}")
        return False
    
def calculate_clip_durations(total_duration: float, num_clips: int) -> List[float]:
    """
    Calculate clip durations based on total duration and number of clips
    """
    if num_clips == 0:
        return []
        
    base_duration = total_duration / num_clips
    variations = [base_duration * 0.8, base_duration * 1.2]  # Add some variation
    durations = []
    
    remaining = total_duration
    for i in range(num_clips):
        if i == num_clips - 1:
            durations.append(remaining)
        else:
            duration = variations[i % len(variations)]
            duration = min(duration, remaining * 0.9)  # Leave some for last clip
            durations.append(duration)
            remaining -= duration
            
    return durations

def add_captions(video_clip: VideoClip, script: str) -> VideoClip:
    """
    Add styled captions to the video with proper type hints
    """
    from moviepy.config import change_settings
    
    # Set font path if needed
    try:
        change_settings({"IMAGEMAGICK_BINARY": "/usr/bin/convert"})
    except:
        pass
        
    # Split script into lines for captions
    lines = [line.strip() for line in script.split('\n') if line.strip()]
    
    # Create text clips
    text_clips: List[VideoClip] = []
    for i, line in enumerate(lines):
        txt_clip = TextClip(
            line,
            fontsize=40,
            color='white',
            font='Arial-Bold',
            stroke_color='black',
            stroke_width=1,
            size=(config.VIDEO_WIDTH * 0.9, None),
            method='caption',
            align='center'
        ).set_position(('center', 0.8), relative=True)
        
        duration = min(5, video_clip.duration / len(lines))
        txt_clip = txt_clip.set_start(i * duration).set_duration(duration)
        text_clips.append(txt_clip)
    
    return CompositeVideoClip([video_clip] + text_clips)

def validate_media_files(media_paths: List[str]) -> List[str]:
    """Validate media files exist and are readable"""
    valid_paths = []
    for path in media_paths:
        if os.path.exists(path):
            try:
                if path.endswith(('.mp4', '.mov', '.avi')):
                    with VideoFileClip(path):
                        valid_paths.append(path)
                else:
                    from PIL import Image
                    with Image.open(path) as img:
                        img.verify()
                    valid_paths.append(path)
            except Exception as e:
                logger.warning(f"Invalid media file {path}: {e}")
        else:
            logger.warning(f"Media file not found: {path}")
    return valid_paths
# --- MAIN PIPELINE ---
@timeit
def create_video(
    topic: str,
    style: str = "explainer",
    output_name: str = None,
    max_media: int = 10
) -> Optional[str]:
    """
    End-to-end video creation pipeline with parallel processing
    """
    clean_temp_files()
    
    # Step 1: Generate script with Groq
    logger.info("Generating script with Groq API...")
    script = generate_script(topic, style)
    logger.debug(f"Generated script:\n{script}")
    
    # Step 2: Extract keywords
    keywords = extract_keywords(script)
    logger.info(f"Extracted keywords: {keywords}")
    
    # Step 3: Search for media (images and videos in parallel)
    logger.info("Searching for media...")
    media_urls = search_media_parallel(keywords, "image", max_media)
    logger.info(f"Found {len(media_urls)} media items")
    
    # Step 4: Download media in parallel
    logger.info("Downloading media...")
    media_paths = download_media_parallel(media_urls, "image")
    logger.info(f"Downloaded {len(media_paths)} files")
    
    # Step 5: Generate voiceover
    logger.info("Generating voiceover...")
    voiceover_path = os.path.join(config.TEMP_DIR, "voiceover.mp3")
    if not generate_voiceover(script, voiceover_path):
        logger.error("Voiceover generation failed")
        return None
        
    # Step 6: Assemble video
    logger.info("Assembling video...")
    output_path = os.path.join(
        config.OUTPUT_DIR,
        output_name or f"{topic.replace(' ', '_')}_{int(time.time())}.mp4"
    )
    
    if not assemble_video_parallel(script, media_paths, voiceover_path, output_path, style):
        logger.error("Video assembly failed")
        return None
        
    logger.info(f"Video created successfully: {output_path}")
    return output_path

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Improved keyword extraction using TF-IDF or RAKE
    """
    try:
        # Try using RAKE for better keyword extraction
        from rake_nltk import Rake
        r = Rake()
        r.extract_keywords_from_text(text)
        keywords = [kw for kw, _ in r.get_ranked_phrases_with_scores()[:max_keywords]]
        return keywords
    except:
        # Fallback to simple method
        import re
        from collections import Counter
        words = re.findall(r'\b\w{3,}\b', text.lower())
        stopwords = set(['the', 'and', 'that', 'this', 'with', 'for', 'are', 'from'])
        filtered = [w for w in words if w not in stopwords]
        return [w for w, _ in Counter(filtered).most_common(max_keywords)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel Video Generation Pipeline with Groq")
    parser.add_argument("topic", help="Topic for the video")
    parser.add_argument("--style", choices=["explainer", "documentary", "promotional", "tutorial"], 
                       default="explainer", help="Video style")
    parser.add_argument("--output", help="Output filename")
    parser.add_argument("--max_media", type=int, default=10, help="Maximum media items to use")
    args = parser.parse_args()
    
    result = create_video(
        topic=args.topic,
        style=args.style,
        output_name=args.output,
        max_media=args.max_media
    )
    
    if not result:
        logger.error("Video creation failed")
        exit(1)
        
    logger.info(f"Successfully created video: {result}")