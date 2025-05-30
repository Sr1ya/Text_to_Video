import os
import glob
import argparse
import shutil

# For script generation (text): You can use OpenAI API, HuggingFace, or a local LLM
# For stock media: Use APIs like Pexels, Unsplash, Pixabay (requires API keys)
# For video assembly: moviepy
# For TTS: gTTS, pyttsx3, or cloud TTS

# --- CONFIGURATION ---

# Insert your API keys here
OPENAI_API_KEY = "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"
PEXELS_API_KEY = "AhOk2xRCnRelEl9nG6GIqmnWnFdEsl6eBtBUl2KGr5xgGLPmPh0DOEKU"
PIXABAY_API_KEY = "50585148-600c994402cc6e3c7951ef55b"

SCRAPED_DATA_DIR = "scraped_data"
LOCAL_IMAGE_DIR = os.path.join(SCRAPED_DATA_DIR, "images")
LOCAL_VIDEO_DIR = os.path.join(SCRAPED_DATA_DIR, "videos")
LOCAL_TEXT_DIR = os.path.join(SCRAPED_DATA_DIR, "text")
OUTPUT_VIDEO = "output_video.mp4"

# --- PIPELINE FUNCTIONS ---

def generate_script(topic):
    """
    Generate a video script for the given topic using OpenAI GPT API.
    Requires the openai package and an API key.
    """
    if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_OPENAI_API_KEY":
        print("Please set your OpenAI API key in text_to_video.py to enable script generation.")
        return f"This is a sample script about {topic}. Replace this with AI-generated content."

    try:
        import openai
        openai.api_key = OPENAI_API_KEY
        prompt = (
            f"Write a concise, engaging video script (100-200 words) about the topic: {topic}. "
            "The script should be suitable for a short explainer video, include an introduction, main points, and a conclusion."
        )
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=350,
            temperature=0.7,
        )
        script = response.choices[0].message.content.strip()
        return script
    except Exception as e:
        print(f"Error generating script: {e}")
        return f"This is a sample script about {topic}. Replace this with AI-generated content."

def search_stock_media(keywords, media_type="image", max_results=5):
    """
    Search for stock images/videos using both Pexels and Pixabay APIs.
    Returns a deduplicated list of URLs.
    """
    import requests

    results = set()
    query = "+".join(keywords)
    query_pexels = " ".join(keywords)

    # --- Pixabay ---
    if PIXABAY_API_KEY and PIXABAY_API_KEY != "YOUR_PIXABAY_API_KEY":
        try:
            if media_type == "image":
                url = f"https://pixabay.com/api/?key={PIXABAY_API_KEY}&q={query}&image_type=photo&per_page={max_results}"
                response = requests.get(url)
                data = response.json()
                for hit in data.get("hits", []):
                    results.add(hit["largeImageURL"])
            elif media_type == "video":
                url = f"https://pixabay.com/api/videos/?key={PIXABAY_API_KEY}&q={query}&per_page={max_results}"
                response = requests.get(url)
                data = response.json()
                for hit in data.get("hits", []):
                    videos = hit.get("videos", {})
                    if "large" in videos:
                        results.add(videos["large"]["url"])
                    elif "medium" in videos:
                        results.add(videos["medium"]["url"])
        except Exception as e:
            print(f"Error fetching from Pixabay: {e}")

    # --- Pexels ---
    if PEXELS_API_KEY and PEXELS_API_KEY != "YOUR_PEXELS_API_KEY":
        headers = {"Authorization": PEXELS_API_KEY}
        try:
            if media_type == "image":
                url = f"https://api.pexels.com/v1/search?query={query_pexels}&per_page={max_results}"
                response = requests.get(url, headers=headers)
                data = response.json()
                for photo in data.get("photos", []):
                    results.add(photo["src"]["original"])
            elif media_type == "video":
                url = f"https://api.pexels.com/videos/search?query={query_pexels}&per_page={max_results}"
                response = requests.get(url, headers=headers)
                data = response.json()
                for video in data.get("videos", []):
                    # Get the highest quality video file
                    files = video.get("video_files", [])
                    if files:
                        best = max(files, key=lambda f: f.get("width", 0))
                        results.add(best["link"])
        except Exception as e:
            print(f"Error fetching from Pexels: {e}")

    if not results:
        print("No stock media found. Check your API keys or try different keywords.")

    return list(results)[:max_results]

def find_local_media(keywords, media_type="image", max_results=5):
    """
    Search local scraped images/videos for files matching keywords.
    Returns a list of file paths.
    """
    directory = LOCAL_IMAGE_DIR if media_type == "image" else LOCAL_VIDEO_DIR
    files = glob.glob(os.path.join(directory, "*"))
    # Simple keyword match in filename
    matches = [f for f in files if any(kw.lower() in os.path.basename(f).lower() for kw in keywords)]
    return matches[:max_results]

def generate_voiceover(script, output_path="voiceover.mp3"):
    """
    Generate a voiceover audio file from the script using gTTS.
    """
    try:
        from gtts import gTTS
        tts = gTTS(script)
        tts.save(output_path)
        print(f"Voiceover saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error generating voiceover: {e}")
        return None

def download_images_to_temp(image_paths_or_urls):
    """
    Download all remote images to a temp directory and return local file paths.
    Local files are copied to the temp dir for uniform handling.
    Returns (list of local paths, temp dir path)
    """
    import requests
    from urllib.parse import urlparse
    import shutil
    import tempfile

    temp_dir = tempfile.mkdtemp(prefix="text2video_imgs_")
    local_paths = []
    for idx, path in enumerate(image_paths_or_urls):
        if path.startswith("http://") or path.startswith("https://"):
            try:
                resp = requests.get(path, stream=True, timeout=10)
                resp.raise_for_status()
                ext = os.path.splitext(urlparse(path).path)[-1] or ".jpg"
                local_path = os.path.join(temp_dir, f"img_{idx}{ext}")
                with open(local_path, "wb") as f:
                    for chunk in resp.iter_content(1024 * 1024):
                        f.write(chunk)
                local_paths.append(local_path)
            except Exception as e:
                print(f"Failed to download image {path}: {e}")
        else:
            # Copy local file to temp dir for uniformity
            try:
                ext = os.path.splitext(path)[-1] or ".jpg"
                local_path = os.path.join(temp_dir, f"img_{idx}{ext}")
                shutil.copy2(path, local_path)
                local_paths.append(local_path)
            except Exception as e:
                print(f"Failed to copy local image {path}: {e}")
    return local_paths, temp_dir

def assemble_video(script, images, videos, voiceover_path, output_path=OUTPUT_VIDEO):
    """
    Assemble the final video using moviepy.
    - Downloads all images to a temp directory for local access.
    - Creates a slideshow from images (if available).
    - Adds the voiceover as the audio track.
    - If no images, creates a blank video with audio.
    Cleans up temp files after use.
    """
    from moviepy import ImageClip, AudioFileClip, concatenate_videoclips, ColorClip, CompositeVideoClip

    if not voiceover_path or not os.path.exists(voiceover_path):
        print("Voiceover file not found. Cannot assemble video.")
        return

    audio_clip = AudioFileClip(voiceover_path)
    duration = audio_clip.duration

    # Download/copy all images to a temp dir
    local_images, temp_dir = download_images_to_temp(images) if images else ([], None)

    clips = []
    if local_images:
        n_images = len(local_images)
        img_duration = duration / n_images
        for img_path in local_images:
            try:
                clip = ImageClip(img_path, duration=img_duration)
                clips.append(clip)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
        if clips:
            video = concatenate_videoclips(clips, method="compose")
        else:
            video = None
    else:
        video = None

    if video is None:
        # Create a blank color video if no images
        video = ColorClip(size=(1280, 720), color=(0, 0, 0), duration=duration).with_audio(audio_clip)
    else:
        video = video.with_audio(audio_clip)

    try:
        video.write_videofile(output_path, fps=24, codec="libx264", audio_codec="aac")
        print(f"Video assembled and saved to {output_path}")
    except Exception as e:
        print(f"Error assembling video: {e}")

    # Clean up temp files
    if temp_dir and os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            print(f"Temporary files at {temp_dir} have been cleaned up.")
        except Exception as e:
            print(f"Error cleaning up temporary files: {e}")

def extract_keywords(script, max_keywords=8):
    """
    Extract keywords from the script for media search.
    Uses simple frequency analysis for better relevance.
    """
    import re
    from collections import Counter

    words = re.findall(r'\b\w+\b', script.lower())
    stopwords = set([
        "about", "this", "that", "there", "their", "which", "would", "could", "should", "where", "when", "with", "from",
        "your", "have", "will", "what", "these", "those", "because", "while", "video", "script", "topic", "using", "also"
    ])
    filtered = [w for w in words if len(w) > 3 and w not in stopwords]
    freq = Counter(filtered)
    keywords = [w for w, _ in freq.most_common(max_keywords)]
    return keywords

# --- MAIN PIPELINE ---

def main(topic):
    print(f"Generating script for topic: {topic}")
    script = generate_script(topic)
    print("Script generated.")

    keywords = extract_keywords(script)
    print(f"Extracted keywords for media search: {keywords}")

    print("Searching for stock images and videos...")
    stock_images = search_stock_media(keywords, media_type="image", max_results=5)
    stock_videos = search_stock_media(keywords, media_type="video", max_results=3)

    print("Searching for relevant local images and videos...")
    local_images = find_local_media(keywords, media_type="image", max_results=5)
    local_videos = find_local_media(keywords, media_type="video", max_results=3)

    all_images = stock_images + local_images
    all_videos = stock_videos + local_videos

    print(f"Total images: {len(all_images)}, Total videos: {len(all_videos)}")

    print("Generating voiceover...")
    voiceover_path = generate_voiceover(script)

    print("Assembling video...")
    assemble_video(script, all_images, all_videos, voiceover_path, output_path=OUTPUT_VIDEO)

    print(f"Video created: {OUTPUT_VIDEO}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text to Video Generator")
    parser.add_argument("topic", help="Topic for the video")
    args = parser.parse_args()
    main(args.topic)