import streamlit as st

# Set page config - MUST be first Streamlit command
st.set_page_config(
    page_title="AI Video Generator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Now import other modules
import os
from groq import Groq
from search_index import search_index
import json
from dotenv import load_dotenv
import numpy as np
from duckduckgo_search import DDGS
from moviepy import ImageClip, concatenate_videoclips, TextClip, CompositeVideoClip, ColorClip, AudioFileClip
import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import tempfile
from gtts import gTTS
import subprocess
import time
import threading
from queue import Queue

# Load environment variables from .env file
load_dotenv()

# Initialize Groq client
@st.cache_resource
def init_groq_client():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))

client = init_groq_client()

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .progress-container {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .step-completed {
        color: #28a745;
        font-weight: bold;
    }
    .step-current {
        color: #007bff;
        font-weight: bold;
    }
    .step-pending {
        color: #6c757d;
    }
</style>
""", unsafe_allow_html=True)

def download_image(url):
    """Download image from URL and save to temporary file"""
    try:
        response = requests.get(url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        if response.status_code == 200:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            temp_file.write(response.content)
            temp_file.close()
            return temp_file.name
        else:
            st.warning(f"Failed to download image. Status code: {response.status_code}")
    except Exception as e:
        st.warning(f"Error downloading image: {str(e)}")
    return None

def generate_audio_for_sentence(sentence, output_path):
    """Generate audio for a sentence using gTTS"""
    try:
        tts = gTTS(text=sentence, lang='en', slow=False)
        tts.save(output_path)
        
        # Optional: Adjust audio speed/quality with ffmpeg if available
        try:
            temp_output = output_path + ".temp.mp3"
            subprocess.run([
                'ffmpeg', '-y', '-i', output_path,
                '-filter:a', 'atempo=1.0',
                '-vn', temp_output
            ], check=True, capture_output=True)
            os.replace(temp_output, output_path)
        except (subprocess.CalledProcessError, FileNotFoundError):
            # If ffmpeg is not available or fails, use original file
            pass
        
        return True
    except Exception as e:
        st.error(f"Error generating audio: {str(e)}")
        return False

def create_text_image(text, image_size, font_size=24):
    """Create a text overlay image using PIL instead of MoviePy TextClip"""
    try:
        # Create a transparent image for text
        text_img = Image.new('RGBA', image_size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(text_img)
        
        # Try to use a system font, fall back to default if not available
        try:
            # Try common system fonts
            font_paths = [
                "/System/Library/Fonts/Arial.ttf",  # macOS
                "/Windows/Fonts/arial.ttf",         # Windows
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
                "/usr/share/fonts/TTF/arial.ttf",   # Some Linux distributions
                "/System/Library/Fonts/Helvetica.ttc",  # macOS alternative
                "/Windows/Fonts/calibri.ttf",       # Windows alternative
            ]
            
            font = None
            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        font = ImageFont.truetype(font_path, font_size)
                        break
                    except Exception:
                        continue
            
            if font is None:
                font = ImageFont.load_default()
                
        except Exception:
            font = ImageFont.load_default()
        
        # Wrap text to fit image width
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            try:
                bbox = draw.textbbox((0, 0), test_line, font=font)
                text_width = bbox[2] - bbox[0]
            except:
                # Fallback for older PIL versions
                text_width = len(test_line) * (font_size * 0.6)
            
            if text_width <= image_size[0] - 40:  # Leave 20px margin on each side
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        # Calculate total text height
        line_height = font_size + 5
        total_text_height = len(lines) * line_height
        
        # Position text at bottom of image
        start_y = max(10, image_size[1] - total_text_height - 20)
        
        # Draw background rectangle
        bg_y = start_y - 10
        bg_height = total_text_height + 20
        draw.rectangle([(0, bg_y), (image_size[0], bg_y + bg_height)], 
                      fill=(0, 0, 0, 180))  # Semi-transparent black
        
        # Draw text lines
        for i, line in enumerate(lines):
            try:
                bbox = draw.textbbox((0, 0), line, font=font)
                text_width = bbox[2] - bbox[0]
            except:
                # Fallback for older PIL versions
                text_width = len(line) * (font_size * 0.6)
            
            x = max(20, (image_size[0] - text_width) // 2)  # Center text with minimum margin
            y = start_y + i * line_height
            
            # Draw text with outline for better visibility
            outline_width = 1
            for dx in [-outline_width, 0, outline_width]:
                for dy in [-outline_width, 0, outline_width]:
                    if dx != 0 or dy != 0:
                        draw.text((x + dx, y + dy), line, font=font, fill=(0, 0, 0, 255))
            
            draw.text((x, y), line, font=font, fill=(255, 255, 255, 255))
        
        # Save text image to temporary file
        temp_text_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        text_img.save(temp_text_file.name, 'PNG')
        temp_text_file.close()
        
        return temp_text_file.name
        
    except Exception as e:
        st.error(f"Error creating text image: {str(e)}")
        return None

def get_audio_duration(audio_path):
    """Get duration of audio file"""
    try:
        audio_clip = AudioFileClip(audio_path)
        duration = audio_clip.duration
        audio_clip.close()
        return duration
    except Exception as e:
        st.warning(f"Error getting audio duration: {str(e)}")
        return 5.0  # Default to 5 seconds

def create_video_with_progress(script_with_images, output_path="output_video.mp4", default_duration=5, progress_callback=None):
    """Create video from images and add subtitles with audio using moviepy with progress tracking"""
    try:
        if not script_with_images or not any(item.get('image') for item in script_with_images):
            if progress_callback:
                progress_callback("error", "No valid images found for video creation")
            return False

        temp_dir = tempfile.mkdtemp()
        audio_files = []
        video_clips = []
        text_image_files = []

        total_steps = len(script_with_images) * 2 + 2  # Audio generation + video clips + concatenation + final output
        current_step = 0

        # Generate all audio files first
        if progress_callback:
            progress_callback("progress", f"Generating audio for {len(script_with_images)} sentences...", current_step / total_steps)
        
        for i, item in enumerate(script_with_images):
            if item.get('sentence'):
                audio_path = os.path.join(temp_dir, f"audio_{i}.mp3")
                if generate_audio_for_sentence(item['sentence'], audio_path):
                    audio_files.append(audio_path)
                    if progress_callback:
                        progress_callback("info", f"Generated audio for sentence {i+1}/{len(script_with_images)}")
                else:
                    audio_files.append(None)
            else:
                audio_files.append(None)
            
            current_step += 1
            if progress_callback:
                progress_callback("progress", f"Audio generation progress: {i+1}/{len(script_with_images)}", current_step / total_steps)

        # Process each image
        if progress_callback:
            progress_callback("progress", "Creating video clips...", current_step / total_steps)
        
        for i, item in enumerate(script_with_images):
            if not item.get('image'):
                current_step += 1
                continue
                
            image_path = download_image(item['image'])
            if not image_path:
                current_step += 1
                continue
                
            try:
                # Determine clip duration based on audio if available
                if i < len(audio_files) and audio_files[i] and os.path.exists(audio_files[i]):
                    clip_duration = get_audio_duration(audio_files[i])
                else:
                    clip_duration = default_duration
                
                # Create image clip
                img_clip = ImageClip(image_path, duration=clip_duration)
                
                # Get image dimensions for text overlay
                with Image.open(image_path) as img:
                    img_size = img.size
                
                # Create text overlay using PIL
                sentence_text = str(item['sentence']).strip()
                if sentence_text and len(sentence_text) > 0:
                    # Limit sentence length
                    if len(sentence_text) > 150:
                        sentence_text = sentence_text[:147] + "..."
                    
                    text_image_path = create_text_image(sentence_text, img_size)
                    if text_image_path:
                        text_image_files.append(text_image_path)
                        
                        # Create text clip from the generated image
                        text_clip = ImageClip(text_image_path, duration=clip_duration)
                        
                        # Composite image and text
                        video_clip = CompositeVideoClip([img_clip, text_clip])
                    else:
                        video_clip = img_clip
                else:
                    video_clip = img_clip
                
                # Add audio if available
                if i < len(audio_files) and audio_files[i] and os.path.exists(audio_files[i]):
                    try:
                        audio_clip = AudioFileClip(audio_files[i])
                        video_clip.audio = audio_clip
                    except Exception as audio_error:
                        if progress_callback:
                            progress_callback("warning", f"Could not add audio: {audio_error}")
                
                video_clips.append(video_clip)
                
                # Clean up downloaded image
                os.unlink(image_path)
                
            except Exception as clip_error:
                if progress_callback:
                    progress_callback("error", f"Error processing clip {i+1}: {clip_error}")
                if os.path.exists(image_path):
                    os.unlink(image_path)
            
            current_step += 1
            if progress_callback:
                progress_callback("progress", f"Created clip {i+1}/{len(script_with_images)}", current_step / total_steps)

        # Concatenate all video clips
        if video_clips:
            if progress_callback:
                progress_callback("progress", f"Concatenating {len(video_clips)} video clips...", current_step / total_steps)
            
            final_clip = concatenate_videoclips(video_clips, method="compose")
            
            current_step += 1
            if progress_callback:
                progress_callback("progress", "Writing final video file...", current_step / total_steps)
            
            # Write the final video file
            final_clip.write_videofile(
                output_path,
                fps=24,
                codec='libx264',
                audio_codec='aac',
                threads=4,
                preset='medium',
                bitrate='2000k',
                # verbose=False,
                logger=None
            )
            
            # Clean up
            final_clip.close()
            for clip in video_clips:
                clip.close()
            
            total_duration = sum(clip.duration for clip in video_clips)
            if progress_callback:
                progress_callback("success", f"Video created successfully! Duration: {total_duration:.2f} seconds")
            
        else:
            if progress_callback:
                progress_callback("error", "No valid video clips were created")
            return False
        
        # Clean up temporary files
        for file in audio_files:
            if file and os.path.exists(file):
                os.unlink(file)
        
        for file in text_image_files:
            if os.path.exists(file):
                os.unlink(file)
        
        try:
            os.rmdir(temp_dir)
        except OSError:
            pass  # Directory might not be empty
        
        return True
        
    except Exception as e:
        if progress_callback:
            progress_callback("error", f"Error creating video: {str(e)}")
        return False

def extract_key_context(query):
    """Extract the most important word or phrase from the user's query"""
    prompt = f"""Extract the single most important word or short phrase from this query that would help in finding relevant images.
    This should be the main subject or topic that would be most visually distinctive.
    
    Query: {query}
    
    Most important word/phrase:"""
    
    completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that extracts key context from queries."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.3,
        max_tokens=50
    )
    
    return completion.choices[0].message.content.strip()

def format_query_with_groq(user_input, purpose="image_search", context=""):
    """Format the user's query using Groq"""
    if purpose == "image_search":
        prompt = f"""Given the following text and context, create multiple image search queries.
        Create 3 different search queries:
        1. A specific, detailed query
        2. A broader query
        3. A simple, keyword-based query
        
        Context: {context}
        Text: {user_input}
        
        Return the queries in this format:
        specific: [specific query]
        broad: [broad query]
        simple: [simple query]"""
    else:
        prompt = f"""Format this query into a clear search query:
        User query: {user_input}
        Formatted search query:"""
    
    completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that formats queries."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.3,
        max_tokens=200
    )
    
    if purpose == "image_search":
        response = completion.choices[0].message.content.strip()
        queries = {}
        for line in response.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                queries[key.strip()] = value.strip()
        return queries
    else:
        return completion.choices[0].message.content.strip()

def search_duckduckgo_images(query, context=""):
    """Search for images using DuckDuckGo"""
    if not query or not query.strip():
        return None
        
    try:
        with DDGS() as ddgs:
            query_variations = format_query_with_groq(query, purpose="image_search", context=context)
            
            # Ensure we have at least one valid query
            if not query_variations or not any(query_variations.values()):
                query_variations = {'default': query}
            
            search_params = [
                {'keywords': query, 'max_results': 5, 'safesearch': 'on', 'size': 'Large', 'type_image': 'photo'},
                {'keywords': query, 'max_results': 5, 'safesearch': 'on', 'size': 'Large'},
                {'keywords': query, 'max_results': 5, 'safesearch': 'on'},
                {'keywords': query, 'max_results': 5}
            ]
            
            for params in search_params:
                try:
                    results = list(ddgs.images(**params))
                    
                    if results:
                        for result in results:
                            if result.get('image'):
                                return result['image']
                except Exception as e:
                    continue
            
            # Final fallback
            try:
                results = list(ddgs.images(keywords=f"{context} {query}", max_results=5))
                if results and results[0].get('image'):
                    return results[0]['image']
            except Exception as e:
                pass
            
            return None
            
    except Exception as e:
        return None

def find_relevant_image(sentence, context=""):
    """Find the most relevant image for a given sentence"""
    
    # Try DuckDuckGo search first
    duckduckgo_image = search_duckduckgo_images(sentence, context)
    if duckduckgo_image:
        return {
            "sentence": sentence,
            "image_path": duckduckgo_image,
            "score": None,
            "source": "duckduckgo",
            "type": "image"
        }
    
    # Try index search
    query_variations = format_query_with_groq(sentence, purpose="image_search", context=context)
    image_query = query_variations['specific'] if isinstance(query_variations, dict) and 'specific' in query_variations else sentence
    
    try:
        results = search_index(image_query, "text", top_k=10)
        
        for result in results:
            if result.get("type") in ["image", "video_frame"]:
                return {
                    "sentence": sentence,
                    "image_path": result.get("content"),
                    "score": result.get("score"),
                    "source": "index",
                    "type": result.get("type")
                }
    except Exception as e:
        pass
    
    return None

def generate_video_script(results, formatted_query):
    """Generate a concise video script from search results"""
    serializable_results = []
    for result in results:
        serializable_result = {}
        for key, value in result.items():
            if isinstance(value, np.float32):
                serializable_result[key] = float(value)
            else:
                serializable_result[key] = value
        serializable_results.append(serializable_result)
    
    prompt = f"""Create a concise and engaging video script based on these search results.
    The script should be informative and flow naturally.
    Break the content into approximately 10-12 short sentences that work well as narration.
    Each sentence should be conversational and easy to understand.
    Focus on the most important and interesting information from the results.
    
    Search Query: {formatted_query}
    
    Search Results:
    {json.dumps(serializable_results, indent=2)}
    
    Create a natural flowing script with clear, informative sentences:"""
    
    completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a professional script writer who creates engaging, informative video scripts. Write clear, conversational sentences that flow naturally without mentioning timing or duration."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.7,
        max_tokens=500
    )
    
    return completion.choices[0].message.content.strip()

# Streamlit UI
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎬 AI Video Generator</h1>
        <p>Generate engaging videos from your prompts with AI-powered scripts and images</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Video settings
    default_duration = st.sidebar.slider("Default clip duration (seconds)", 10, 60, 15)
    
    # Check for required environment variables
    if not os.getenv("GROQ_API_KEY"):
        st.error("❌ GROQ_API_KEY not found in environment variables. Please add it to your .env file.")
        return

    # Main input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_prompt = st.text_area(
            "Enter your video prompt:",
            placeholder="e.g., 'Tell me about the history of artificial intelligence' or 'Explain how solar panels work'",
            height=100
        )
    
    with col2:
        st.write("") # Spacing
        st.write("") # Spacing
        generate_button = st.button("🎬 Generate Video", type="primary", use_container_width=True)

    if generate_button and user_prompt:
        # Initialize session state for progress tracking
        if 'progress_messages' not in st.session_state:
            st.session_state.progress_messages = []
        
        # Progress container
        progress_container = st.container()
        
        with progress_container:
            st.markdown('<div class="progress-container">', unsafe_allow_html=True)
            
            # Create progress bar and status elements
            progress_bar = st.progress(0)
            status_text = st.empty()
            progress_messages = st.empty()
            
            st.markdown('</div>', unsafe_allow_html=True)

        # Progress callback function
        def update_progress(msg_type, message, progress=None):
            if progress is not None:
                progress_bar.progress(progress)
            
            # Update status
            if msg_type == "error":
                status_text.error(f"❌ {message}")
            elif msg_type == "success":
                status_text.success(f"✅ {message}")
            elif msg_type == "warning":
                status_text.warning(f"⚠️ {message}")
            elif msg_type == "info":
                status_text.info(f"ℹ️ {message}")
            else:
                status_text.info(f"🔄 {message}")
            
            # Add to message log
            st.session_state.progress_messages.append(f"{msg_type.upper()}: {message}")
            
            # Display recent messages
            if st.session_state.progress_messages:
                recent_messages = st.session_state.progress_messages[-5:]  # Show last 5 messages
                progress_messages.text("\n".join(recent_messages))

        try:
            # Step 1: Extract key context
            update_progress("progress", "Extracting key context from your prompt...", 0.1)
            key_context = extract_key_context(user_prompt)
            update_progress("info", f"Key context identified: {key_context}")

            # Step 2: Format query
            update_progress("progress", "Formatting search query...", 0.2)
            formatted_query = format_query_with_groq(user_prompt)
            if isinstance(formatted_query, dict):
                formatted_query = formatted_query.get('specific', user_prompt)
            update_progress("info", f"Search query formatted: {formatted_query}")

            # Step 3: Search index
            update_progress("progress", "Searching knowledge base...", 0.3)
            results = search_index(formatted_query, "text", top_k=10)
            
            if not results:
                update_progress("error", "No relevant information found in the knowledge base.")
                return
            
            update_progress("info", f"Found {len(results)} relevant results")

            # Step 4: Generate script
            update_progress("progress", "Generating video script...", 0.4)
            script = generate_video_script(results, formatted_query)
            update_progress("info", "Video script generated successfully")

            # Step 5: Process sentences and find images
            update_progress("progress", "Processing script sentences...", 0.5)
            sentences = [s.strip() for s in script.split('\n') if s.strip()]
            update_progress("info", f"Script broken into {len(sentences)} sentences")

            # Step 6: Find images for each sentence
            update_progress("progress", "Finding relevant images for each sentence...", 0.6)
            script_with_images = []
            
            for i, sentence in enumerate(sentences):
                update_progress("info", f"Finding image for sentence {i+1}/{len(sentences)}")
                image_result = find_relevant_image(sentence, key_context)
                if image_result:
                    script_with_images.append({
                        "sentence": sentence,
                        "image": image_result["image_path"],
                        "score": image_result["score"],
                        "source": image_result["source"]
                    })
                else:
                    script_with_images.append({
                        "sentence": sentence,
                        "image": None,
                        "score": None,
                        "source": None
                    })

            # Step 7: Create video
            update_progress("progress", "Starting video creation...", 0.7)
            
            # Generate unique filename
            output_filename = f"generated_video_{int(time.time())}.mp4"
            
            # Create video with progress tracking
            success = create_video_with_progress(
                script_with_images, 
                output_filename, 
                default_duration, 
                update_progress
            )
            
            if success and os.path.exists(output_filename):
                progress_bar.progress(1.0)
                update_progress("success", "Video generation completed successfully!")
                
                # Display final results
                st.markdown("---")
                st.header("🎉 Video Generated Successfully!")
                
                # Show video
                with open(output_filename, 'rb') as video_file:
                    video_bytes = video_file.read()
                    st.video(video_bytes)
                
                # Provide download button
                st.download_button(
                    label="📥 Download Video",
                    data=video_bytes,
                    file_name=output_filename,
                    mime="video/mp4"
                )
                
                # Show script details
                with st.expander("📝 View Generated Script & Image Sources"):
                    for i, item in enumerate(script_with_images):
                        st.write(f"**Sentence {i+1}:** {item['sentence']}")
                        if item['image']:
                            st.write(f"*Image source: {item['source']}*")
                            if item['score']:
                                st.write(f"*Relevance score: {item['score']:.4f}*")
                        else:
                            st.write("*No image found*")
                        st.write("---")
                
            else:
                update_progress("error", "Failed to generate video. Please try again.")

        except Exception as e:
            update_progress("error", f"An error occurred: {str(e)}")
            st.exception(e)

    elif generate_button and not user_prompt:
        st.warning("⚠️ Please enter a prompt to generate a video.")

    # Instructions
    with st.expander("ℹ️ How to use"):
        st.markdown("""
        1. **Enter your prompt**: Describe what you want the video to be about
        2. **Configure settings**: Adjust video clip duration in the sidebar
        3. **Generate**: Click the generate button and wait for the process to complete
        4. **Download**: Once generated, you can view and download your video
        
        **Example prompts:**
        - "Explain the water cycle"
        - "Tell me about the history of space exploration"
        - "How do electric cars work?"
        - "The benefits of renewable energy"
        """)

if __name__ == "__main__":
    main()