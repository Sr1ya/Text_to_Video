import os
from groq import Groq
from search_index import search_index
import json
from dotenv import load_dotenv
import numpy as np
from duckduckgo_search import DDGS
from moviepy import ImageClip, concatenate_videoclips, TextClip, CompositeVideoClip, ColorClip, AudioFileClip
import requests
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from io import BytesIO
import tempfile
from gtts import gTTS
import subprocess
import time
import cv2

# Load environment variables from .env file
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Configuration
COMPANY_LOGO_PATH = "logo.png"  # Path to your company logo
STANDARD_VIDEO_SIZE = (1920, 1080)
WATERMARK_KEYWORDS = ['vakilsearch']

def download_image(url):
    """Download image from URL and save to temporary file"""
    try:
        print(f"Attempting to download image from: {url}")
        response = requests.get(url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        if response.status_code == 200:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            temp_file.write(response.content)
            temp_file.close()
            print(f"Successfully downloaded image to: {temp_file.name}")
            return temp_file.name
        else:
            print(f"Failed to download image. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error downloading image: {str(e)}")
    return None

def has_watermark_or_logo(image_path):
    """Check if image has watermarks or unwanted logos using text detection"""
    try:
        # Load image with OpenCV
        img = cv2.imread(image_path)
        if img is None:
            return False
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to get better text detection
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Use Tesseract OCR to detect text (if available)
        try:
            import pytesseract
            text = pytesseract.image_to_string(thresh).lower()
            
            # Check for watermark keywords
            for keyword in WATERMARK_KEYWORDS:
                if keyword in text:
                    print(f"Watermark detected: {keyword}")
                    return True
        except ImportError:
            print("pytesseract not available, skipping OCR watermark detection")
        
        # Check image dimensions - very wide/tall watermarks
        height, width = gray.shape
        if width > height * 3 or height > width * 3:
            print("Suspicious aspect ratio detected")
            return True
            
        # Check for text-like patterns in corners (common watermark locations)
        corners = [
            gray[0:int(height*0.2), 0:int(width*0.3)],  # Top-left
            gray[0:int(height*0.2), int(width*0.7):],   # Top-right
            gray[int(height*0.8):, 0:int(width*0.3)],   # Bottom-left
            gray[int(height*0.8):, int(width*0.7):]     # Bottom-right
        ]
        
        for corner in corners:
            if corner.size > 0:
                # Look for high contrast areas (typical of watermarks)
                std_dev = np.std(corner)
                if std_dev > 50:  # High variation suggests text/watermarks
                    print("High contrast corner detected (possible watermark)")
                    return True
        
        return False
        
    except Exception as e:
        print(f"Error checking watermark: {str(e)}")
        return False

def resize_image_standard(image_path, target_size=STANDARD_VIDEO_SIZE):
    """Resize image to standard size with consistent scaling"""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Calculate scaling to fit within target size while maintaining aspect ratio
            img_ratio = img.width / img.height
            target_ratio = target_size[0] / target_size[1]
            
            if img_ratio > target_ratio:
                # Image is wider than target - fit by width
                new_width = target_size[0]
                new_height = int(new_width / img_ratio)
            else:
                # Image is taller than target - fit by height
                new_height = target_size[1]
                new_width = int(new_height * img_ratio)
            
            # Resize image with high quality
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Create new image with target size and black background
            final_img = Image.new('RGB', target_size, (0, 0, 0))
            
            # Center the resized image
            offset = ((target_size[0] - new_width) // 2, (target_size[1] - new_height) // 2)
            final_img.paste(resized_img, offset)
            
            # Save standardized image
            output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg').name
            final_img.save(output_path, 'JPEG', quality=95, optimize=True)
            return output_path
            
    except Exception as e:
        print(f"Error standardizing image: {str(e)}")
        return image_path

def create_animated_logo_clip(duration=5.0):
    """Create an animated logo clip with zoom-in effect"""
    try:
        if not os.path.exists(COMPANY_LOGO_PATH):
            print(f"Logo not found at {COMPANY_LOGO_PATH}, creating placeholder")
            return create_placeholder_logo_clip(duration)
        
        # Load and resize logo
        with Image.open(COMPANY_LOGO_PATH) as logo_img:
            # Convert to RGBA for transparency
            if logo_img.mode != 'RGBA':
                logo_img = logo_img.convert('RGBA')
            
            # Resize logo to reasonable size (max 400px)
            logo_ratio = logo_img.width / logo_img.height
            if logo_img.width > logo_img.height:
                new_width = min(400, logo_img.width)
                new_height = int(new_width / logo_ratio)
            else:
                new_height = min(400, logo_img.height)
                new_width = int(new_height * logo_ratio)
            
            logo_resized = logo_img.resize((new_width, new_height), Image.LANCZOS)
            
            # Save temporary logo
            temp_logo_path = tempfile.NamedTemporaryFile(delete=False, suffix='.png').name
            logo_resized.save(temp_logo_path, 'PNG')
        
        # Create background
        bg_clip = ColorClip(size=STANDARD_VIDEO_SIZE, color=(240, 240, 240), duration=duration)
        
        # Create logo clip with animation
        logo_clip = ImageClip(temp_logo_path, duration=duration)
        
        # Center the logo
        logo_clip = logo_clip.set_position('center')
        
        # Add zoom-in animation (starts small, grows to normal size)
        def zoom_effect(t):
            # Zoom from 0.3 to 1.0 over the duration
            zoom_factor = 0.3 + (0.7 * min(t / (duration * 0.6), 1))
            return zoom_factor
        
        logo_clip = logo_clip.resize(zoom_effect)
        
        # Add fade-in effect
        logo_clip = logo_clip.crossfadein(0.5)
        
        # Composite logo over background
        final_clip = CompositeVideoClip([bg_clip, logo_clip])
        
        # Clean up temp logo file
        os.unlink(temp_logo_path)
        
        return final_clip
        
    except Exception as e:
        print(f"Error creating animated logo: {str(e)}")
        return create_placeholder_logo_clip(duration)

def create_placeholder_logo_clip(duration=5.0):
    """Create a placeholder logo clip when actual logo is not available"""
    try:
        # Create a simple colored background with company name
        bg_clip = ColorClip(size=STANDARD_VIDEO_SIZE, color=(45, 55, 72), duration=duration)
        
        # Add company name text (replace with your company name)
        company_name = "YOUR COMPANY"  # Replace with actual company name
        
        txt_clip = TextClip(company_name, 
                           fontsize=80, 
                           color='white', 
                           font='Arial-Bold',
                           duration=duration)
        txt_clip = txt_clip.set_position('center')
        
        # Add animation - fade in and slight scale
        txt_clip = txt_clip.crossfadein(0.5)
        
        final_clip = CompositeVideoClip([bg_clip, txt_clip])
        return final_clip
        
    except Exception as e:
        print(f"Error creating placeholder logo: {str(e)}")
        # Return simple color clip as last resort
        return ColorClip(size=STANDARD_VIDEO_SIZE, color=(45, 55, 72), duration=duration)

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
        print(f"Error generating audio: {str(e)}")
        return False

def create_text_image(text, image_size, font_size=36):
    """Create a text overlay image with better visibility"""
    try:
        # Create a transparent image for text
        text_img = Image.new('RGBA', image_size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(text_img)
        
        # Try to use a bold system font
        try:
            font_paths = [
                "/System/Library/Fonts/Arial Bold.ttf",  # macOS
                "/Windows/Fonts/arialbd.ttf",            # Windows
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
                "/usr/share/fonts/TTF/arialbd.ttf",     # Some Linux distributions
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
        
        # Wrap text to fit image width with margins
        words = text.split()
        lines = []
        current_line = []
        max_width = image_size[0] - 120  # Larger margins for better readability
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            try:
                bbox = draw.textbbox((0, 0), test_line, font=font)
                text_width = bbox[2] - bbox[0]
            except:
                text_width = len(test_line) * (font_size * 0.6)
            
            if text_width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        # Calculate text block dimensions
        line_height = int(font_size * 1.3)
        total_text_height = len(lines) * line_height
        padding = 30
        
        # Position text block at bottom center
        text_block_width = image_size[0] - 120
        text_block_height = total_text_height + padding * 2
        text_block_y = image_size[1] - text_block_height - 60
        
        # Draw semi-transparent background with rounded corners
        bg_color = (0, 0, 0, 200)  # Darker semi-transparent
        
        # Create rounded rectangle background
        background = Image.new('RGBA', image_size, (0, 0, 0, 0))
        bg_draw = ImageDraw.Draw(background)
        
        # Draw rounded rectangle
        bg_draw.rounded_rectangle(
            [(60, text_block_y), 
             (60 + text_block_width, text_block_y + text_block_height)],
            radius=15,
            fill=bg_color
        )
        
        # Composite background
        text_img = Image.alpha_composite(text_img, background)
        draw = ImageDraw.Draw(text_img)
        
        # Draw text with white color and black outline for readability
        text_color = (255, 255, 255)
        outline_color = (0, 0, 0)
        outline_width = 2
        
        for i, line in enumerate(lines):
            try:
                bbox = draw.textbbox((0, 0), line, font=font)
                text_width = bbox[2] - bbox[0]
            except:
                text_width = len(line) * (font_size * 0.6)
            
            x = (image_size[0] - text_width) // 2
            y = text_block_y + padding + i * line_height
            
            # Draw outline
            for ox in range(-outline_width, outline_width + 1):
                for oy in range(-outline_width, outline_width + 1):
                    if ox != 0 or oy != 0:
                        draw.text((x + ox, y + oy), line, font=font, fill=outline_color)
            
            # Draw main text
            draw.text((x, y), line, font=font, fill=text_color)
        
        # Save text image
        temp_text_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        text_img.save(temp_text_file.name, 'PNG')
        temp_text_file.close()
        
        return temp_text_file.name
        
    except Exception as e:
        print(f"Error creating text image: {str(e)}")
        return None

def get_audio_duration(audio_path):
    """Get duration of audio file"""
    try:
        audio_clip = AudioFileClip(audio_path)
        duration = audio_clip.duration
        audio_clip.close()
        return duration
    except Exception as e:
        print(f"Error getting audio duration: {str(e)}")
        return 5.0  # Default to 5 seconds

def search_company_website_first(query, context=""):
    """Search company website/index first before external sources"""
    try:
        print(f"Searching company index for: {query}")
        results = search_index(query, "text", top_k=15)  # Get more results to filter
        
        if results:
            # Filter for image results
            image_results = [r for r in results if r.get("type") in ["image", "video_frame"]]
            if image_results:
                print(f"Found {len(image_results)} image results in company index")
                return image_results[0].get("content")  # Return first image
        
        print("No suitable images found in company index")
        return None
        
    except Exception as e:
        print(f"Error searching company index: {str(e)}")
        return None

def search_duckduckgo_images_filtered(query, context=""):
    """Search for images using DuckDuckGo with watermark filtering"""
    if not query or not query.strip():
        print("Error: Empty search query")
        return None
        
    try:
        with DDGS() as ddgs:
            search_params = [
                {'keywords': f"{query} high quality -watermark -stock", 'max_results': 10, 'safesearch': 'on', 'size': 'Large', 'type_image': 'photo'},
                {'keywords': f"{context} {query} -logo -watermark", 'max_results': 10, 'safesearch': 'on', 'size': 'Large'},
                {'keywords': query, 'max_results': 15, 'safesearch': 'on', 'size': 'Large'},
            ]
            
            for params in search_params:
                try:
                    print(f"Searching DuckDuckGo with: {params['keywords']}")
                    results = list(ddgs.images(**params))
                    
                    if results:
                        print(f"Found {len(results)} images, filtering for quality...")
                        for result in results:
                            image_url = result.get('image')
                            if image_url:
                                # Quick URL filtering for obvious watermarks
                                url_lower = image_url.lower()
                                if any(keyword in url_lower for keyword in WATERMARK_KEYWORDS):
                                    print(f"Skipping watermarked URL: {image_url}")
                                    continue
                                
                                # Download and check for watermarks
                                temp_path = download_image(image_url)
                                if temp_path:
                                    if not has_watermark_or_logo(temp_path):
                                        print(f"Found clean image: {image_url}")
                                        os.unlink(temp_path)  # Clean up temp file
                                        return image_url
                                    else:
                                        print(f"Image has watermark, skipping: {image_url}")
                                        os.unlink(temp_path)  # Clean up temp file
                                
                except Exception as e:
                    print(f"Error with DuckDuckGo search: {str(e)}")
                    continue
            
            print("No clean images found in DuckDuckGo")
            return None
            
    except Exception as e:
        print(f"Error in DuckDuckGo search: {str(e)}")
        return None

def find_relevant_image(sentence, context=""):
    """Find the most relevant image for a given sentence, prioritizing company sources"""
    print(f"\nSearching for image for sentence: {sentence}")
    
    # First, try company website/index
    print("Searching company sources first...")
    company_image = search_company_website_first(sentence, context)
    if company_image:
        print(f"Found image in company sources: {company_image}")
        return {
            "sentence": sentence,
            "image_path": company_image,
            "score": 1.0,  # High score for company images
            "source": "company",
            "type": "image"
        }
    
    # If no company image, try external sources with filtering
    print("Company sources failed, trying external sources...")
    external_image = search_duckduckgo_images_filtered(sentence, context)
    if external_image:
        print(f"Found clean external image: {external_image}")
        return {
            "sentence": sentence,
            "image_path": external_image,
            "score": 0.7,  # Lower score for external images
            "source": "external",
            "type": "image"
        }
    
    print("No suitable image found - will use logo animation")
    return None

def create_video(script_with_images, output_path="output_video.mp4", default_duration=5):
    """Create video ensuring all sentences are included"""
    try:
        if not script_with_images:
            print("No content to create video")
            return

        temp_dir = tempfile.mkdtemp()
        audio_files = []
        video_clips = []
        text_image_files = []
        
        print(f"\nCreating video with {len(script_with_images)} segments...")
        
        # Generate all audio files first
        for i, item in enumerate(script_with_images):
            sentence = item.get('sentence', '').strip()
            if sentence:
                audio_path = os.path.join(temp_dir, f"audio_{i}.mp3")
                if generate_audio_for_sentence(sentence, audio_path):
                    audio_files.append(audio_path)
                    print(f"Generated audio for segment {i+1}")
                else:
                    audio_files.append(None)
                    print(f"Failed to generate audio for segment {i+1}")
            else:
                audio_files.append(None)

        # Process each segment (ensure all sentences are included)
        for i, item in enumerate(script_with_images):
            print(f"\nProcessing segment {i+1}/{len(script_with_images)}")
            
            sentence = item.get('sentence', '').strip()
            if not sentence:
                print("No sentence provided, skipping")
                continue
            
            # Determine clip duration
            if i < len(audio_files) and audio_files[i]:
                clip_duration = get_audio_duration(audio_files[i])
            else:
                clip_duration = default_duration
            
            print(f"Segment duration: {clip_duration:.2f}s")
            
            # Handle image or use logo
            image_path = item.get('image')
            video_clip = None
            
            if image_path:
                # Try to use provided image
                print(f"Attempting to use image: {image_path}")
                
                # Download if it's a URL
                if image_path.startswith('http'):
                    downloaded_path = download_image(image_path)
                    if downloaded_path:
                        image_path = downloaded_path
                    else:
                        print("Failed to download image, using logo")
                        image_path = None
                
                if image_path and os.path.exists(image_path):
                    # Standardize image size
                    standardized_path = resize_image_standard(image_path)
                    
                    # Create image clip
                    img_clip = ImageClip(standardized_path, duration=clip_duration)
                    video_clip = img_clip
                    
                    # Clean up downloaded file if it was temporary
                    if image_path != standardized_path:
                        try:
                            os.unlink(image_path)
                        except:
                            pass
                    
                    print("Successfully created image clip")
                else:
                    print("Image file not accessible, using logo")
                    image_path = None
            
            # Use logo if no image available
            if not image_path or video_clip is None:
                print("Creating animated logo clip")
                video_clip = create_animated_logo_clip(clip_duration)
            
            # Add text overlay
            if sentence:
                # Limit sentence length for display
                display_sentence = sentence
                if len(display_sentence) > 150:
                    display_sentence = display_sentence[:147] + "..."
                
                text_image_path = create_text_image(display_sentence, STANDARD_VIDEO_SIZE)
                if text_image_path:
                    text_image_files.append(text_image_path)
                    text_clip = ImageClip(text_image_path, duration=clip_duration)
                    video_clip = CompositeVideoClip([video_clip, text_clip])
            
            # Add audio if available
            if i < len(audio_files) and audio_files[i]:
                try:
                    audio_clip = AudioFileClip(audio_files[i])
                    video_clip = video_clip.set_audio(audio_clip)
                    print("Audio added successfully")
                except Exception as e:
                    print(f"Could not add audio: {str(e)}")
            
            video_clips.append(video_clip)
            print(f"Segment {i+1} completed successfully")

        # Concatenate all video clips
        if video_clips:
            print(f"\nConcatenating {len(video_clips)} video segments...")
            final_clip = concatenate_videoclips(video_clips, method="compose")
            
            # Write final video with high quality settings
            print("Writing final video file...")
            final_clip.write_videofile(
                output_path,
                fps=24,
                codec='libx264',
                audio_codec='aac',
                threads=4,
                preset='medium',
                bitrate='4000k',  # Higher bitrate for better quality
                audio_bitrate='192k'
            )
            
            # Clean up
            final_clip.close()
            for clip in video_clips:
                clip.close()
            
            print(f"\nVideo created successfully: {output_path}")
            print(f"Total segments processed: {len(video_clips)}")
        else:
            print("No valid video clips were created")
        
        # Clean up temporary files
        for file in audio_files + text_image_files:
            if file and os.path.exists(file):
                try:
                    os.unlink(file)
                except:
                    pass
        
        try:
            os.rmdir(temp_dir)
        except OSError:
            pass
        
    except Exception as e:
        print(f"Error creating video: {str(e)}")
        # Cleanup on error
        try:
            if 'temp_dir' in locals():
                for file in os.listdir(temp_dir):
                    try:
                        os.unlink(os.path.join(temp_dir, file))
                    except:
                        pass
                os.rmdir(temp_dir)
        except:
            pass

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

def generate_video_script(results, formatted_query):
    """Generate a focused, content-rich video script from search results"""
    serializable_results = []
    for result in results:
        serializable_result = {}
        for key, value in result.items():
            if isinstance(value, np.float32):
                serializable_result[key] = float(value)
            else:
                serializable_result[key] = value
        serializable_results.append(serializable_result)
    
    prompt = f"""Create a focused, content-rich video script based on these search results.
    
    REQUIREMENTS:
    - Focus entirely on factual, valuable content
    - Start with a clear introduction that sets the context
    - Break into 10-15 short, informative sentences
    - Each sentence should provide specific, useful information
    - Avoid filler words and generic statements
    - Make it educational and engaging
    - End with a strong conclusion or call-to-action
    
    Search Query: {formatted_query}
    
    Search Results:
    {json.dumps(serializable_results, indent=2)}
    
    Create a content-focused script where every sentence adds value:"""
    
    completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a professional content writer who creates focused, informative video scripts. Every sentence must provide valuable information. Avoid fluff and focus on facts, insights, and actionable content."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.6,
        max_tokens=600
    )
    
    return completion.choices[0].message.content.strip()

def display_results(results):
    """Display the search results in a formatted way"""
    if not results:
        print("No results found.")
        return
        
    print("\n=== Search Results ===")
    for result in results:
        print(f"\nRank {result['rank']} | Score: {result['score']:.4f}")
        print(f"Type: {result['type']}")
        print(f"Content: {result['content'][:200]}...")
        if result.get('timestamp'):
            print(f"Timestamp: {result['timestamp']}")
        if result.get('source'):
            print(f"Source: {result['source']}")
        print("-" * 50)

def main():
    # Get user input
    user_input = input("Enter your search query: ")
    
    # Extract key context from user's query
    key_context = extract_key_context(user_input)
    print(f"\nExtracted key context: {key_context}")
    
    # Format query using Groq for initial search
    formatted_query = format_query_with_groq(user_input)
    if isinstance(formatted_query, dict):
        formatted_query = formatted_query.get('specific', user_input)
    print(f"\nFormatted query: {formatted_query}")
    
    # Search the index
    try:
        results = search_index(formatted_query, "text", top_k=15)  # Get more results for better content
        display_results(results)
        
        if results:
            print("\n=== Generated Content-Focused Video Script ===")
            print("-" * 50)
            script = generate_video_script(results, formatted_query)
            print(script)
            print("-" * 50)
            
            print("\n=== Finding Images for Each Sentence (Company Sources First) ===")
            sentences = [s.strip() for s in script.split('\n') if s.strip()]
            
            # Debug: Print the sentences to see what we're working with
            print(f"DEBUG: Extracted {len(sentences)} sentences from script:")
            for i, sentence in enumerate(sentences):
                print(f"  {i+1}: {sentence[:80]}{'...' if len(sentence) > 80 else ''}")
            
            script_with_images = []
            
            for sentence in sentences:
                print(f"\nProcessing sentence: {sentence[:100]}...")
                image_result = find_relevant_image(sentence, key_context)
                
                # Always include the sentence, even if no image is found
                script_item = {
                    "sentence": sentence,
                    "image": None,
                    "score": None,
                    "source": None
                }
                
                if image_result:
                    script_item.update({
                        "image": image_result["image_path"],
                        "score": image_result["score"],
                        "source": image_result["source"]
                    })
                    source = image_result['source']
                    score_info = f" (Score: {image_result['score']:.4f})" if image_result['score'] is not None else ""
                    print(f"✓ Found image from {source}: {image_result['image_path'][:80]}...{score_info}")
                else:
                    print("⚠ No image found - will use animated logo")
                
                script_with_images.append(script_item)
            
            print("\n=== Final Script Summary ===")
            print(f"Total sentences: {len(script_with_images)}")
            with_images = sum(1 for item in script_with_images if item['image'])
            with_logos = len(script_with_images) - with_images
            print(f"Sentences with images: {with_images}")
            print(f"Sentences with logo animation: {with_logos}")
            
            company_images = sum(1 for item in script_with_images if item.get('source') == 'company')
            external_images = sum(1 for item in script_with_images if item.get('source') == 'external')
            print(f"Company images: {company_images}")
            print(f"External images: {external_images}")
            
            print("\n" + "="*60)
            for i, item in enumerate(script_with_images, 1):
                print(f"\nSegment {i}:")
                print(f"Text: {item['sentence'][:100]}{'...' if len(item['sentence']) > 100 else ''}")
                if item['image']:
                    score_info = f" (Score: {item['score']:.4f})" if item['score'] is not None else ""
                    print(f"Visual: {item['source'].upper()} image{score_info}")
                else:
                    print("Visual: Animated company logo")
                print("-" * 40)
            
            print(f"\n{'='*60}")
            print("🎬 GENERATING VIDEO...")
            print(f"{'='*60}")
            
            # Create the video with all enhancements
            create_video(script_with_images, "enhanced_output_video.mp4")
            
            print(f"\n{'='*60}")
            print("✅ VIDEO CREATION COMPLETE!")
            print("📁 Output file: enhanced_output_video.mp4")
            print(f"🎞️  Total segments: {len(script_with_images)}")
            print(f"🏢 Company visuals: {company_images + with_logos}")
            print(f"🌐 External visuals: {external_images}")
            print(f"{'='*60}")
            
    except Exception as e:
        print(f"Error during search or processing: {str(e)}")
        import traceback
        traceback.print_exc()
        print("No results found.")

if __name__ == "__main__":
    # Configuration check
    print("🚀 Enhanced Video Generator Starting...")
    print(f"📺 Video dimensions: {STANDARD_VIDEO_SIZE[0]}x{STANDARD_VIDEO_SIZE[1]}")
    print(f"🏢 Company logo path: {COMPANY_LOGO_PATH}")
    print(f"🔍 Watermark keywords: {', '.join(WATERMARK_KEYWORDS)}")
    
    if not os.path.exists(COMPANY_LOGO_PATH):
        print(f"⚠️  Logo not found at {COMPANY_LOGO_PATH} - will use placeholder")
    else:
        print("✅ Company logo found")
    
    print("-" * 50)
    main()