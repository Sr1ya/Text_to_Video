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

# Load environment variables from .env file
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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

def resize_image(image_path, target_size=(1920, 1080)):
    """Resize image to target size while maintaining aspect ratio"""
    try:
        with Image.open(image_path) as img:
            # Calculate new size maintaining aspect ratio
            img_ratio = img.width / img.height
            target_ratio = target_size[0] / target_size[1]
            
            if img_ratio > target_ratio:
                # Image is wider than target
                new_height = target_size[1]
                new_width = int(new_height * img_ratio)
            else:
                # Image is taller than target
                new_width = target_size[0]
                new_height = int(new_width / img_ratio)
                
            # Resize image
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Create new image with target size
            new_img = Image.new('RGB', target_size, (0, 0, 0))
            
            # Paste resized image centered
            offset = ((target_size[0] - new_width) // 2, (target_size[1] - new_height) // 2)
            new_img.paste(resized_img, offset)
            
            # Save resized image
            resized_path = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg').name
            new_img.save(resized_path, 'JPEG', quality=90)
            return resized_path
            
    except Exception as e:
        print(f"Error resizing image: {str(e)}")
        return image_path  # Return original if resize fails

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

def create_text_image(text, image_size, font_size=32):
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
                font = ImageFont.load_default().font_variant(size=font_size)
        except Exception:
            font = ImageFont.load_default()
        
        # Wrap text to fit image width with margins
        words = text.split()
        lines = []
        current_line = []
        max_width = image_size[0] - 80  # 40px margin on each side
        
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
        line_height = int(font_size * 1.2)
        total_text_height = len(lines) * line_height
        padding = 20
        
        # Position text block at bottom center
        text_block_width = image_size[0] - 80
        text_block_height = total_text_height + padding * 2
        text_block_y = image_size[1] - text_block_height - 40
        
        # Draw semi-transparent background
        bg_color = (0, 0, 0, 180)  # Dark semi-transparent
        draw.rectangle(
            [(40, text_block_y), 
             (40 + text_block_width, text_block_y + text_block_height)],
            fill=bg_color
        )
        
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

def create_video(script_with_images, output_path="output_video.mp4", default_duration=5):
    """Create video with consistent image sizes and captions"""
    try:
        if not script_with_images:
            print("No content to create video")
            return

        temp_dir = tempfile.mkdtemp()
        audio_files = []
        video_clips = []
        text_image_files = []
        
        # Standard video dimensions
        video_size = (1920, 1080)
        
        # Generate all audio files first
        for i, item in enumerate(script_with_images):
            if item.get('sentence'):
                audio_path = os.path.join(temp_dir, f"audio_{i}.mp3")
                if generate_audio_for_sentence(item['sentence'], audio_path):
                    audio_files.append(audio_path)
                else:
                    audio_files.append(None)
            else:
                audio_files.append(None)

        # Process each image
        for i, item in enumerate(script_with_images):
            print(f"\nProcessing clip {i+1}/{len(script_with_images)}")
            
            if not item.get('image'):
                print("No image provided, skipping")
                continue
                
            image_path = download_image(item['image'])
            if not image_path:
                print("Could not download image, skipping")
                continue
                
            try:
                # Resize image to standard size
                resized_path = resize_image(image_path, video_size)
                if resized_path != image_path:
                    os.unlink(image_path)  # Delete original if we created a resized version
                    image_path = resized_path
                
                # Determine clip duration
                if i < len(audio_files) and audio_files[i]:
                    clip_duration = get_audio_duration(audio_files[i])
                else:
                    clip_duration = default_duration
                
                # Create image clip
                img_clip = ImageClip(image_path, duration=clip_duration)
                
                # Create text overlay if we have text
                sentence_text = str(item.get('sentence', '')).strip()
                if sentence_text:
                    # Shorten very long text
                    if len(sentence_text) > 150:
                        sentence_text = sentence_text[:147] + "..."
                    
                    text_image_path = create_text_image(sentence_text, video_size)
                    if text_image_path:
                        text_image_files.append(text_image_path)
                        text_clip = ImageClip(text_image_path, duration=clip_duration)
                        video_clip = CompositeVideoClip([img_clip, text_clip])
                    else:
                        video_clip = img_clip
                else:
                    video_clip = img_clip
                
                # Add audio if available
                if i < len(audio_files) and audio_files[i]:
                    try:
                        audio_clip = AudioFileClip(audio_files[i])
                        video_clip.audio = audio_clip
                    except Exception as e:
                        print(f"Could not add audio: {str(e)}")
                
                video_clips.append(video_clip)
                os.unlink(image_path)
                
            except Exception as e:
                print(f"Error processing clip: {str(e)}")
                if os.path.exists(image_path):
                    os.unlink(image_path)
                continue

        # Concatenate all video clips
        if video_clips:
            final_clip = concatenate_videoclips(video_clips, method="compose")
            
            # Write final video
            final_clip.write_videofile(
                output_path,
                fps=24,
                codec='libx264',
                audio_codec='aac',
                threads=4,
                preset='medium',
                bitrate='3000k'  # Increased bitrate for better quality
            )
            
            # Clean up
            final_clip.close()
            for clip in video_clips:
                clip.close()
            
            print(f"\nVideo created successfully: {output_path}")
        else:
            print("No valid video clips were created")
        
        # Clean up temporary files
        for file in audio_files + text_image_files:
            if file and os.path.exists(file):
                os.unlink(file)
        
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
                    os.unlink(os.path.join(temp_dir, file))
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

def search_duckduckgo_images(query, context=""):
    """Search for images using DuckDuckGo"""
    if not query or not query.strip():
        print("Error: Empty search query")
        return None
        
    try:
        with DDGS() as ddgs:
            query_variations = format_query_with_groq(query, purpose="image_search", context=context)
            print(f"Generated query variations: {query_variations}")
            
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
                    print(f"Searching with parameters: {params}")
                    results = list(ddgs.images(**params))
                    
                    if results:
                        print(f"Found {len(results)} images")
                        for result in results:
                            if result.get('image'):
                                print(f"Found valid image: {result['image']}")
                                return result['image']
                except Exception as e:
                    print(f"Error with current search attempt: {str(e)}")
                    continue
            
            print("\nTrying final fallback with original query...")
            try:
                results = list(ddgs.images(keywords=f"{context} {query}", max_results=5))
                if results and results[0].get('image'):
                    return results[0]['image']
            except Exception as e:
                print(f"Final fallback attempt failed: {str(e)}")
            
            print("All search attempts failed to find images")
            return None
            
    except Exception as e:
        print(f"Error in DuckDuckGo search: {str(e)}")
        return None

def find_relevant_image(sentence, context=""):
    """Find the most relevant image for a given sentence"""
    print(f"\nSearching for image for sentence: {sentence}")
    
    print("Trying DuckDuckGo search first...")
    duckduckgo_image = search_duckduckgo_images(sentence, context)
    if duckduckgo_image:
        print(f"Found image in DuckDuckGo: {duckduckgo_image}")
        return {
            "sentence": sentence,
            "image_path": duckduckgo_image,
            "score": None,
            "source": "duckduckgo",
            "type": "image"
        }
    
    print("DuckDuckGo search failed, trying index...")
    query_variations = format_query_with_groq(sentence, purpose="image_search", context=context)
    image_query = query_variations['specific'] if isinstance(query_variations, dict) and 'specific' in query_variations else sentence
    
    print(f"Formatted search query for index: {image_query}")
    
    try:
        results = search_index(image_query, "text", top_k=10)
        print(f"Found {len(results)} results in index")
        
        for result in results:
            if result.get("type") in ["image", "video_frame"]:
                print(f"Found image in index: {result.get('content')}")
                return {
                    "sentence": sentence,
                    "image_path": result.get("content"),
                    "score": result.get("score"),
                    "source": "index",
                    "type": result.get("type")
                }
    except Exception as e:
        print(f"Error searching index: {str(e)}")
    
    print("No image found in any source")
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
        results = search_index(formatted_query, "text", top_k=10)
        display_results(results)
        
        if results:
            print("\n=== Generated Video Script ===")
            print("-" * 50)
            script = generate_video_script(results, formatted_query)
            print(script)
            print("-" * 50)
            
            print("\n=== Finding Relevant Images for Each Sentence ===")
            sentences = [s.strip() for s in script.split('\n') if s.strip()]
            
            # Debug: Print the sentences to see what we're working with
            print(f"DEBUG: Extracted {len(sentences)} sentences from script:")
            for i, sentence in enumerate(sentences):
                print(f"  {i+1}: {sentence[:80]}{'...' if len(sentence) > 80 else ''}")
            
            script_with_images = []
            
            for sentence in sentences:
                print(f"\nProcessing sentence: {sentence}")
                image_result = find_relevant_image(sentence, key_context)
                if image_result:
                    source = image_result['source']
                    score_info = f" (Score: {image_result['score']:.4f})" if image_result['score'] is not None else ""
                    print(f"Found relevant image from {source}: {image_result['image_path']}{score_info}")
                    script_with_images.append({
                        "sentence": sentence,
                        "image": image_result["image_path"],
                        "score": image_result["score"],
                        "source": source
                    })
                else:
                    print("No relevant image found")
                    script_with_images.append({
                        "sentence": sentence,
                        "image": None,
                        "score": None,
                        "source": None
                    })
            
            print("\n=== Final Script with Images ===")
            for item in script_with_images:
                print(f"\nSentence: {item['sentence']}")
                if item['image']:
                    score_info = f" (Relevance Score: {item['score']:.4f})" if item['score'] is not None else ""
                    print(f"Image: {item['image']} (Source: {item['source']}){score_info}")
                else:
                    print("No matching image found")
                print("-" * 50)
            
            print("\n=== Generating Video ===")
            create_video(script_with_images)
    except Exception as e:
        print(f"Error during search or processing: {str(e)}")
        print("No results found.")

if __name__ == "__main__":
    main()