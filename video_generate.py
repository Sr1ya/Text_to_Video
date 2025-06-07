import os
from groq import Groq
from search_index import search_index
import json
from dotenv import load_dotenv
import numpy as np
from duckduckgo_search import DDGS
from moviepy import ImageClip, concatenate_videoclips, TextClip, CompositeVideoClip, ColorClip, AudioFileClip
import requests
from PIL import Image
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
        response = requests.get(url, timeout=10)
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

def generate_audio_for_sentence(sentence, output_path):
    """Generate audio for a sentence using gTTS"""
    try:
        tts = gTTS(text=sentence, lang='en', slow=False)
        tts.save(output_path)
        
        temp_output = output_path + ".temp.mp3"
        subprocess.run([
            'ffmpeg', '-y',
            '-i', output_path,
            '-filter:a', 'atempo=1',
            '-vn',
            temp_output
        ], check=True)
        
        os.replace(temp_output, output_path)
        return True
    except Exception as e:
        print(f"Error generating audio: {str(e)}")
        return False

def create_video(script_with_images, output_path="output_video.mp4", duration_per_image=5):
    """Create video from images and add subtitles with audio using moviepy"""
    try:
        if not script_with_images or not any(item['image'] for item in script_with_images):
            print("No valid images found for video creation")
            return

        temp_dir = tempfile.mkdtemp()
        audio_files = []
        video_clips = []

        # Generate all audio files first
        print("\nGenerating audio for each sentence...")
        for i, item in enumerate(script_with_images):
            audio_path = os.path.join(temp_dir, f"audio_{i}.mp3")
            if generate_audio_for_sentence(item['sentence'], audio_path):
                audio_files.append(audio_path)
                print(f"Generated audio for sentence {i+1}/{len(script_with_images)}")

        # Process each image
        print("\nCreating video clips...")
        for i, item in enumerate(script_with_images):
            print(f"\nProcessing image {i+1}/{len(script_with_images)}")
            if item['image']:
                image_path = download_image(item['image'])
                if image_path:
                    print(f"Downloaded image to: {image_path}")
                    
                    # Create image clip
                    img_clip = ImageClip(image_path, duration=duration_per_image)
                    
                    # Create text clip for subtitle
                    txt_clip = TextClip(
                        item['sentence'], 
                        fontsize=24, 
                        color='white', 
                        font='Arial-Bold',
                        size=(img_clip.size[0]*0.9, None),
                        method='caption'
                    )
                    txt_clip = txt_clip.set_position(('center', 'bottom')).set_duration(duration_per_image)
                    
                    # Create semi-transparent background for text
                    txt_bg = ColorClip(
                        size=(img_clip.size[0], txt_clip.size[1] + 20),
                        color=(0, 0, 0),
                        duration=duration_per_image
                    )
                    txt_bg = txt_bg.set_opacity(0.7).set_position(('center', img_clip.size[1] - txt_clip.size[1] - 20))
                    
                    # Composite all elements
                    video_clip = CompositeVideoClip([
                        img_clip,
                        txt_bg,
                        txt_clip
                    ])
                    
                    # Set audio if available
                    if i < len(audio_files):
                        audio_clip = AudioFileClip(audio_files[i])
                        video_clip = video_clip.set_audio(audio_clip)
                    
                    video_clips.append(video_clip)
                    os.unlink(image_path)
                    print("Cleaned up image file")
                else:
                    print(f"Warning: Could not download image for sentence {i+1}")
            else:
                print(f"Warning: No image provided for sentence {i+1}")
        
        # Concatenate all video clips
        if video_clips:
            print("\nConcatenating video clips...")
            final_clip = concatenate_videoclips(video_clips, method="compose")
            
            # Write the final video file
            print("\nWriting final video file...")
            final_clip.write_videofile(
                output_path,
                fps=24,
                codec='libx264',
                audio_codec='aac',
                threads=4,
                preset='fast',
                bitrate='3000k'
            )
            
            # Clean up temporary files
            print("\nCleaning up temporary files...")
            for file in audio_files:
                os.unlink(file)
            os.rmdir(temp_dir)
            
            print(f"\nVideo created successfully: {output_path}")
            print(f"Video duration: {len(script_with_images) * duration_per_image} seconds")
        else:
            print("No valid video clips were created")
        
    except Exception as e:
        print(f"Error creating video: {str(e)}")
        try:
            for file in audio_files:
                if os.path.exists(file):
                    os.unlink(file)
            if os.path.exists(temp_dir):
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
    try:
        with DDGS() as ddgs:
            query_variations = format_query_with_groq(query, purpose="image_search", context=context)
            print(f"Generated query variations: {query_variations}")
            
            search_params = [
                {'max_results': 5, 'safesearch': 'on', 'size': 'Large', 'type_image': 'photo'},
                {'max_results': 5, 'safesearch': 'on', 'size': 'Large'},
                {'max_results': 5}
            ]
            
            for query_type, search_query in query_variations.items():
                print(f"\nTrying {query_type} query: {search_query}")
                
                for params in search_params:
                    try:
                        print(f"Searching with parameters: {params}")
                        results = list(ddgs.images(search_query, **params))
                        
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
                results = list(ddgs.images(f"{context} {query}", max_results=5))
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
    """Generate a 1-minute video script from search results"""
    serializable_results = []
    for result in results:
        serializable_result = {}
        for key, value in result.items():
            if isinstance(value, np.float32):
                serializable_result[key] = float(value)
            else:
                serializable_result[key] = value
        serializable_results.append(serializable_result)
    
    prompt = f"""Create a concise 1-minute video script based on these search results. 
    Each line should take 5 seconds to speak (about 12-15 words per line).
    Total script should be 12 lines (1 minute total).
    Focus on the most important information from the results.
    Make it flow naturally and be engaging.
    
    Search Query: {formatted_query}
    
    Search Results:
    {json.dumps(serializable_results, indent=2)}
    
    Create a script with exactly 12 lines, each line taking 5 seconds to speak:"""
    
    completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a professional script writer who creates concise, engaging video scripts."
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
            print("\n=== Generated Video Script (1 minute) ===")
            print("Each line takes 5 seconds to speak:")
            print("-" * 50)
            script = generate_video_script(results, formatted_query)
            print(script)
            print("-" * 50)
            
            print("\n=== Finding Relevant Images for Each Sentence ===")
            sentences = [s.strip() for s in script.split('\n') if s.strip()]
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