import torch
import faiss
import pickle
import numpy as np
import cv2
import librosa
import uuid
import time
import logging
import os
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from transformers import (
    CLIPProcessor, 
    CLIPModel,
    WhisperForConditionalGeneration,
    WhisperProcessor
)
from config import *
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
clip_processor = CLIPProcessor.from_pretrained(MODEL_NAME)
clip_model.eval()

whisper_processor = WhisperProcessor.from_pretrained(AUDIO_MODEL)
whisper_model = WhisperForConditionalGeneration.from_pretrained(AUDIO_MODEL).to(AUDIO_DEVICE)
whisper_model.eval()
EMBEDDING_DIM = 768

def ensure_directories():
    """Ensure all required directories exist"""
    os.makedirs(INDEX_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info("Ensured all required directories exist")

def save_index(index, metadata):
    """Save index and metadata to disk"""
    try:
        faiss.write_index(index, FAISS_INDEX_PATH)
        with open(METADATA_PATH, 'wb') as f:
            pickle.dump(metadata, f)
        logger.info(f"Saved index with {index.ntotal} entries to {FAISS_INDEX_PATH}")
    except Exception as e:
        logger.error(f"Error saving index: {str(e)}")
        raise

def load_or_create_index(create_if_not_exists=False):
    """Load existing index or create a new one if specified"""
    ensure_directories()
    
    try:
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH):
            index = faiss.read_index(FAISS_INDEX_PATH)
            with open(METADATA_PATH, 'rb') as f:
                metadata = pickle.load(f)
            logger.info(f"Loaded existing index with {index.ntotal} entries")
            return index, metadata
        
        if create_if_not_exists:
            logger.info("No existing index found, creating new one")
            index = faiss.IndexFlatIP(EMBEDDING_DIM)
            metadata = {}
            save_index(index, metadata)
            logger.info(f"Created new index with dimension {EMBEDDING_DIM}")
            return index, metadata
        else:
            raise FileNotFoundError("No index found in the indices directory")
        
    except Exception as e:
        if create_if_not_exists:
            logger.error(f"Error loading/creating index: {str(e)}")
            logger.info("Creating new index after error")
            index = faiss.IndexFlatIP(EMBEDDING_DIM)
            metadata = {}
            save_index(index, metadata)
            return index, metadata
        else:
            raise

def process_video(video_path, progress_callback=None):
    start_time = time.time()
    logger.info(f"Starting video processing: {video_path}")
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        logger.info(f"Video info - FPS: {fps}, Total frames: {total_frames}, Duration: {duration:.1f}s")
        
        # Load audio in chunks to avoid memory issues
        logger.info("Loading audio...")
        audio, sr = librosa.load(video_path, sr=16000)
        logger.info(f"Audio loaded - Duration: {librosa.get_duration(y=audio, sr=sr):.1f}s")
        
        frames = []
        previous_frame = None
        current_time = 0
        processed_frames = 0
        
        while current_time < duration:
            if progress_callback:
                progress = min(1.0, current_time / duration)
                progress_callback(progress)
            
            cap.set(cv2.CAP_PROP_POS_MSEC, current_time*1000)
            ret, frame = cap.read()
            if not ret:
                break
                
            # Resize frame to reduce memory usage
            frame = cv2.resize(frame, (640, 360))
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if previous_frame is not None:
                try:
                    similarity = ssim(previous_frame, gray_frame)
                    if similarity > VIDEO_SIMILARITY_THRESHOLD:
                        current_time += VIDEO_FRAME_INTERVAL
                        continue
                except Exception as e:
                    logger.warning(f"Frame similarity check failed: {str(e)}")
            
            # Calculate audio chunk boundaries for continuous 10-second intervals
            chunk_start = current_time
            chunk_end = min(duration, current_time + VIDEO_FRAME_INTERVAL)
            
            # Extract audio chunk
            audio_chunk = audio[int(chunk_start*sr):int(chunk_end*sr)]
            
            frames.append({
                "frame_time": current_time,
                "frame_image": Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                "audio_start": chunk_start,
                "audio_end": chunk_end,
                "audio_chunk": audio_chunk
            })
            
            processed_frames += 1
            if processed_frames % 10 == 0:
                logger.info(f"Processed {processed_frames} frames, current time: {current_time:.1f}s")
            
            previous_frame = gray_frame
            current_time += VIDEO_FRAME_INTERVAL
            
        cap.release()
        processing_time = time.time() - start_time
        logger.info(f"Video processing complete - {processed_frames} frames in {processing_time:.1f}s")
        return frames
        
    except Exception as e:
        logger.error(f"Video processing failed: {str(e)}")
        if cap is not None:
            cap.release()
        raise

def transcribe_audio(audio_chunk, sr=16000):
    try:
        if len(audio_chunk) == 0:
            logger.warning("Empty audio chunk")
            return None
            
        logger.info(f"Transcribing audio chunk of length {len(audio_chunk)/sr:.1f}s")
        inputs = whisper_processor(
            audio_chunk, 
            sampling_rate=sr, 
            return_tensors="pt"
        ).input_features.to(AUDIO_DEVICE)
        
        with torch.no_grad():
            predicted_ids = whisper_model.generate(inputs)
        
        transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        logger.info(f"Transcription: {transcription[:100]}...")
        return transcription
    except Exception as e:
        logger.error(f"Audio transcription failed: {str(e)}")
        return None

def embed_text(text):
    try:
        inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            text_emb = clip_model.get_text_features(**inputs)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        return text_emb.cpu().squeeze(0).numpy()
    except Exception as e:
        logger.error(f"Text embedding failed: {str(e)}")
        raise

def embed_image(image):
    try:
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            img_emb = clip_model.get_image_features(**inputs)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        return img_emb.cpu().squeeze(0).numpy()
    except Exception as e:
        logger.error(f"Image embedding failed: {str(e)}")
        raise

def chunk_text(text, chunk_size=CHUNK_SIZE):
    try:
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
            current_chunk.append(word)
            current_length += len(word) + 1
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.error(f"Text chunking failed: {str(e)}")
        raise