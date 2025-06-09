from utils import *
from PIL import Image
import librosa
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def get_search_results(scores, indices, metadata):
    results = []
    for rank, (score, idx) in enumerate(zip(scores, indices)):
        item = metadata.get(idx, {})
        if not item:
            continue
            
        result = {
            "rank": rank + 1,
            "score": score,
            "type": item["type"],
            "content": item["content"],
            "timestamp": item.get("timestamp"),
            "source": item.get("source"),
            "linked_audio": item.get("linked_audio"),
            "linked_frame": item.get("linked_frame")
        }
        results.append(result)
    return results

def search_index(query, query_type, top_k=5):
    try:
        index, metadata = load_or_create_index(create_if_not_exists=False)
        
        if query_type == "text":
            emb = embed_text(query)
        elif query_type == "image":
            image = Image.open(query).convert("RGB")
            emb = embed_image(image)
        elif query_type == "audio":
            audio, sr = librosa.load(query, sr=16000)
            text = transcribe_audio(audio, sr)
            emb = embed_text(text) if text else None
        elif query_type == "video":
            frames = process_video(query)
            emb = np.mean([embed_image(f["frame_image"]) for f in frames], axis=0)
        else:
            raise ValueError("Invalid query type")

        if emb is None:
            print("Failed to generate query embedding")
            return None

        D, I = index.search(emb.reshape(1, -1), top_k)
        return get_search_results(D[0], I[0], metadata)
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        print("Please run generate_index.py first to create an index.")
        return None
    except Exception as e:
        print(f"Error searching index: {str(e)}")
        return None

def get_text_content(query, top_k=5):
    results = search_index(query, "text", top_k)
    if not results:
        return []
    
    text_content = []
    for result in results:
        if result["type"] == "text":
            text_content.append(result["content"])
    return text_content

if __name__ == "__main__":
    query_input = "how does a piston work"
    results = search_index(query_input, "text", 5)
    
    if results:
        print("\n Search Results:")
        for result in results:
            print(f"\n Rank {result['rank']} | Score: {result['score']:.4f}")
            print(f" Content: {result['content'][:200]}...")
