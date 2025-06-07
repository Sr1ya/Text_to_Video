import os
import logging
from add_to_index import add_file
from utils import save_index, load_or_create_index, EMBEDDING_DIM
from config import *  # Import other config variables we might need
from datetime import datetime
import faiss
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_new_index():
    """Create a new index and delete any existing ones"""
    # Delete existing index files if they exist
    if os.path.exists(FAISS_INDEX_PATH):
        os.remove(FAISS_INDEX_PATH)
        logger.info("Deleted existing index file")
    if os.path.exists(METADATA_PATH):
        os.remove(METADATA_PATH)
        logger.info("Deleted existing metadata file")
    
    # Create new index
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    metadata = {}
    logger.info("Created new index")
    
    return index, metadata

def process_directory(directory_path, index, metadata, should_stop=None, progress_callback=None):
    """
    Recursively process all files in the directory and its subdirectories.
    Optionally accepts:
        - should_stop: a callable returning True if the process should stop.
        - progress_callback: a callable accepting (current_file, total_files).
    """
    if not os.path.exists(directory_path):
        logger.error(f"Directory not found: {directory_path}")
        return False

    logger.info(f"Processing directory: {directory_path}")
    
    # Track statistics
    total_files = 0
    processed_files = 0
    failed_files = 0

    # First, count total files for progress reporting
    all_files = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            all_files.append(os.path.join(root, file))
    total_files = len(all_files)

    for idx, file_path in enumerate(all_files, 1):
        if should_stop and should_stop():
            logger.info("Indexing stopped by user.")
            break

        try:
            logger.info(f"Processing file {idx}: {file_path}")
            if add_file(file_path, index, metadata, len(metadata)):
                processed_files += 1
            else:
                failed_files += 1
            
            # Save progress periodically
            if idx % 10 == 0:
                faiss.write_index(index, FAISS_INDEX_PATH)
                with open(METADATA_PATH, 'wb') as f:
                    pickle.dump(metadata, f)
                logger.info(f"Saved progress after {idx} files")
                
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            failed_files += 1
            continue

        if progress_callback:
            progress_callback(idx, total_files)

    # Final save
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(METADATA_PATH, 'wb') as f:
        pickle.dump(metadata, f)
    
    # Log summary
    logger.info(f"\nProcessing complete!")
    logger.info(f"Total files found: {total_files}")
    logger.info(f"Successfully processed: {processed_files}")
    logger.info(f"Failed to process: {failed_files}")
    
    return True

if __name__ == "__main__":
    # Specify the directory path here
    data_directory = "/home/SC/startcoding/startCoding_Video_Model/scraped_data"  # Change this to your desired directory path
    
    # Create new index
    index, metadata = create_new_index()
    
    # Process the directory
    process_directory(data_directory, index, metadata)
