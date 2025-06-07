import streamlit as st
import os
from webscrape import AdvancedWebScraper
from generate_index import create_new_index, process_directory
from video_generator import (
    extract_key_context,
    format_query_with_groq,
    search_index,
    generate_video_script,
    find_relevant_image,
    create_video
)

st.set_page_config(page_title="Text-to-Video Generator", layout="wide")

st.title("📹 Text-to-Video Generator Workflow")

# --- Step 1: Webscraping ---
st.header("Step 1: Webscraping")
with st.form("webscrape_form"):
    url = st.text_input("Enter the URL to scrape", "")
    max_pages = st.number_input("Max pages to scrape", min_value=1, max_value=200, value=50)
    delay = st.number_input("Delay between requests (seconds)", min_value=0.0, max_value=10.0, value=1.0)
    output_dir = st.text_input("Output directory", "scraped_data")
    submitted = st.form_submit_button("Start Webscraping")

if submitted and url:
    st.info(f"Starting webscraping for: {url}")
    try:
        scraper = AdvancedWebScraper(
            base_url=url,
            max_pages=int(max_pages),
            delay=float(delay),
            output_dir=output_dir
        )
        scraper.crawl_and_scrape()
        st.success(f"Webscraping complete. Data saved to '{output_dir}'")
    except Exception as e:
        st.error(f"Webscraping failed: {e}")

# --- Step 2: Indexing ---
st.header("Step 2: Indexing")
if st.button("Run Indexing on Scraped Data"):
    st.info(f"Indexing data in: {output_dir}")
    try:
        index, metadata = create_new_index()
        success = process_directory(output_dir, index, metadata)
        if success:
            st.success("Indexing complete.")
        else:
            st.error("Indexing failed. Check logs for details.")
    except Exception as e:
        st.error(f"Indexing failed: {e}")

# --- Step 3: Video Generation ---
st.header("Step 3: Video Generation")
with st.form("video_form"):
    video_query = st.text_input("Enter the topic or query for the video", "")
    video_output = st.text_input("Output video filename", "output_video.mp4")
    video_submitted = st.form_submit_button("Generate Video")

if video_submitted and video_query:
    st.info("Generating video script and finding images...")
    try:
        key_context = extract_key_context(video_query)
        formatted_query = format_query_with_groq(video_query)
        if isinstance(formatted_query, dict):
            formatted_query = formatted_query.get('specific', video_query)
        st.write(f"**Key context:** {key_context}")
        st.write(f"**Formatted query:** {formatted_query}")

        results = search_index(formatted_query, "text", top_k=10)
        if not results:
            st.error("No relevant data found in the index for this query.")
        else:
            script = generate_video_script(results, formatted_query)
            st.subheader("Generated Script")
            st.write(script)

            st.info("Finding relevant images for each sentence...")
            sentences = [s.strip() for s in script.split('\n') if s.strip()]
            script_with_images = []
            progress_bar = st.progress(0)
            for i, sentence in enumerate(sentences):
                image_result = find_relevant_image(sentence, key_context)
                script_with_images.append({
                    "sentence": sentence,
                    "image": image_result["image_path"] if image_result else None,
                    "score": image_result["score"] if image_result else None,
                    "source": image_result["source"] if image_result else None
                })
                progress_bar.progress((i + 1) / len(sentences))
            st.success("Image search complete.")

            st.subheader("Script with Images")
            for item in script_with_images:
                st.markdown(f"**Sentence:** {item['sentence']}")
                if item['image']:
                    st.image(item['image'], caption=f"Source: {item['source']}")
                else:
                    st.warning("No image found for this sentence.")

            st.info("Generating video...")
            create_video(script_with_images, output_path=video_output)
            if os.path.exists(video_output):
                st.success(f"Video created: {video_output}")
                st.video(video_output)
            else:
                st.error("Video generation failed.")
    except Exception as e:
        st.error(f"Video generation failed: {e}")

st.markdown("---")
st.markdown("Developed as a unified workflow for text-to-video generation. Each step must be run in order for best results.")