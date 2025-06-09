import streamlit as st
import os
import time
from webscrape import AdvancedWebScraper 

def main():
    st.set_page_config(
        page_title="Advanced Web Scraper",
        page_icon="🌐",
        layout="wide"
    )
    
    st.title("🌐 Advanced Web Scraper")
    st.markdown("""
    This tool scrapes websites for text, images, videos, and other media content.
    It supports downloading videos from platforms like YouTube, Vimeo, TikTok, and more.
    """)
    
    with st.sidebar:
        st.header("Settings")
        base_url = st.text_input("Base URL", placeholder="https://example.com")
        max_pages = st.number_input("Maximum Pages to Scrape", min_value=1, max_value=1000, value=50)
        delay = st.number_input("Delay Between Requests (seconds)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        output_dir = st.text_input("Output Directory", value="scraped_data")
        
        st.markdown("---")
        st.markdown("**Note:** Large websites may take significant time to scrape.")
        st.markdown("Be respectful of website terms of service and robots.txt.")
    
    if st.button("Start Scraping", type="primary"):
        if not base_url:
            st.error("Please enter a valid URL")
            return
            
        # Initialize scraper
        scraper = AdvancedWebScraper(
            base_url=base_url,
            max_pages=max_pages,
            delay=delay,
            output_dir=output_dir
        )
        
        # Set up progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        
        # Initialize metrics
        total_pages = min(max_pages, 1000)  # Just for progress estimation
        visited_count = 0
        downloaded_files = {
            'text': 0,
            'images': 0,
            'videos': 0,
            'audio': 0,
            'other': 0
        }
        
        # Create a placeholder for logs
        log_placeholder = st.empty()
        logs = []
        
        # Custom print function to capture logs
        def streamlit_print(*args, **kwargs):
            message = " ".join(str(arg) for arg in args)
            logs.append(message)
            if len(logs) > 20:  # Keep last 20 logs
                logs.pop(0)
            log_placeholder.code("\n".join(logs), language="log")
        
        # Monkey-patch the scraper's print statements
        import builtins
        original_print = builtins.print
        builtins.print = streamlit_print
        
        try:
            # Start scraping
            with st.spinner("Starting scraping process..."):
                scraper.crawl_and_scrape()
                
                # Update metrics (in a real implementation, you'd track these from the scraper)
                visited_count = len(scraper.visited_urls)
                
                # Count downloaded files (this would need to be tracked in the scraper class)
                for root, dirs, files in os.walk(output_dir):
                    if "text" in root:
                        downloaded_files['text'] += len(files)
                    elif "images" in root:
                        downloaded_files['images'] += len(files)
                    elif "videos" in root:
                        downloaded_files['videos'] += len(files)
                    elif "audio" in root:
                        downloaded_files['audio'] += len(files)
                    elif "other" in root:
                        downloaded_files['other'] += len(files)
                
                # Update progress
                progress_bar.progress(1.0)
                status_text.success("Scraping complete!")
                
                # Display metrics
                with metrics_col1:
                    st.metric("Pages Visited", visited_count)
                with metrics_col2:
                    st.metric("Text Files Saved", downloaded_files['text'])
                with metrics_col3:
                    st.metric("Media Files Saved", sum(downloaded_files.values()) - downloaded_files['text'])
                
                # Show directory structure
                st.subheader("Downloaded Content")
                st.markdown(f"""
                - **Text files:** `{output_dir}/text/` ({downloaded_files['text']} files)
                - **Images:** `{output_dir}/images/` ({downloaded_files['images']} files)
                - **Videos:** `{output_dir}/videos/` ({downloaded_files['videos']} files)
                - **Audio:** `{output_dir}/audio/` ({downloaded_files['audio']} files)
                - **Other files:** `{output_dir}/other/` ({downloaded_files['other']} files)
                """)
                
                # Offer download option
                if any(downloaded_files.values()):
                    st.download_button(
                        label="Download All as ZIP",
                        data=create_zip(output_dir),
                        file_name=f"scraped_{urlparse(base_url).netloc}.zip",
                        mime="application/zip"
                    )
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        finally:
            # Restore original print function
            builtins.print = original_print

def create_zip(output_dir):
    """Create a zip file of the scraped content (placeholder implementation)"""
    # In a real implementation, you would use zipfile to create an archive
    # This is just a placeholder
    return b"ZIP file content would be here"

if __name__ == "__main__":
    main()