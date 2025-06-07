import os
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import argparse
from collections import deque
import time
import re
from pathlib import Path
import yt_dlp as youtube_dl
import mimetypes

class AdvancedWebScraper:
    def __init__(self, base_url, max_pages=50, delay=1, user_agent=None, output_dir="scraped_data"):
        self.base_url = base_url
        self.domain = urlparse(base_url).netloc
        self.visited_urls = set()
        self.urls_to_visit = deque([base_url])
        self.max_pages = max_pages
        self.delay = delay
        self.user_agent = user_agent or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        self.output_dir = output_dir
        self.create_output_dirs()
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.user_agent})
        self.video_platforms = {
            'youtube': [
                r'(https?://)?(www\.)?(youtube\.com|youtu\.be)/.+',
                r'(https?://)?(www\.)?youtube\.com/watch\?v=[\w-]+',
                r'(https?://)?(www\.)?youtu\.be/[\w-]+',
                r'(https?://)?(www\.)?youtube\.com/shorts/[\w-]+'
            ],
            'vimeo': [
                r'(https?://)?(www\.)?vimeo\.com/\d+',
                r'(https?://)?player\.vimeo\.com/video/\d+'
            ],
            'dailymotion': [
                r'(https?://)?(www\.)?dailymotion\.com/video/[\w-]+',
                r'(https?://)?dai\.ly/[\w-]+'
            ],
            'tiktok': [
                r'(https?://)?(www\.)?tiktok\.com/@[^/]+/video/\d+',
                r'(https?://)?vm\.tiktok\.com/[\w-]+'
            ]
        }
        
    def create_output_dirs(self):
        """Create directory structure for scraped data"""
        Path(f"{self.output_dir}/text").mkdir(parents=True, exist_ok=True)
        Path(f"{self.output_dir}/images").mkdir(parents=True, exist_ok=True)
        Path(f"{self.output_dir}/audio").mkdir(parents=True, exist_ok=True)
        Path(f"{self.output_dir}/videos").mkdir(parents=True, exist_ok=True)
        Path(f"{self.output_dir}/other").mkdir(parents=True, exist_ok=True)
        
    def is_valid_url(self, url):
        """Check if URL belongs to target domain and is valid"""
        parsed = urlparse(url)
        return bool(parsed.netloc) and bool(parsed.scheme)
    
    def sanitize_filename(self, filename):
        """Remove invalid characters from filenames"""
        filename = re.sub(r'[\\/*?:"<>|]', "_", filename)
        return filename[:200]  # Limit filename length
    
    def is_video_platform_url(self, url):
        """Check if URL is from a known video platform"""
        for platform, patterns in self.video_platforms.items():
            for pattern in patterns:
                if re.match(pattern, url, re.IGNORECASE):
                    return platform
        return None
    
    def download_video_from_platform(self, url, platform):
        """Download video from supported platforms using yt-dlp"""
        try:
            ydl_opts = {
                'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
                'outtmpl': f'{self.output_dir}/videos/%(title)s.%(ext)s',
                'quiet': False,
                'no_warnings': False,
                'ignoreerrors': True,
                'retries': 3,
                'extract_flat': False,
                'merge_output_format': 'mp4',
                'concurrent_fragment_downloads': 3,
                'http_headers': {'User-Agent': self.user_agent}
            }
            
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(url, download=True)
                if info_dict:
                    filename = ydl.prepare_filename(info_dict)
                    print(f"Downloaded {platform} video: {filename}")
                    return True
        except Exception as e:
            print(f"Error downloading {platform} video {url}: {str(e)[:200]}...")
        return False
    
    def download_direct_video(self, url):
        """Download direct video files with improved error handling"""
        try:
            headers = {'User-Agent': self.user_agent}
            with self.session.get(url, headers=headers, stream=True, timeout=30) as response:
                if response.status_code == 200:
                    # Get filename from URL or content-disposition
                    content_disposition = response.headers.get('content-disposition')
                    if content_disposition:
                        filename = re.findall('filename=(.+)', content_disposition)[0].strip('"\'')
                    else:
                        filename = Path(urlparse(url).path).name
                    
                    if not filename:
                        filename = f"video_{int(time.time())}.mp4"
                    
                    filename = self.sanitize_filename(filename)
                    save_path = f"{self.output_dir}/videos/{filename}"
                    
                    # Check if file exists and add suffix if needed
                    counter = 1
                    while os.path.exists(save_path):
                        name, ext = os.path.splitext(filename)
                        save_path = f"{self.output_dir}/videos/{name}_{counter}{ext}"
                        counter += 1
                    
                    # Download the file with progress
                    file_size = int(response.headers.get('content-length', 0))
                    downloaded = 0
                    start_time = time.time()
                    
                    with open(save_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=1024*1024):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                if downloaded % (5 * 1024 * 1024) == 0:
                                    elapsed = time.time() - start_time
                                    speed = downloaded / (elapsed + 0.0001) / (1024 * 1024)
                                    print(f"Downloading: {downloaded/(1024*1024):.1f}MB/{file_size/(1024*1024):.1f}MB "
                                          f"({speed:.1f}MB/s)", end='\r')
                    
                    print(f"\nDownloaded video: {save_path}")
                    return True
                else:
                    print(f"HTTP Error {response.status_code} for {url}")
        except Exception as e:
            print(f"Error downloading video {url}: {str(e)[:200]}...")
        return False
    
    def download_file(self, url, file_type):
        """Download and save a file with proper content type handling"""
        # First check if this is a video platform URL
        platform = self.is_video_platform_url(url)
        if platform and file_type == 'video':
            return self.download_video_from_platform(url, platform)
        
        try:
            headers = {'User-Agent': self.user_agent}
            response = self.session.get(url, headers=headers, stream=True, timeout=10)
            if response.status_code != 200:
                return False
                
            # Determine file extension from content type or URL
            content_type = response.headers.get('content-type', '').split(';')[0]
            ext = mimetypes.guess_extension(content_type) or Path(urlparse(url).path).suffix or '.bin'
            
            # Create filename
            filename = self.sanitize_filename(Path(urlparse(url).path).name)
            if not filename or filename == ext:
                filename = f"file_{int(time.time())}{ext}"
            elif not filename.endswith(ext):
                filename += ext
            
            # Determine save path based on file type
            if file_type == 'image':
                save_path = f"{self.output_dir}/images/{filename}"
            elif file_type == 'audio':
                save_path = f"{self.output_dir}/audio/{filename}"
            elif file_type == 'video':
                save_path = f"{self.output_dir}/videos/{filename}"
            else:
                save_path = f"{self.output_dir}/other/{filename}"
            
            # Save file
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"Downloaded: {url} -> {save_path}")
            return True
        except Exception as e:
            print(f"Error downloading {url}: {str(e)[:200]}...")
        return False
    
    def extract_text(self, url, soup):
        """Extract and save text content from page"""
        try:
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'iframe', 'noscript']):
                element.decompose()
            
            # Get text with paragraph spacing
            text = '\n\n'.join(p.get_text().strip() for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']) if p.get_text().strip())
            
            # Create filename
            filename = self.sanitize_filename(Path(urlparse(url).path).name or "index") + ".txt"
            save_path = f"{self.output_dir}/text/{filename}"
            
            # Save text
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(f"URL: {url}\n\n")
                f.write(text)
            print(f"Saved text: {save_path}")
            return True
        except Exception as e:
            print(f"Error saving text from {url}: {str(e)[:200]}...")
        return False
    
    def scrape_page(self, url):
        """Scrape all media from a single page"""
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code != 200:
                return False
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract and save text
            self.extract_text(url, soup)
            
            # Find and download all media
            media_tags = {
                'image': ['img', 'image', 'meta[property="og:image"]'],
                'audio': ['audio', 'source[type^="audio/"]', 'meta[property="og:audio"]'],
                'video': ['video', 'source[type^="video/"]', 'iframe', 'meta[property="og:video"]']
            }
            
            for media_type, selectors in media_tags.items():
                for selector in selectors:
                    for tag in soup.select(selector):
                        src = tag.get('src') or tag.get('href') or tag.get('data-src') or tag.get('content')
                        if src:
                            media_url = urljoin(url, src)
                            if self.is_valid_url(media_url):
                                if tag.name == 'iframe' and media_type == 'video':
                                    platform = self.is_video_platform_url(media_url)
                                    if platform:
                                        self.download_video_from_platform(media_url, platform)
                                else:
                                    self.download_file(media_url, media_type)
            
            # Find other downloadable files
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                if any(href.lower().endswith(ext) for ext in ['.pdf', '.doc', '.docx', '.xls', '.ppt']):
                    file_url = urljoin(url, href)
                    if self.is_valid_url(file_url):
                        self.download_file(file_url, 'other')
            
            # Find video links in page text
            for pattern in [p for sublist in self.video_platforms.values() for p in sublist]:
                for match in re.finditer(pattern, response.text):
                    video_url = match.group(0)
                    if video_url.startswith('//'):
                        video_url = 'https:' + video_url
                    if self.is_valid_url(video_url):
                        platform = self.is_video_platform_url(video_url)
                        if platform:
                            self.download_video_from_platform(video_url, platform)
            
            return True
        except Exception as e:
            print(f"Error scraping {url}: {str(e)[:200]}...")
            return False
    
    def get_all_links(self, url):
        """Extract all links from a page"""
        links = set()
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code != 200:
                return links
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                full_url = urljoin(url, href)
                
                # Clean URL
                parsed = urlparse(full_url)
                clean_url = parsed._replace(fragment='', query='').geturl()
                
                if self.is_valid_url(clean_url):
                    links.add(clean_url)
                    
        except Exception as e:
            print(f"Error processing {url}: {str(e)[:200]}...")
            
        return links
    
    def crawl_and_scrape(self, should_stop=None, progress_callback=None):
        """
        Main function to crawl and scrape the website.
        Optionally accepts:
            - should_stop: a callable returning True if the process should stop.
            - progress_callback: a callable accepting (visited_count, max_pages).
        """
        while self.urls_to_visit and len(self.visited_urls) < self.max_pages:
            if should_stop and should_stop():
                print("Webscraping stopped by user.")
                break

            current_url = self.urls_to_visit.popleft()
            
            if current_url in self.visited_urls:
                continue
                
            print(f"\nProcessing: {current_url}")
            
            self.visited_urls.add(current_url)
            
            # Scrape the current page
            self.scrape_page(current_url)
            
            # Get new links to visit
            new_links = self.get_all_links(current_url)
            for link in new_links:
                if link not in self.visited_urls and link not in self.urls_to_visit:
                    self.urls_to_visit.append(link)
            
            if progress_callback:
                progress_callback(len(self.visited_urls), self.max_pages)
            
            time.sleep(self.delay)
            
        print(f"\nScraping complete. Visited {len(self.visited_urls)} pages.")
    
        


def main():
    parser = argparse.ArgumentParser(description='Advanced Web Scraper with Video Download')
    parser.add_argument('url', help='Base URL to start scraping from')
    parser.add_argument('--max-pages', type=int, default=50, help='Maximum number of pages to scrape')
    parser.add_argument('--delay', type=float, default=1, help='Delay between requests in seconds')
    parser.add_argument('--output', default='scraped_data', help='Output directory name')
    args = parser.parse_args()
    
    # Check if yt-dlp is installed
    try:
        import yt_dlp as youtube_dl
    except ImportError:
        print("yt-dlp not found. Installing it with pip...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'yt-dlp'])
        import yt_dlp as youtube_dl
    
    scraper = AdvancedWebScraper(
        base_url=args.url,
        max_pages=args.max_pages,
        delay=args.delay,
        output_dir=args.output
    )
    
    scraper.crawl_and_scrape()
    print(f"\nAll scraped data has been saved to the '{args.output}' directory.")

if __name__ == '__main__':
    main()