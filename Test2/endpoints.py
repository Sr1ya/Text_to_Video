import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import argparse
from collections import deque
import time

class EndpointScraper:
    def __init__(self, base_url, max_pages=50, delay=1, user_agent=None):
        self.base_url = base_url
        self.domain = urlparse(base_url).netloc
        self.visited_urls = set()
        self.urls_to_visit = deque([base_url])
        self.max_pages = max_pages
        self.delay = delay
        self.user_agent = user_agent or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        self.endpoints = set()
        
    def is_valid_url(self, url):
        parsed = urlparse(url)
        return bool(parsed.netloc) and bool(parsed.scheme) and self.domain in parsed.netloc
    
    def get_all_links(self, url):
        links = set()
        try:
            headers = {'User-Agent': self.user_agent}
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                return links
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                full_url = urljoin(url, href)
                
                # Clean URL by removing fragments and query parameters if needed
                parsed = urlparse(full_url)
                clean_url = parsed._replace(fragment='', query='').geturl()
                
                if self.is_valid_url(clean_url):
                    links.add(clean_url)
                    
        except Exception as e:
            print(f"Error processing {url}: {e}")
            
        return links
    
    def scrape(self):
        while self.urls_to_visit and len(self.visited_urls) < self.max_pages:
            current_url = self.urls_to_visit.popleft()
            
            if current_url in self.visited_urls:
                continue
                
            print(f"Processing: {current_url}")
            
            self.visited_urls.add(current_url)
            links = self.get_all_links(current_url)
            
            for link in links:
                if link not in self.visited_urls and link not in self.urls_to_visit:
                    self.urls_to_visit.append(link)
                    self.endpoints.add(link)
            
            time.sleep(self.delay)  # Be polite with delay between requests
            
        return self.endpoints
    
    def save_to_file(self, filename='endpoints.txt'):
        with open(filename, 'w') as f:
            for endpoint in sorted(self.endpoints):
                f.write(endpoint + '\n')
        print(f"Saved {len(self.endpoints)} endpoints to {filename}")

def main():
    parser = argparse.ArgumentParser(description='Website Endpoint Scraper')
    parser.add_argument('url', help='Base URL to start scraping from')
    parser.add_argument('--max-pages', type=int, default=50, help='Maximum number of pages to scrape')
    parser.add_argument('--delay', type=float, default=1, help='Delay between requests in seconds')
    parser.add_argument('--output', default='endpoints.txt', help='Output file name')
    args = parser.parse_args()
    
    scraper = EndpointScraper(
        base_url=args.url,
        max_pages=args.max_pages,
        delay=args.delay
    )
    
    endpoints = scraper.scrape()
    scraper.save_to_file(args.output)
    
    print(f"\nFound {len(endpoints)} unique endpoints:")
    for endpoint in sorted(endpoints):
        print(endpoint)

if __name__ == '__main__':
    main()