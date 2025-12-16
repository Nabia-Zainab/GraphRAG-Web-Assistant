import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from typing import List, Set
from urllib.parse import urljoin, urlparse
import time

class WebLoader:
    def __init__(self, url: str, max_depth: int = 2, max_pages: int = 20):
        self.start_url = url
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.visited: Set[str] = set()
        self.base_domain = urlparse(url).netloc

    def is_same_domain(self, url: str) -> bool:
        return urlparse(url).netloc == self.base_domain

    def clean_text(self, soup: BeautifulSoup) -> str:
        # Remove scripts and styles
        for script in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            script.decompose()
        
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        return '\n'.join(chunk for chunk in chunks if chunk)

    def scrape_page(self, url: str) -> List[Document]:
        if url in self.visited or len(self.visited) >= self.max_pages:
            return []
        
        print(f"Scraping: {url}")
        self.visited.add(url)
        
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            text = self.clean_text(soup)
            title = soup.title.string if soup.title else "No Title"
            
            docs = [Document(page_content=text, metadata={"source": url, "title": title})]
            
            # Find all links
            links = []
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                full_url = urljoin(url, href)
                # Remove fragment
                full_url = full_url.split('#')[0]
                
                if self.is_same_domain(full_url) and full_url not in self.visited:
                    links.append(full_url)
            
            return docs, links
            
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return [], []

    def load(self) -> List[Document]:
        """
        Recursively fetches content starting from the base URL.
        """
        all_docs = []
        queue = [(self.start_url, 0)] # (url, depth)
        
        while queue and len(self.visited) < self.max_pages:
            current_url, current_depth = queue.pop(0)
            
            if current_depth > self.max_depth:
                continue
                
            page_docs, new_links = self.scrape_page(current_url)
            all_docs.extend(page_docs)
            
            # Add new links to queue if depth allows
            if current_depth < self.max_depth:
                for link in new_links:
                    if link not in self.visited:
                        queue.append((link, current_depth + 1))
            
            # Be polite
            time.sleep(0.5)
            
        return all_docs

if __name__ == "__main__":
    # Test
    url = "https://example.com"
    loader = WebLoader(url, max_depth=1, max_pages=5)
    docs = loader.load()
    print(f"Loaded {len(docs)} docs from {len(loader.visited)} pages")
