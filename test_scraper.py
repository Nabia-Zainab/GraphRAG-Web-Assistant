from web_loader import WebLoader

def test_scraper():
    print("Testing WebLoader...")
    url = "https://www.example.com"
    loader = WebLoader(url)
    docs = loader.load()
    
    if docs:
        print(f"✅ Success! Loaded doc from {url}")
        print(f"Title: {docs[0].metadata.get('title')}")
        print(f"Content Length: {len(docs[0].page_content)} characters")
        print("Snippet:", docs[0].page_content[:100].replace('\n', ' '))
    else:
        print("❌ Failed to load doc.")

if __name__ == "__main__":
    test_scraper()
