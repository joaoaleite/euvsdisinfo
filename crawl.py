import os
from utils import load_cache, load_tocrawl, crawl

api_token = os.getenv("DIFFBOT_API_KEY")
tocrawl_df = load_tocrawl()

CRAWL_CACHE_PATH = "data/caches/positive_cache.json"
cache = load_cache(CRAWL_CACHE_PATH)
print("Cached:", len(cache))

if __name__ == "__main__":
    crawl(tocrawl_df=tocrawl_df, api_token=api_token, cache_path=CRAWL_CACHE_PATH, verbose=True)
