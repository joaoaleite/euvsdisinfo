import os
from utils import load_cache, load_tocrawl, crawl
from multiprocessing import Pool

api_token = os.getenv("DIFFBOT_API_KEY")
tocrawl_df = load_tocrawl()

CRAWL_CACHE_PATH = "data/caches/cache.json"
cache = load_cache(CRAWL_CACHE_PATH)
print("Cached:", len(cache))

if __name__ == "__main__":
    num_processes = 16
    chunk_size = len(tocrawl_df) // num_processes
    chunks = [tocrawl_df[i : i + chunk_size] for i in range(0, len(tocrawl_df), chunk_size)]

    with Pool(num_processes) as pool:
        results = pool.starmap(crawl, [(chunk, api_token, CRAWL_CACHE_PATH, True) for chunk in chunks])
