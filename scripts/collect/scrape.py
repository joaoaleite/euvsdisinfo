import os
from utils import load_cache, load_tocrawl, dump_cache
from tqdm import tqdm
from multiprocessing import Pool
from urllib.parse import urlencode
import json
import requests

api_token = os.getenv("DIFFBOT_API_KEY")
tocrawl_df = load_tocrawl()

CRAWL_CACHE_PATH = "data/caches/cache.json"
cache = load_cache(CRAWL_CACHE_PATH)
print("Cached:", len(cache))


def crawl(tocrawl_df, api_token, cache_path, verbose=False):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110"
            " Safari/537.36"
        )
    }
    failed = 0
    for row in tqdm(tocrawl_df.itertuples(), total=len(tocrawl_df)):
        try:
            if row.id in [c["id"] for c in load_cache(cache_path)]:
                continue

            timeout = 1000 * 60  # 1 minute
            url = row.publication_url
            params = urlencode({"token": api_token, "url": url, "timeout": timeout})
            diffbot_url = f"https://api.diffbot.com/v3/article?{params}"

            # Make the GET request to the Diffbot Article API
            diffbot_response = requests.get(diffbot_url, headers=headers)
            if diffbot_response.status_code == 200:
                diffbot_data = json.loads(diffbot_response.content)

                # if failed, try to get it from archive.org
                if "objects" in diffbot_data.keys() and diffbot_data["objects"][0]["text"] == "":
                    url_archieve = row.archive_url
                    params = urlencode({"token": api_token, "url": url_archieve, "timeout": timeout})
                    diffbot_url = f"https://api.diffbot.com/v3/article?{params}"

                    # Make the GET request to the Diffbot Article API
                    diffbot_response = requests.get(diffbot_url, headers=headers)

                    if diffbot_response.status_code == 200:
                        diffbot_data = json.loads(diffbot_response.content)

                if "error" in diffbot_data.keys():
                    failed += 1
                    continue

                article_text = diffbot_data["objects"][0]["text"]
                article_lang = diffbot_data["objects"][0].get("humanLanguage")
                article_title = diffbot_data["objects"][0].get("title")
                article_publisher = diffbot_data["objects"][0].get("siteName")
                article_author = diffbot_data["objects"][0].get("author")
                article_date = diffbot_data["objects"][0].get("estimatedDate")
                article_country = diffbot_data["objects"][0].get("publisherCountry")
                named_entities = None
                if diffbot_data["objects"][0].get("tags") is not None:
                    named_entities = ",".join([d["label"] for d in diffbot_data["objects"][0]["tags"]])

                d = {
                    "id": row.id,
                    "url": url,
                    "text": article_text,
                    "detected_language": article_lang,
                    "title": article_title,
                    "publisher": article_publisher,
                    "author": article_author,
                    "debunk_id": row.debunk_id,
                    "date": article_date,
                    "article_country": article_country,
                    "article_named_entities": named_entities,
                    "class": row.label,
                    "domain_name": row.domain_name,
                }

                dump_cache(d, cache_path)
                if verbose:
                    print(d)

        except Exception as e:
            print("ERROR:", e)
            continue
    print("Finished crawling. Failed:", failed)


if __name__ == "__main__":
    num_processes = 16
    chunk_size = len(tocrawl_df) // num_processes
    chunks = [tocrawl_df[i : i + chunk_size] for i in range(0, len(tocrawl_df), chunk_size)]

    with Pool(num_processes) as pool:
        results = pool.starmap(crawl, [(chunk, api_token, CRAWL_CACHE_PATH, True) for chunk in chunks])
