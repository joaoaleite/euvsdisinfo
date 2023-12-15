import os
from tqdm import tqdm
from multiprocessing import Pool
from urllib.parse import urlencode
import json
import requests
import pandas as pd

api_token = os.getenv("DIFFBOT_API_KEY")


def load_cache(p):
    cache = []
    if os.path.exists(p):
        with open(p, "r") as f:
            for i, line in enumerate(f):
                try:
                    cache.append(json.loads(line))
                except json.decoder.JSONDecodeError:
                    print("Wrong formatting at line", i + 1)

    return cache


def dump_cache(line, p):
    with open(p, "a") as f:
        f.write(json.dumps(line) + "\n")


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
    for i, row in tqdm(tocrawl_df.iterrows(), total=len(tocrawl_df)):
        try:
            if row.article_id in [c["article_id"] for c in load_cache(cache_path)]:
                continue

            timeout = 1000 * 60  # 1 minute
            url = row.article_url
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
                article_title = diffbot_data["objects"][0].get("title")

                d = dict(row)
                d["article_text"] = article_text
                d["article_title"] = article_title

                dump_cache(d, cache_path)
                if verbose:
                    print(d)

        except Exception as e:
            print("ERROR:", e)
            continue
    print("Finished crawling. Failed:", failed)


if __name__ == "__main__":
    df = pd.read_csv("data/euvsdisinfo_base.csv")
    df["article_text"] = None
    df["article_title"] = None
    num_processes = 4
    chunk_size = len(df) // num_processes
    chunks = [df[i : i + chunk_size] for i in range(0, len(df), chunk_size)]

    with Pool(num_processes) as pool:
        results = pool.starmap(crawl, [(chunk, api_token, CRAWL_CACHE_PATH, True) for chunk in chunks])
