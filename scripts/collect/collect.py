import os
from tqdm import tqdm
from multiprocessing import Pool
from urllib.parse import urlencode
import json
import requests
import pandas as pd
from loguru import logger
from waybackpy import WaybackMachineCDXServerAPI
from waybackpy.exceptions import NoCDXRecordFound
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Crawl articles from EUvsDisinfo")
    parser.add_argument(
        "--api_token",
        type=str,
        required=True,
        help="Diffbot API token for accessing the Article API",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run the crawler in parallel using multiple processes",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=4,
        help="Number of processes to use for parallel crawling (default: 4)",
    )
    parser.add_argument(
        "--cache_path",
        type=str,
        default="data/cache.json",
        help="Path to the cache file where crawled articles will be stored (default: data/cache.json)",
    )
    parser.add_argument(
        "--tocrawl_path",
        type=str,
        default="data/euvsdisinfo_base.csv",
        help="Path to the CSV file containing articles to crawl (default: data/euvsdisinfo_base.csv)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/euvsdisinfo.csv",
        help="Path to the output CSV file where crawled articles will be saved (default: data/euvsdisinfo.csv)",
    )

    return parser.parse_args()


def load_cache(p):
    cache = []
    if os.path.exists(p):
        with open(p, "r") as f:
            for i, line in enumerate(f):
                try:
                    cache.append(json.loads(line))
                except json.decoder.JSONDecodeError:
                    print("Wrong formatting at line", i + 1)

    else:
        print("Cache file does not exist. Creating a new one.")
        with open(p, "w") as f:
            pass

    return cache


def dump_cache(line, p):
    with open(p, "a") as f:
        f.write(json.dumps(line) + "\n")


def get_archive_url(url, user_agent):
    cdx_api = WaybackMachineCDXServerAPI(url, user_agent)
    try:
        oldest = cdx_api.oldest()
        return oldest.archive_url
    except NoCDXRecordFound:
        return None


def crawl(tocrawl_df, api_token, cache_path):
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
                if (
                    "objects" in diffbot_data.keys()
                    and diffbot_data["objects"][0]["text"] == ""
                ) or "error" in diffbot_data.keys():
                    url_archive = get_archive_url(url, headers["User-Agent"])
                    if not url_archive:
                        failed += 1
                        logger.error(
                            f"Failed to find archive url for ID {row.article_id})\noriginal url: {url}"
                        )
                        continue

                    params = urlencode(
                        {"token": api_token, "url": url_archive, "timeout": timeout}
                    )
                    diffbot_url = f"https://api.diffbot.com/v3/article?{params}"

                    diffbot_response = requests.get(diffbot_url, headers=headers)
                    if diffbot_response.status_code == 200:
                        diffbot_data = json.loads(diffbot_response.content)

                    if "error" in diffbot_data.keys():
                        failed += 1
                        logger.error(
                            f"Diffbot error for ID {row.article_id} ({diffbot_data['error']})\noriginal url: {url}\narchive url: {url_archive}"
                        )
                        continue

                article_text = diffbot_data["objects"][0]["text"]
                article_title = diffbot_data["objects"][0].get("title")

                d = dict(row)
                d["article_text"] = article_text
                d["article_title"] = article_title

                dump_cache(d, cache_path)
                logger.debug(
                    f"Successfully crawled {row.article_id} - {row.article_url}"
                )

        except Exception as e:
            print("ERROR:", e)
            continue

    print("Finished crawling. Failed:", failed)


if __name__ == "__main__":
    args = parse_args()
    api_token = args.api_token
    CRAWL_CACHE_PATH = args.cache_path
    cache = load_cache(CRAWL_CACHE_PATH)
    print("Cached:", len(cache))

    parallel = args.parallel
    if parallel:
        num_processes = args.num_processes

    tocrawl_path = args.tocrawl_path
    output_path = args.output_path

    df = pd.read_csv(tocrawl_path)
    df["article_text"] = None
    df["article_title"] = None

    if parallel:
        chunk_size = len(df) // num_processes
        chunks = [df[i : i + chunk_size] for i in range(0, len(df), chunk_size)]

        with Pool(num_processes) as pool:
            results = pool.starmap(
                crawl, [(chunk, api_token, CRAWL_CACHE_PATH) for chunk in chunks]
            )
    else:
        crawl(df, api_token, CRAWL_CACHE_PATH)

    df = load_cache(CRAWL_CACHE_PATH)
    df = pd.DataFrame(df)
    df.to_csv(output_path, index=False)
