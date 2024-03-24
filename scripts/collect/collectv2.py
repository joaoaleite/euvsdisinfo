import newspaper
import json
import pandas as pd
from tqdm import tqdm
import os
import requests
from waybackpy import WaybackMachineCDXServerAPI
from urllib.parse import urlparse
from multiprocessing import Pool
from newspaper import Config
from time import sleep
import numpy as np

user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:78.0) Gecko/20100101 Firefox/78.0"


CACHE_PATH = "data/caches/cache4.json"
config = Config()
config.browser_user_agent = user_agent
config.request_timeout = 30


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


def extract_domain(url):
    try:
        parsed_url = urlparse(url)
    except ValueError:
        return None

    domain = parsed_url.netloc
    domain = domain.replace("www.", "")

    return domain


def get_original_url(shortened_url):
    try:
        response = requests.head(shortened_url, allow_redirects=True)
    except Exception:
        return shortened_url  # return it back again if it fails
    return response.url


def get_wayback_url(url):
    try:
        cdx_api = WaybackMachineCDXServerAPI(url, "")
        archive_url = cdx_api.oldest().archive_url
    except Exception:
        return None

    return archive_url


def collect_url(url):
    attempts = 0
    while attempts < 6:
        try:
            article = newspaper.Article(url=url, config=config)
            article.download()
            article.parse()
            break
        except newspaper.article.ArticleException:
            sleep(10)
            attempts += 1
            continue
        except Exception:
            article = None
            break

    return article


def crawl(df):
    missing_articles = 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        if row.article_id in [c["article_id"] for c in load_cache(CACHE_PATH)]:
            continue
        url = row["article_url"]
        if "bit.ly" in url:
            url = get_original_url(url)

        archive_url = row.archive_url if row.archive_url is not None else get_wayback_url(url)
        if archive_url is not None:
            article = collect_url(archive_url)

        if "app.cappture.cc" in archive_url and article is None:  # sometimes app.capture.cc does not work
            archive_url = get_wayback_url(url)
            article = collect_url(archive_url)

        # if wayback url is not available, or scraping it returned nothing, try the original url
        if archive_url is None or article is None or article.text is None or article.text == "":
            article = collect_url(url)

        if article.text is not None:
            d = dict(row)
            d["article_text"] = str(article.text)
            d["article_title"] = str(article.title)
            d["article_url"] = str(url)
            d["published_date"] = str(article.publish_date)
            d["domain_name"] = extract_domain(url)
            d["archive_url"] = str(archive_url)
            dump_cache(d, CACHE_PATH)

        print(article.title + "\n\n")
    print(f"Missing articles: {missing_articles}")


if __name__ == "__main__":
    df = pd.read_csv("data/urls.csv")
    df = df.replace({np.nan: None})
    df["article_text"] = None
    df["article_title"] = None
    num_processes = 8
    chunk_size = len(df) // num_processes
    chunks = [df[i : i + chunk_size] for i in range(0, len(df), chunk_size)]

    with Pool(num_processes) as pool:
        results = pool.starmap(crawl, [(chunk,) for chunk in chunks])
    df = pd.DataFrame(load_cache(CACHE_PATH))
    df.dropna(subset=["article_text"], inplace=True)
    df.to_csv("data/euvsdisinforaw.csv", index=False)
