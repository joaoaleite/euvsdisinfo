import pandas as pd
import os
import json
from bs4 import BeautifulSoup
import uuid
import requests
from urllib.parse import urlparse, urlencode
from tqdm import tqdm


def load_debunks():
    """Loads a dataframe containing the debunk articles from EuvsDisinfo."""
    path_raw = os.path.join("data/raw/")
    dfs = []
    for fname in os.listdir(path_raw):
        if fname.endswith(".json"):
            fpath = os.path.join(path_raw, fname)

            df = pd.read_json(fpath)
            df = pd.DataFrame(df["disinfoCases"].tolist())
            dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df.drop_duplicates(subset="id", inplace=True)
    df.reset_index(drop=True, inplace=True)

    df = df.rename({"id": "debunk_id"}, axis=1)
    return df


def get_most_recent_archive(url):
    api_url = f"http://archive.org/wayback/available?url={url}"
    response = requests.get(api_url)
    data = response.json()

    # Check if URL is in Wayback Machine archive
    if "closest" in data["archived_snapshots"]:
        # Get URL of most recent archive
        archive_url = data["archived_snapshots"]["closest"]["url"]
        return archive_url
    else:
        return None


def extract_domain(url):
    try:
        parsed_url = urlparse(url)
    except ValueError:
        return None

    domain = parsed_url.netloc
    domain = domain.replace("www.", "")
    return domain


def load_tocrawl():
    """Loads a dataframe containing the article URLs that will be crawled."""
    raw_df = load_debunks()
    raw_df = raw_df.explode("publishedIn")
    raw_df = raw_df.dropna(subset=["publishedIn"])
    raw_df = raw_df.reset_index(drop=True)
    tocrawl_df_misinfo = pd.concat([pd.DataFrame(raw_df["publishedIn"].tolist())])
    tocrawl_df_misinfo["debunk_id"] = raw_df["debunk_id"]
    tocrawl_df_misinfo["label"] = "misinformation"
    # extracts the domain name even though it already exists, just so that they are consistent with the support articles
    # whose domain names are extracted in this manner.
    tocrawl_df_misinfo["domain_name"] = tocrawl_df_misinfo["publication_url"].apply(lambda x: extract_domain(x))

    urls = raw_df["disproof"].apply(lambda x: BeautifulSoup(x, "html.parser").find_all("a"))
    urls = urls.apply(lambda x: [link.get("href") for link in x])
    tocrawl_df_support = pd.DataFrame()
    tocrawl_df_support["publication_url"] = urls
    tocrawl_df_support["debunk_id"] = raw_df["debunk_id"]
    tocrawl_df_support["label"] = "support"
    tocrawl_df_support = tocrawl_df_support.explode("publication_url")
    tocrawl_df_support = (
        tocrawl_df_support.drop_duplicates(subset=["publication_url"])
        .dropna(subset=["publication_url"])
        .reset_index(drop=True)
    )
    tocrawl_df_support["id"] = tocrawl_df_support["publication_url"].apply(lambda x: uuid.uuid5(uuid.NAMESPACE_URL, x))
    tocrawl_df_support["domain_name"] = tocrawl_df_support["publication_url"].apply(lambda x: extract_domain(x))

    tocrawl_df = pd.concat([tocrawl_df_misinfo, tocrawl_df_support], ignore_index=True)

    return tocrawl_df


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
                    url = row.archive_url
                    params = urlencode({"token": api_token, "url": url, "timeout": timeout})
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
                }

                dump_cache(d, cache_path)
                if verbose:
                    print(d)

        except Exception as e:
            print("ERROR:", e)
            continue
    print("Finished crawling. Failed:", failed)


# %%
# remove_domains = [
#     "euvsdisinfo",
#     ".pdf"
#     "blog"
#     "facebook",
#     "twitter",
#     "instagram",
#     "fake",
#     "bellingcat",
#     "youtube",
#     "medium",
#     "google",
#     "wikipedia",
#     "page-not-found",
#     "bit.ly",
#     "opinion"
# ]
