import pandas as pd
import os
import json
from bs4 import BeautifulSoup
import uuid
import requests
from urllib.parse import urlparse


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
