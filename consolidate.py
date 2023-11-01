from utils import load_cache, load_debunks, normalise_domains
from bs4 import BeautifulSoup
import pandas as pd
import os


def html_to_text(html_string):
    soup = BeautifulSoup(html_string, "html.parser")
    text = soup.get_text()
    return text


CACHE_PATH = os.path.join("data", "caches", "cache.json")
assert os.path.exists(CACHE_PATH)

crawled_df = pd.DataFrame(load_cache(CACHE_PATH))
crawled_df.isna().sum()
crawled_df["publisher"] = crawled_df["publisher"].str.casefold()
crawled_df["disproof"] = crawled_df["disproof"].apply(html_to_text)  # convert HTML to text

crawled_df = normalise_domains(crawled_df)  # normalise most common domains (e.g. rt.ru, rt.com, rt.it -> rt)

crawled_df = crawled_df.drop_duplicates(subset=["id"])
crawled_df = crawled_df.drop_duplicates(subset=["text"])

# remove articles with less than 700 characters
condition = crawled_df["text"].str.len() > 700
df_check = crawled_df[condition]

# remove non-news articles (orgs, fact-checkers, etc.)
domains_df = pd.read_csv("data/annotated_domains.csv")
domains_df = domains_df[domains_df["type"] == "News Outlet"].reset_index(drop=True)
crawled_df = crawled_df[crawled_df["publisher"].isin(domains_df["publisher"])].reset_index(drop=True)

crawled_df = crawled_df[
    [
        "debunk_id",
        "title",
        "summary",
        "disproof",
        "id",
        "publisher",
        "url",
        "text",
        "detected_language",
        "date",
        "class",
    ]
]
consolidated_df = crawled_df.rename({"title": "claim", "detected_language": "language"}, axis=1)

consolidated_df.to_csv("data/euvsdisinfo.csv", index=False)
