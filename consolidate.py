# %%
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

# %%
crawled_df = pd.DataFrame(load_cache(CACHE_PATH))
crawled_df.isna().sum()

debunk_df = load_debunks()
debunk_df = debunk_df.rename({"title": "debunk_title"}, axis=1)
crawled_df = crawled_df.merge(debunk_df, on="debunk_id")
crawled_df["keywords"] = crawled_df["details"].apply(lambda x: x.get("keywords"))
crawled_df["publisher"] = crawled_df["publisher"].str.casefold()
crawled_df["disproof"] = crawled_df["disproof"].apply(html_to_text)  # convert HTML to text

crawled_df = normalise_domains(crawled_df)  # normalise most common domains (e.g. rt.ru, rt.com, rt.it -> rt)

crawled_df = crawled_df.drop_duplicates(subset=["id"])
crawled_df = crawled_df.drop_duplicates(subset=["text"])

# remove articles with less than 700 characters
condition = crawled_df["text"].str.len() > 700
df_check = crawled_df[condition]

# %%
# remove non-news articles (orgs, fact-checkers, etc.)
domains_df = pd.read_csv("data/annotated_domains.csv")
domains_df = domains_df[domains_df["type"] == "News Outlet"].reset_index(drop=True)
crawled_df = crawled_df[crawled_df["publisher"].isin(domains_df["publisher"])].reset_index(drop=True)

crawled_df = crawled_df[
    [
        "debunk_id",
        "debunk_title",
        "summary",
        "disproof",
        "keywords",
        "id",
        "title",
        "publisher",
        "url",
        "text",
        "detected_language",
        "date",
        "class",
    ]
]
consolidated_df = crawled_df.rename(
    {
        "detected_language": "article_language",
        "id": "article_id",
        "title": "article_title",
        "url": "article_url",
        "text": "article_text",
        "date": "article_date",
        "publisher": "article_publisher",
    },
    axis=1,
)

consolidated_df.to_csv("data/euvsdisinfo.csv", index=False)
