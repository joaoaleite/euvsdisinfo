# %%
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from urllib.parse import urlparse
import hashlib

# %%
def extract_domain(row):
    domain = row.url
    if "app.cappture" not in row.url:
        parsed_url = urlparse(row.url)
        domain = parsed_url.netloc

    return domain

# %%
df_debunk = pd.read_csv("raw_data/euvsdisinfo_raw.csv")

# %%
df = pd.read_json("crawled_data/positive_cache.json", lines=True)
if "debunk_id" in df.columns:
    df = df.drop(["debunk_id"], axis=1)
df = df[df["detected_language"] == "en"]
df = df[~df["title"].str.startswith("Cappture: No more dead links")]
# df = df[~df["article_domain"].str.casefold().str.contains("youtube")]
df = df[df["text"].str.len() > 1000]
df.loc[:, "article_domain"] = df.apply(lambda x: extract_domain(x), axis=1)
df_merged = df_debunk.merge(df, on="article_id")
df = df.reset_index(drop=True)
df["debunk_id"] = df_merged["debunk_id"]
df["keywords"] = df_merged["keywords"]

df_pos = df

# %%
df = pd.read_json("crawled_data/negative_cache.json", lines=True)
df["article_domain"] = None
df = df[df["detected_language"] == "en"]
df.loc[:, "article_domain"] = df.apply(lambda x: extract_domain(x), axis=1)
df_neg = df
df_merged["url_disproof"] = df_merged["urls_cited_in_disproof"].str.split(",")
df_exploded = df_merged.explode("url_disproof")
df_join = df_exploded[["debunk_id", "debunk_date_x", "url_disproof", "keywords"]]
df_join["article_id"] = df_join["url_disproof"].apply(lambda x: hashlib.md5(x.encode("utf-8")).hexdigest())
df_join = df_join.drop_duplicates(["article_id"]).reset_index(drop=True)
df_neg = df_neg.merge(df_join, on="article_id")
df_neg["debunk_date"] = df_neg["debunk_date_x"]
df_neg = df_neg[df_pos.columns]

# %%
df_neg["objective"] = 0
df_pos["objective"] = 1

# %%
assert all(df_neg.columns == df_pos.columns)

# %%
df = pd.concat([df_neg, df_pos])

# %%
df.to_csv("data/consolidated.csv", index=False)


