# %%
from utils import load_cache, load_debunks, extract_domain
import pandas as pd
import os

CACHE_PATH = os.path.join("data", "caches", "cache.json")
assert os.path.exists(CACHE_PATH)

crawled_df = pd.DataFrame(load_cache(CACHE_PATH))

debunks_df = load_debunks()
debunks_df = debunks_df[["summary", "disproof", "debunk_id"]]
consolidated_df = crawled_df.merge(debunks_df, on="debunk_id")
# %%
consolidated_df["domain_name"] = consolidated_df["url"].apply(lambda x: extract_domain(x))

# %%
consolidated_df[["publisher", "url"]].value_counts()
