# %%
import requests
from urllib.parse import urlencode
import pandas as pd
import json
from tqdm import tqdm
import os
import hashlib

# %%
def load_cache(p):
    cache = []
    if os.path.exists(p):
        with open(p, "r") as f:
            for i, line in enumerate(f):
                try:
                    cache.append(json.loads(line))
                except:
                    print("Wrong formatting at line", i+1)

    return cache

def dump_cache(line, p):
    with open(p, "a") as f:
        f.write(json.dumps(line)+"\n")

# %%
raw_df = pd.read_csv("raw_data/euvsdisinfo_raw.csv")

# %%
remove_domains = [
    "euvsdisinfo",
    ".pdf"
    "blog"
    "facebook",
    "twitter",
    "instagram",
    "fake",
    "bellingcat",
    "youtube",
    "medium",
    "google",
    "wikipedia",
    "page-not-found",
    "bit.ly",
    "opinion"
]

# %%
api_token = os.getenv("DIFFBOT_API_KEY")

# %%
raw_df = raw_df.drop_duplicates(["article_url"]).reset_index(drop=True)

# %%
# Uncomment to select a specific language to collect
# notna_idxs = raw_df["languages"].notna()
# raw_df = raw_df[(notna_idxs) & (raw_df.loc[notna_idxs, "languages"].apply(lambda x: "English" in x.split(",")))] # english articles
raw_df = raw_df.dropna().reset_index(drop=True)
raw_df = raw_df[["debunk_id", "article_id", "article_url", "article_archive_url", "debunk_date"]]

# %%
CRAWL_CACHE_PATH = "crawled_data/positive_cache.json"
cache = load_cache(CRAWL_CACHE_PATH)
print("Cached:", len(cache))

# %%
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
for row in tqdm(raw_df.itertuples(), total=len(raw_df)):
    try:
        if row.article_id in [c["article_id"] for c in load_cache(CRAWL_CACHE_PATH)]:
            continue

        url = row.article_url
        params = urlencode({'token': api_token, 'url': url, "timeout": 1000*60})
        diffbot_url = f'https://api.diffbot.com/v3/article?{params}'
        
        # Make the GET request to the Diffbot Article API
        diffbot_response = requests.get(diffbot_url, headers=headers)
        if diffbot_response.status_code == 200:
            diffbot_data = json.loads(diffbot_response.content)

            if "objects" in diffbot_data.keys() and diffbot_data['objects'][0]['text'] == "":  # get from archieved file
                url = row.article_archive_url
                params = urlencode({'token': api_token, 'url': url, "timeout": 1000*60})
                diffbot_url = f'https://api.diffbot.com/v3/article?{params}'
                
                # Make the GET request to the Diffbot Article API
                diffbot_response = requests.get(diffbot_url, headers=headers)
                
                if diffbot_response.status_code == 200:
                    diffbot_data = json.loads(diffbot_response.content)
                    
            if "error" in diffbot_data.keys():
                continue

            article_text = diffbot_data['objects'][0]['text']
            text_md5 = hashlib.md5(article_text.encode("utf-8")).hexdigest()
            article_lang = diffbot_data['objects'][0].get('humanLanguage')
            article_type = diffbot_data['objects'][0].get('type')
            article_title = diffbot_data['objects'][0].get('title')
            article_publisher = diffbot_data['objects'][0].get('siteName')
            article_author = diffbot_data['objects'][0].get('author')
            article_date = diffbot_data['objects'][0].get('estimatedDate')
            article_country = diffbot_data['objects'][0].get('publisherCountry')
            named_entities = None
            if diffbot_data['objects'][0].get("tags") is not None:
                named_entities = ",".join([d["label"] for d in diffbot_data['objects'][0]["tags"]])

            d = {
                "md5": text_md5,
                "url": url,
                "text": article_text,
                "detected_language": article_lang,
                "title": article_title,
                "publisher": article_publisher,
                "author": article_author,
                "debunk_id": row.debunk_id,
                "debunk_date": row.debunk_date,
                "article_id": row.article_id,
                "article_country": article_country,
                "article_named_entities": named_entities
            }

            dump_cache(d, CRAWL_CACHE_PATH)
            print(d)
    except Exception as e:
        print("ERROR:", e)
        continue
