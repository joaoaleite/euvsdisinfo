import pandas as pd
from urllib.parse import urlencode
import requests
import os
import hashlib
import json
from tqdm import tqdm

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

def unshorten_url(url):
    session = requests.Session()
    resp = session.head(url, allow_redirects=True)

    return resp.url

# %%
df = pd.read_json("crawled_data/positive_cache.json", lines=True)

# df = df[df["detected_language"] == "en"] # Uncomment to collect a specific language
df = df[~df["title"].str.startswith("Cappture: No more dead links")]
# df = df[df["text"].str.len() > 1000]
df = df.reset_index(drop=True)
positive_df = df

# %%
df = pd.read_csv("raw_data/euvsdisinfo_raw.csv")
df_merged = df.merge(positive_df, on="article_id")

# %%
df_merged["url_disproof"] = df_merged["urls_cited_in_disproof"].str.split(",")
df_exploded = df_merged.explode("url_disproof")

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
    "opinion"
]

# %%
raw_df = df_exploded[df_exploded["url_disproof"].apply(lambda x: all(([d not in x for d in remove_domains])))]
raw_df = raw_df.drop_duplicates(["url_disproof"]).reset_index(drop=True)
cols = [
    "debunk_id",
    "url_disproof",
    "debunk_id"
]
raw_df = raw_df[cols]
raw_df["article_id"] = raw_df["url_disproof"].apply(lambda x: hashlib.md5(x.encode("utf-8")).hexdigest())
raw_df

# %%
api_token = os.getenv("DIFFBOT_API_KEY")
CRAWL_CACHE_PATH = "crawled_data/negative_cache.json"
cache = load_cache(CRAWL_CACHE_PATH)
print("Cached:", len(cache))

# %%
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}


for row in tqdm(raw_df.itertuples(), total=len(raw_df)):
    try:
        if row.article_id in [c["article_id"] for c in load_cache(CRAWL_CACHE_PATH)]:
            continue

        if "bit.ly" in row.url_disproof:
            unshortened_url = unshorten_url(row.url_disproof)
        else:
            unshortened_url = row.url_disproof

        params = urlencode({'token': api_token, 'url': unshortened_url, "timeout": 1000*60})
        diffbot_url = f'https://api.diffbot.com/v3/article?{params}'
                
        diffbot_response = requests.get(diffbot_url, headers=headers)
        if ( # bad request on original url, try waybackmachine
            diffbot_response.status_code != 200 or 
            (
                (
                    diffbot_response.status_code == 200 and (
                        ("error" in json.loads(diffbot_response.content)) or
                        ("objects" in json.loads(diffbot_response.content) and json.loads(diffbot_response.content)['objects'][0]['text'] == "")
                    )
                )
            )
        ):
            # Define the endpoint URL and parameters for the Wayback Machine API
            wbm_url = f'https://archive.org/wayback/available?{params}'
            wbm_params = {'output': 'json'}

            # Make the GET request to the Wayback Machine API
            wbm_response = requests.get(wbm_url, headers=headers, params=wbm_params)

            # Check if there is an archived version of the URL available
            if wbm_response.status_code == 200:
                wbm_data = json.loads(wbm_response.content)
                if 'closest' in wbm_data['archived_snapshots']:
                    # Extract the timestamp of the closest archived version of the URL
                    timestamp = wbm_data['archived_snapshots']['closest']['timestamp']
                    
                    # Define the URL of the archived version of the article
                    url_archive = f'http://web.archive.org/web/{timestamp}/{unshortened_url}'
                    # Define the endpoint URL and parameters for the Diffbot Article API
                
                    params = urlencode({'token': api_token, 'url': url_archive, "timeout": 1000*60})
                    diffbot_url = f'https://api.diffbot.com/v3/article?{params}'
                    
                    # Make the GET request to the Diffbot Article API
                    diffbot_response = requests.get(diffbot_url, headers=headers)
        
        # Extract the article text from the Diffbot response
        if diffbot_response.status_code == 200:
            diffbot_data = json.loads(diffbot_response.content)
            if "error" in diffbot_data:
                pass
            else:
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
                    "url": row.url_disproof,
                    "unshortened_url": unshortened_url,
                    "text": article_text,
                    "detected_language": article_lang,
                    "title": article_title,
                    "publisher": article_publisher,
                    "author": article_author,
                    "article_date": article_date,
                    "article_id": row.article_id,
                    "article_country": article_country,
                    "article_named_entities": named_entities
                }

                dump_cache(d, CRAWL_CACHE_PATH)
                print(d)
    except Exception as e:
        print("ERROR:", e)
        continue


