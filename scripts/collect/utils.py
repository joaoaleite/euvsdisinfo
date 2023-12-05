import pandas as pd
import os
import json
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import uuid
import numpy as np
import requests


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
    tocrawl_df_misinfo["label"] = "disinformation"
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
    tocrawl_df_support["id"] = tocrawl_df_support["publication_url"].apply(
        lambda x: str(uuid.uuid5(uuid.NAMESPACE_URL, x))
    )
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


def normalise_domains(crawled_df):
    for publisher in [
        "nato",
        "politico",
        "sputnik",
        "spiegel",
        "bbc",
        "cnn",
        "rbc",
        "reuters",
        "jazeera",
        "риа новости",
        "vesti",
        "rt arabic",
        "kp",
        "телеканал царьград",
        "новости россии, снг и мира - иа regnum",
        "новая газета",
        "украина.ру",
        "российская газета",
        "политнавигатор",
        "rt на русском",
        "rt",
        "телеканал «звезда»",
        "rubaltic",
        "известия",
        "взгляд",
        "baltnews",
        "1tv",
        "موقع نبض",
        "نافذة على العالم",
        "أخبارك.نت",
        "نيوز فور مي",
    ]:
        search = replace = publisher
        if publisher == "риа новости":
            replace = "ria"
        elif publisher == "телеканал царьград":
            replace = "tsargrad"
        elif publisher == "rt на русском" or publisher == "rt arabic":
            replace = "rt"
        elif publisher == "rt":
            search = r"\brt\b"
            replace = "rt"
        elif publisher == "телеканал «звезда»":
            replace = "zvezda"
        elif publisher == "известия":
            replace = "izvestia"
        elif publisher == "взгляд":
            replace = "vzglyad"
        elif publisher == "موقع نبض":
            replace = "nabd"
        elif publisher == "نافذة على العالم":
            replace = "naftha"
        elif publisher == "أخبارك.نت":
            replace = "akhbarak"
        elif publisher == "украина.ру":
            replace = "ukraina.ru"
        elif publisher == "политнавигатор":
            replace = "politnavigator"
        elif publisher == "российская газета":
            replace = "rg"
        elif publisher == "kp":
            search = r"\bkp\b"
            replace = "kp"
        elif publisher == "новая газета":
            replace = "novaya gazeta"
        elif publisher == "новости россии, снг и мира - иа regnum":
            replace = "regnum"
        elif publisher == "نيوز فور مي":
            replace = "newsformy.com"
        elif publisher == "vesti":
            search = r"\bvesti\b"
            replace = "vesti"
        elif publisher == "jazeera":
            replace = "al jazeera"

        crawled_df.loc[
            (crawled_df["publisher"].str.contains(search)) | (crawled_df["domain_name"].str.contains(search)),
            "publisher",
        ] = replace

    return crawled_df


def get_language_distributions_table(df):
    languages = df["article_language"].unique()
    classes = df["class"].unique().tolist()
    total_articles = []
    class_distributions = []

    for language in languages:
        total_articles.append(len(df[df["article_language"] == language]))
        for class_ in classes:
            class_distributions.append(len(df[(df["article_language"] == language) & (df["class"] == class_)]))

    class_distributions = np.array(class_distributions).reshape(len(languages), len(classes))

    distributions_df = pd.DataFrame(
        {
            "total": class_distributions.sum(1),
            "disinformation": class_distributions[:, 1],
            "support": class_distributions[:, 0],
        },
        index=languages,
    ).sort_values("total", ascending=False)

    return distributions_df


language_codes = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "it": "Italian",
    "es": "Spanish",
    "pl": "Polish",
    "cs": "Czech",
    "hu": "Hungarian",
    "ro": "Romanian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "sv": "Swedish",
    "fi": "Finnish",
    "da": "Danish",
    "et": "Estonian",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "bg": "Bulgarian",
    "el": "Greek",
    "sk": "Slovak",
    "sl": "Slovenian",
    "hr": "Croatian",
    "ga": "Irish",
    "mt": "Maltese",
    "cy": "Welsh",
    "gd": "Gaelic",
    "sq": "Albanian",
    "sr": "Serbian",
    "mk": "Macedonian",
    "bs": "Bosnian",
    "is": "Icelandic",
    "no": "Norwegian",
    "uk": "Ukrainian",
    "ru": "Russian",
    "be": "Belarusian",
    "hy": "Armenian",
    "ka": "Georgian",
    "az": "Azerbaijani",
    "uz": "Uzbek",
    "kk": "Kazakh",
    "tr": "Turkish",
    "he": "Hebrew",
    "ar": "Arabic",
    "fa": "Persian",
    "ur": "Urdu",
    "hi": "Hindi",
    "bn": "Bengali",
    "pa": "Punjabi",
    "gu": "Gujarati",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "th": "Thai",
    "lo": "Lao",
    "my": "Burmese",
    "km": "Khmer",
    "af": "Afrikaans",
    "ja": "Japanese",
    "ko": "Korean",
    "id": "Indonesian",
    "ca": "Catalan",
    "zh": "Chinese",
    "tl": "Filipino",
}
