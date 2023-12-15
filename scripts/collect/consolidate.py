# %%
from utils import load_cache, load_debunks, normalise_domains, language_codes, get_language_distributions_table
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd
import os
from polyglot.detect import Detector
from polyglot.detect.base import logger as polyglot_logger
from pycld2 import error as pycld2_error
from nltk.tokenize import sent_tokenize
import string

polyglot_logger.setLevel("ERROR")


def html_to_text(html_string):
    soup = BeautifulSoup(html_string, "html.parser")
    text = soup.get_text()
    return text


def parse_date(date_string):
    for fmt in ["%Y-%m-%d", "%d/%m/%Y"]:
        try:
            return datetime.strptime(date_string, fmt)
        except ValueError:
            continue
    raise ValueError(f"No valid format found for date: {date_string}")


def normalise_dates(crawled_df):
    crawled_df.loc[crawled_df["debunk_date"].notna(), "debunk_date"] = crawled_df.loc[
        crawled_df["debunk_date"].notna(), "debunk_date"
    ].apply(lambda x: parse_date(x).strftime("%d-%m-%Y"))
    crawled_df["published_date"] = crawled_df["date"]
    crawled_df.loc[crawled_df["published_date"].notna(), "published_date"] = crawled_df.loc[
        crawled_df["published_date"].notna(), "published_date"
    ].apply(lambda x: datetime.strptime(x, "%a, %d %b %Y %H:%M:%S %Z").strftime("%d-%m-%Y"))

    return crawled_df


def remove_imbalanced_languages(df, threshold=0.95, min_articles=25):
    # remove highly imbalanced languages and infrequent languages
    distributions_df = get_language_distributions_table(df)
    distributions_df = distributions_df[
        (distributions_df["disinformation"] / distributions_df["total"] < threshold)
        & (distributions_df["disinformation"] / distributions_df["total"] > 1 - threshold)
        & (distributions_df["total"] > min_articles)
    ]

    selected_languages = distributions_df.index.tolist()
    df = df[df["article_language"].isin(selected_languages)].reset_index(drop=True)

    return df


def normalise_language(crawled_df):
    # set the language as the one indicated in the debunking article if they only mentioned one language.
    # if they mentioned more, we cannot now which language refers to which url.
    # In this case, we use the detected language from Diffbot.
    # The same applies for supporting articles, since they don't spcify their language in the debunking article.
    crawled_df["language"] = (
        crawled_df["details"].apply(lambda x: x.get("articleLanguages")).fillna("").apply(lambda x: x.split(","))
    )
    crawled_df["language"] = crawled_df.apply(
        lambda x: x["language"][0] if len(x["language"]) == 1 else x["detected_language"], axis=1
    )
    crawled_df.loc[crawled_df["class"] == "support", "language"] = crawled_df.loc[crawled_df["class"] == "support"][
        "detected_language"
    ]
    crawled_df.loc[crawled_df["language"].str.len() == 2, "language"] = crawled_df[
        crawled_df["language"].str.len() == 2
    ]["language"].apply(lambda x: language_codes[x])

    # Check if the detected language is correct. Sometimes, encoding errors happen with Polyglot. It returns a set of
    # languages with probabilities. We assign only the top language in cases when the prediction is reliable
    # and where the URL is active

    new_lang = []
    for _, row in crawled_df.iterrows():
        try:
            pred_lang = Detector(row.text, quiet=True)
        except pycld2_error:
            new_lang.append(None)
            continue

        reliable = pred_lang.reliable
        lang_name = pred_lang.language.name

        if reliable and row.language != lang_name:
            new_lang.append(lang_name)
        else:
            new_lang.append(row.language)

    crawled_df["language"] = new_lang
    return crawled_df


def filter_domains(crawled_df):
    crawled_df = normalise_domains(crawled_df)  # normalise most common domains (e.g. rt.ru, rt.com, rt.it -> rt)

    # remove supporting articles whose domain ends with .org
    # crawled_df = crawled_df[~((crawled_df["domain_name"].str.endswith(".org")) & (crawled_df["class"] == "support"))]

    # remove non-news articles (orgs, fact-checkers, etc.)
    domains_df = pd.read_csv("data/annotated_domains.csv")

    # Remove unwanted publishers/domains
    domains_df = domains_df[domains_df["type"] != "News Outlet"].reset_index(drop=True)
    for row in domains_df.itertuples():
        publisher = row.publisher.lower()
        domain_name = row.domain_name.lower()

        if len(publisher) > 4:  # avoid small names matching too many things
            if "." in publisher:
                publisher = publisher.split(".")[0]

        crawled_df = crawled_df[~crawled_df["publisher"].str.casefold().str.contains(publisher)]
        crawled_df = crawled_df[~crawled_df["publisher"].str.casefold().str.contains(domain_name)]
        crawled_df = crawled_df[~crawled_df["domain_name"].str.casefold().str.contains(domain_name)]
        crawled_df = crawled_df[~crawled_df["domain_name"].str.casefold().str.contains(publisher)]

    # crawled_df = crawled_df[
    #     (~crawled_df["publisher"].str.casefold().str.contains(domains_df["publisher"]))
    #     & ~(crawled_df["publisher"].str.casefold().str.contains(domains_df["domain_name"]))
    #     & ~(crawled_df["domain_name"].str.casefold().str.contains(domains_df["publisher"]))
    #     & ~(crawled_df["domain_name"].str.casefold().str.contains(domains_df["domain_name"]))
    # ]

    # domains_df = domains_df[domains_df["type"] == "News Outlet"].reset_index(drop=True)
    # crawled_df = crawled_df[
    #     (crawled_df["publisher"].isin(domains_df["publisher"]))
    #     | (crawled_df["publisher"].isin(domains_df["domain_name"]))
    #     | (crawled_df["domain_name"].isin(domains_df["publisher"]))
    #     | (crawled_df["domain_name"].isin(domains_df["domain_name"]))
    # ]

    # swap the publisher for the domain if the publisher is cappture.cc
    crawled_df.loc[crawled_df["publisher"].str.casefold().str.contains("cappture.cc"), "publisher"] = crawled_df.loc[
        crawled_df["publisher"].str.casefold().str.contains("cappture.cc"), "domain_name"
    ]

    # remove articles mentioned in the debunking text from publishers that have been flagged with disinformation content
    publishers_with_both_classes = set(
        crawled_df[crawled_df["class"] == "disinformation"]["publisher"].unique()
    ).intersection(set(crawled_df[crawled_df["class"] == "support"]["publisher"].unique()))

    crawled_df = crawled_df.drop(
        index=crawled_df[
            (crawled_df["publisher"].isin(publishers_with_both_classes)) & (crawled_df["class"] != "disinformation")
        ].index
    ).reset_index(drop=True)

    return crawled_df


def custom_rules(df):
    # some more specific filtering rules that were empirically found
    df.loc[df["article_title"].str.contains("Cappture"), "article_title"] = ""  # remove archive titles
    df = df[~df["article_title"].str.casefold().str.contains(r"\b404\b")]  # remove 404 request articles
    df = df[~df["article_title"].str.casefold().str.contains(r"\berror\b")]
    df = df[~df["article_publisher"].str.casefold().str.contains("google")]
    df = df[~df["article_publisher"].str.casefold().str.contains("tiktok")]
    df = df[~df["article_publisher"].str.casefold().str.contains("instagram")]
    df = df[~df["article_publisher"].str.casefold().str.contains("apple podcasts")]

    df = df[df["article_text"].str.len() > 1]  # remove empty strings
    df = df.dropna(subset=["article_text"]).reset_index(drop=True)
    df = df[~df["article_language"].isin(["", "unknown"])].dropna(subset="article_language").reset_index(drop=True)

    return df


def filter_by_ngram(df):
    def remove_punctuation(input_string):
        # Using string.punctuation to get a string of all ASCII punctuation characters
        translator = str.maketrans("", "", string.punctuation)

        # Removing punctuation using translate method
        result_string = input_string.translate(translator)

        return result_string

    filter_rules = [
        "recurring prokremlin disinformation narrative",
        "prokremlin disinformation narrative about",
        "disinformation narrative about the",
        "see other examples of",
        "a recurring prokremlin disinformation",
        "this is a recurring",
        "disinformation cases alleging that",
        "similar cases claiming that",
        "prokremlin disinformation narratives about",
        "recurring prokremlin disinformation narratives",
        "read more about the",
        "read similar cases claiming",
        "is a recurring prokremlin",
        "other examples of similar",
        "recurring prokremlin narrative about",
        "a recurring prokremlin narrative",
        "a recurring disinformation narrative",
        "earlier disinformation cases alleging",
        "see earlier disinformation cases",
        "disinformation narratives about the",
        "recurring prokremlin disinformation",
        "prokremlin disinformation narrative",
        "disinformation narrative about",
        "a recurring prokremlin",
        "see other examples",
        "prokremlin disinformation narratives",
        "recurring prokremlin narrative",
        "other examples of",
        "disinformation narratives about",
        "is a recurring",
    ]

    # tokenize the disproof text into sentences
    debunk_df = load_debunks()
    sentences = debunk_df["disproof"].apply(sent_tokenize).explode().tolist()

    filtered_sentences = []
    for sentence in sentences:
        clean_sentence = remove_punctuation(html_to_text(sentence.lower()))
        for rule in filter_rules:
            if rule in clean_sentence:
                filtered_sentences.append(sentence)
                break

    urls = []
    urls = [BeautifulSoup(sentence, "html.parser").find_all("a") for sentence in filtered_sentences]
    urls = [url.get("href") for lurl in urls for url in lurl]

    matched_urls = df[df["article_url"].isin(urls)]

    trustworthy = [
        "bbc",
        "reuters",
        "the guardian",
        "dw.com",
        "radiofreeeurope/radioliberty",
        "washington post",
        "cnn",
        "ap news",
        "euronews",
        "politico",
        "npr",
        "new york times",
        "france 24",
        "polygraph.info",
    ]

    ids_to_remove = matched_urls[~matched_urls["article_publisher"].isin(trustworthy)]["article_id"].tolist()
    df = df[~df["article_id"].isin(ids_to_remove)].reset_index(drop=True)

    return df


if __name__ == "__main__":
    CACHE_PATH = os.path.join("data", "caches", "cache.json")
    assert os.path.exists(CACHE_PATH)
    assert os.path.exists("data/annotated_domains.csv")

    crawled_df = pd.DataFrame(load_cache(CACHE_PATH))
    crawled_df.isna().sum()

    # merge debunks with crawled articles
    debunk_df = load_debunks()
    debunk_df = debunk_df.rename({"title": "debunk_title"}, axis=1)
    crawled_df = crawled_df.merge(debunk_df, on="debunk_id")
    crawled_df["keywords"] = crawled_df["details"].apply(lambda x: x.get("keywords"))
    crawled_df["debunk_date"] = crawled_df["details"].apply(lambda x: x.get("dateOfPublication"))
    crawled_df["disproof"] = crawled_df["disproof"].apply(html_to_text)  # convert HTML to text
    crawled_df["publisher"] = crawled_df["publisher"].str.casefold()

    crawled_df = normalise_dates(crawled_df)
    crawled_df = normalise_language(crawled_df)

    crawled_df = crawled_df.drop_duplicates(subset=["id"])
    crawled_df = crawled_df.drop_duplicates(subset=["text"])

    # remove articles with less than 700 characters
    print("Initial size:", len(crawled_df))

    size = len(crawled_df)
    condition = crawled_df["text"].str.len() > 700
    crawled_df = crawled_df[condition]
    print("'Remove short articles':", size - len(crawled_df))

    size = len(crawled_df)
    crawled_df = filter_domains(crawled_df)
    print("'Filter Domains':", size - len(crawled_df))

    crawled_df = crawled_df[
        [
            "debunk_id",
            "debunk_title",
            # "summary",
            # "disproof",
            "keywords",
            "id",
            "title",
            "publisher",
            "domain_name",
            "url",
            "text",
            "language",
            "debunk_date",
            "published_date",
            "class",
        ]
    ]
    consolidated_df = crawled_df.rename(
        {
            "language": "article_language",
            "id": "article_id",
            "title": "article_title",
            "url": "article_url",
            "text": "article_text",
            "publisher": "article_publisher",
            "domain_name": "article_domain",
        },
        axis=1,
    )

    size = len(consolidated_df)

    custom_rules(consolidated_df)
    print("'Custom rules':", size - len(consolidated_df))

    size = len(consolidated_df)
    consolidated_df = filter_by_ngram(consolidated_df)
    print("'Filter by ngram':", size - len(consolidated_df))

    consolidated_df_full = consolidated_df.copy()
    consolidated_df.to_csv("data/euvsdisinfo_full.csv", index=False)

    size = len(consolidated_df)
    consolidated_df = remove_imbalanced_languages(consolidated_df)
    consolidated_df.to_csv("data/euvsdisinfo.csv", index=False)
    print("'Remove imbalanced languages':", size - len(consolidated_df))

    print("Full size:", len(consolidated_df_full))
    print("Short size:", len(consolidated_df))

# %%
