# %%
from utils import load_cache, load_debunks, normalise_domains, language_codes, get_language_distributions_table
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd
import os
from polyglot.detect import Detector
import icu


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
    for _, row in crawled_df.iterrows():
        try:
            pred_lang = Detector(row.article_text, quiet=True)
            reliable = pred_lang.reliable
            lang_name = pred_lang.language.name
        except:
            continue

        if reliable and row.language != lang_name:
            row.language = lang_name

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
    condition = crawled_df["text"].str.len() > 700
    crawled_df = crawled_df[condition]

    crawled_df = filter_domains(crawled_df)

    crawled_df = crawled_df[
        [
            "debunk_id",
            # "debunk_title",
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

    # some more specific filtering rules that were empirically found
    consolidated_df.loc[consolidated_df["article_title"].str.contains("Cappture"), "article_title"] = (
        ""  # remove noisy titles
    )
    consolidated_df = consolidated_df[
        ~consolidated_df["article_title"].str.casefold().str.contains(r"\b404\b")
    ]  # remove 404 request articles
    consolidated_df = consolidated_df[
        ~consolidated_df["article_title"].str.casefold().str.contains(r"\berror\b")
    ]  # remove error articles
    consolidated_df = consolidated_df[~consolidated_df["article_publisher"].str.casefold().str.contains("google")]
    consolidated_df = consolidated_df[~consolidated_df["article_publisher"].str.casefold().str.contains("tiktok")]
    consolidated_df = consolidated_df[~consolidated_df["article_publisher"].str.casefold().str.contains("instagram")]
    consolidated_df = consolidated_df[
        ~consolidated_df["article_publisher"].str.casefold().str.contains("apple podcasts")
    ]

    consolidated_df = consolidated_df[consolidated_df["article_text"].str.len() > 1]  # remove empty strings

    # %%
    # remove highly imbalanced languages and infrequent languages
    distributions_df = get_language_distributions_table(consolidated_df)
    distributions_df = distributions_df[
        (distributions_df["disinformation"] / distributions_df["total"] < 0.95)
        & (distributions_df["disinformation"] / distributions_df["total"] > 0.05)
        & (distributions_df["total"] > 25)
    ]

    selected_languages = distributions_df.index.tolist()
    # top_languages = crawled_df["language"].value_counts().head(15).index.to_list()
    consolidated_df = consolidated_df[consolidated_df["article_language"].isin(selected_languages)].reset_index(
        drop=True
    )

    consolidated_df = consolidated_df.dropna(subset=["article_text"]).reset_index(drop=True)

    consolidated_df.to_csv("data/euvsdisinfo.csv", index=False)

# %%
