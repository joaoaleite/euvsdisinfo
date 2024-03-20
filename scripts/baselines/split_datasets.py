# %%
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import numpy as np


def get_language_distributions_table(df):
    languages = df["language"].unique()
    classes = df["label"].unique().tolist()
    total_articles = []
    class_distributions = []

    for language in languages:
        total_articles.append(len(df[df["language"] == language]))
        for class_ in classes:
            class_distributions.append(len(df[(df["language"] == language) & (df["label"] == class_)]))

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


def remove_imbalanced_languages(df, threshold=0.95, min_articles=25):
    # remove highly imbalanced languages and infrequent languages
    distributions_df = get_language_distributions_table(df)
    distributions_df = distributions_df[
        (distributions_df["disinformation"] / distributions_df["total"] < threshold)
        & (distributions_df["disinformation"] / distributions_df["total"] > 1 - threshold)
        & (distributions_df["total"] > min_articles)
    ]

    selected_languages = distributions_df.index.tolist()
    df = df[df["language"].isin(selected_languages)].reset_index(drop=True)

    return df


SEED = 42

exp_data_path = os.path.join("data", "experiments")
os.makedirs(exp_data_path, exist_ok=True)

data_path = os.path.join("data")
euvsdisinfo_path = os.path.join(data_path, "euvsdisinfo.csv")
mmcovid_path = os.path.join(data_path, "news_collection.json")
fakecovid_path = os.path.join(data_path, "FakeCovid.csv")

if __name__ == "__main__":
    # EUvsDisinfo
    if os.path.exists(euvsdisinfo_path):
        df = pd.read_csv(euvsdisinfo_path)
        df = df.fillna("")
        df["text"] = df.apply(
            lambda x: x["article_title"] + " " + x["article_text"] if x["article_title"] != "" else x["article_text"],
            axis=1,
        )  # combine the title and text into a single column
        df["language"] = df["article_language"]
        df["stratify"] = df["class"] + df["language"]
        df["class"] = df["class"].apply(lambda x: 1 if x == "disinformation" else 0)
        df["label"] = df["class"]

        df = remove_imbalanced_languages(df)

        train_df, dev_df = train_test_split(df, test_size=0.1, stratify=df["stratify"], random_state=42)

        train_df = train_df[["text", "label", "language", "keywords", "published_date"]]
        dev_df = dev_df[["text", "label", "language", "keywords", "published_date"]]

        train_df.to_csv(os.path.join(exp_data_path, "euvsdisinfo.csv"), index=False)
        dev_df.to_csv(os.path.join(exp_data_path, "euvsdisinfo_dev.csv"), index=False)
    else:
        raise Warning("EUvsDisinfo not found at data/euvsdisinfo.csv")

    # MM-Covid
    if os.path.exists(mmcovid_path):
        df = pd.read_json(mmcovid_path, orient="records", lines=True)
        df = df.dropna(subset=["ref_source", "label", "lang"])
        langs = ["en", "es", "pt", "hi", "fr", "it"]
        df = df[df["lang"].isin(langs)]
        df["text"] = df["ref_source"].apply(lambda x: x.get("text"))
        df["text_len"] = df["text"].apply(lambda x: len(x.split()))
        df = df[df["text_len"] > 10]
        df = df[["text", "label", "lang"]]
        df = df.reset_index(drop=True)
        lang_map = {
            "en": "English",
            "es": "Spanish",
            "pt": "Portuguese",
            "hi": "Hindi",
            "fr": "French",
            "it": "Italian",
        }
        df["lang"] = df["lang"].apply(lambda x: lang_map[x])
        df = df.rename(columns={"text": "text", "label": "label", "lang": "language"})
        df["label"] = df["label"].apply(lambda x: 0 if x == "real" else 1)
        df = remove_imbalanced_languages(df)

        df["label+language"] = df["label"].astype(str) + "_" + df["language"]
        df_train, df_dev = train_test_split(df, test_size=0.1, random_state=SEED, stratify=df["label+language"])
        df_train = df_train[["text", "label", "language"]]
        df_dev = df_dev[["text", "label", "language"]]
        df_train.to_csv(os.path.join(exp_data_path, "mmcovid.csv"), index=False)
        df_dev.to_csv(os.path.join(exp_data_path, "mmcovid_dev.csv"), index=False)

    else:
        raise Warning("MM-Covid not found at data/news_collection.json")

    # FakeCovid
    if os.path.exists(fakecovid_path):
        df = pd.read_csv(fakecovid_path)
        df = df.rename(columns={"article_text": "text", "class": "label", "article_language": "language"})
        df = remove_imbalanced_languages(df)
        df["label+language"] = df["label"].astype(str) + "_" + df["language"]
        df_train, df_dev = train_test_split(df, test_size=0.1, random_state=SEED, stratify=df["label+language"])
        df_train = df_train[["text", "label", "language"]]
        df_dev = df_dev[["text", "label", "language"]]
        df_train.to_csv(os.path.join(exp_data_path, "fakecovid.csv"), index=False)
        df_dev.to_csv(os.path.join(exp_data_path, "fakecovid_dev.csv"), index=False)
    else:
        raise Warning("FakeCovid not found at data/FakeCovid.csv")

# %%
