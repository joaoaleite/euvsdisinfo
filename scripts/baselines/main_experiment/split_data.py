import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/euvsdisinfo.csv")
df = df.fillna("")
df["text"] = df.apply(
    lambda x: x["article_title"] + " " + x["article_text"] if x["article_title"] != "" else x["article_text"], axis=1
)  # combine the title and text into a single column
df["language"] = df["article_language"]
df["stratify"] = df["class"] + df["language"]
df["class"] = df["class"].apply(lambda x: 1 if x == "disinformation" else 0)
df["label"] = df["class"]

# Split the dataframe into train, dev, and test sets
train_df, dev_df = train_test_split(df, test_size=0.1, stratify=df["stratify"], random_state=42)

train_df = train_df[["text", "label", "language"]]
dev_df = dev_df[["text", "label", "language"]]

print("Train set size:", len(train_df))
print("Dev set size:", len(dev_df))

train_df.to_csv("data/train.csv", index=False)
dev_df.to_csv("data/dev.csv", index=False)
