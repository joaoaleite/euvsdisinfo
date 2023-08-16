# %%
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# %%
df = pd.read_csv("data/consolidated.csv")

# %%
# remove domains that are not news sources from negative examples
filter_domains_df = pd.read_csv("filter_negative_domains.csv")
filter_domains_df = filter_domains_df[filter_domains_df["is_news"] == True]
domains_list = filter_domains_df["index"].tolist()

neg_df = df.loc[df["objective"] == 0]
pos_df = df.loc[df["objective"] == 1]
neg_df = neg_df[neg_df["article_domain"].isin(domains_list)]

df = pd.concat([neg_df, pos_df])
df = df.dropna(subset=["debunk_date", "text"])
df = df.drop_duplicates(["text"])
df = df.reset_index(drop=True)
df.loc[:, "article_domain"] = df["article_domain"].apply(lambda x: x.split("www.")[1] if x.startswith("www") else x)

# %%
df.loc[df["debunk_date"].notna(), "debunk_date"] = df.loc[df["debunk_date"].notna(), "debunk_date"].apply(lambda x: datetime.strptime(x, '%d/%m/%Y'))
df.loc[df["debunk_date"].notna(), "year"] = df.loc[df["debunk_date"].notna(), "debunk_date"].apply(lambda x: x.year)
df.loc[df["debunk_date"].notna(), "quarter"] = df.loc[df["debunk_date"].notna(), "debunk_date"].apply(lambda x: x.quarter)
df.loc[df["debunk_date"].notna(), "month"] = df.loc[df["debunk_date"].notna(), "debunk_date"].apply(lambda x: x.month)

# %%
df

# %%
df.to_csv("data/en_euvsdisinfo.csv", index=False)

# %%
grouped_df = df.groupby(['year', 'quarter']).size().reset_index(name='count')
pivot_df = grouped_df.pivot(index='year', columns='quarter', values='count')

pivot_df.plot(kind='bar', stacked=True, alpha=0.7)
plt.tight_layout()
plt.show()


# %%
for year in df["year"].unique():
    df_year = df[df["year"] == year]
    keywords = df_year.loc[~df_year["keywords"].isna(), "keywords"].apply(lambda x: [t.strip().lower() for t in x.split(",")]).explode().value_counts().index.tolist()[:25]
    named_entities = df_year.loc[~df_year["article_named_entities"].isna(), "article_named_entities"].apply(lambda x: [t.strip().lower() for t in x.split(",")]).explode().value_counts().index.tolist()[:25]

    print(year)
    print(keywords)
    print(named_entities)

# %%
df_2023 = df[df["year"] == 2023]
df_2023["objective"].value_counts()

# %%
df_2023["keywords"].apply(lambda x: [t.strip().lower() for t in x.split(",")]).explode().value_counts()

# %%
df_2023.loc[~df_2023["article_named_entities"].isna(), "article_named_entities"].apply(lambda x: [t.strip().lower() for t in x.split(",")]).explode().value_counts()


