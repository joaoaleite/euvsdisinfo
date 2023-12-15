# %%
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# %%
PRETRAINED_NAME = "bert-base-multilingual-cased"

# %%
import pandas as pd
from datasets import Dataset
import evaluate
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import random

# %%
metric = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels, average="macro")


# %%
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_NAME)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)


# %%
df = pd.read_csv("data/euvsdisinfo.csv")
df["text"] = df["article_title"].fillna("") + " " + df["article_text"]
df["label"] = df["class"].apply(lambda x: 0 if x == "support" else 1)
languages = df["article_language"].unique()
df["label+language"] = df["label"].astype(str) + df["article_language"]

# %%
df = df[["text", "label", "article_language", "label+language"]]

# %%
final_results = []
for seed in range(10):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    train_df, test_df = train_test_split(df, test_size=0.3, random_state=seed, stratify=df["label+language"])
    language_weights = {
        language: len(test_df[test_df["article_language"] == language]) / len(test_df)
        for language in test_df["article_language"].unique()
    }

    print(language_weights)
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label", "article_language"])
    # define the model
    model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_NAME, num_labels=2)

    training_args = TrainingArguments(
        output_dir="./results-multi",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        num_train_epochs=3,
        weight_decay=0.01,
        push_to_hub=False,
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        save_strategy="epoch",
        seed=seed,
        data_seed=seed,
    )

    # define the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # train the model
    trainer.train()

    # predict
    predictions = trainer.predict(test_dataset)
    preds = predictions.predictions.argmax(-1)
    labels = predictions.label_ids
    langs = test_dataset["article_language"]

    preds_df = pd.DataFrame({"preds": preds, "labels": labels, "langs": langs})
    f1 = evaluate.load("f1")
    precision_negative = evaluate.load("precision", pos_label=0)
    precision_positive = evaluate.load("precision", pos_label=1)
    recall_negative = evaluate.load("recall", pos_label=0)
    recall_positive = evaluate.load("recall", pos_label=1)
    results = {"language": [], "f1": [], "weight": [], "seed": []}

    weighted_f1 = 0
    for language in set(langs):
        df_lang = preds_df[preds_df["langs"] == language]
        l = df_lang["labels"].tolist()
        p = df_lang["preds"].tolist()
        f1_score = f1.compute(predictions=p, references=l, average="macro")["f1"]
        p_neg = precision_negative.compute(predictions=p, references=l, average="binary")["precision"]
        p_pos = precision_positive.compute(predictions=p, references=l, average="binary")["precision"]
        r_neg = recall_negative.compute(predictions=p, references=l, average="binary")["recall"]
        r_pos = recall_positive.compute(predictions=p, references=l, average="binary")["recall"]

        weight = len(df_lang) / len(preds_df)
        weighted_f1 += f1_score * weight

        print(language, len(df_lang), f1_score, p_neg, p_pos, r_neg, r_pos)
        results["language"].append(language)
        results["f1"].append(f1_score)
        results["weight"].append(weight)
        results["seed"].append(seed)

    # avg_f1 /= len(set(langs))
    print("Weighted F1", weighted_f1)
    final_results.append(results)

# %%
languages

# %%
languages = [
    "English",
    "Russian",
    "German",
    "French",
    "Spanish",
    "Georgian",
    "Czech",
    "Polish",
    "Italian",
    "Lithuanian",
    "Romanian",
    "Slovak",
    "Serbian",
    "Finnish",
]

results = pd.concat([pd.DataFrame.from_dict(r) for r in final_results])
results = (
    results.groupby(["language"])
    .agg({"f1": ["mean", "std"], "weight": ["mean"]})
    .reset_index()
    .set_index("language")
    .loc[languages]
)
weighted_f1 = (results["f1"]["mean"].values * results["weight"]["mean"].values).sum()
avg_f1 = results["f1"]["mean"].mean()
results.loc["Weighted F1"] = [weighted_f1, "", ""]
results.loc["Avg. F1"] = [avg_f1, "", ""]
results.columns = ["F1 Mean", "F1 Std", "Weight"]
results = results.reset_index()

print(results)
results.to_csv("results_multilingual.csv", index=False)
