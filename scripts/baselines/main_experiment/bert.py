# %%
# import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# %%
PRETRAINED_NAME = "bert-base-multilingual-cased"

# %%
import pandas as pd
from datasets import Dataset
import evaluate
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import StratifiedKFold
import numpy as np
import torch
import random
import yaml

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

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
df = pd.read_csv("data/train.csv")
df["label+language"] = df["label"].astype(str) + df["language"]

with open("scripts/baselines/main_experiment/euvsdisinfo_bert_config.yaml") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

lr = config["lr"]["value"]
weight_decay = config["weight_decay"]["value"]
train_epochs = config["train_epochs"]["value"]

# %%
final_results = []
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
for i, (train_idxs, test_idxs) in enumerate(skf.split(df, df["label+language"])):
    train_df = df.iloc[train_idxs]
    test_df = df.iloc[test_idxs]

    language_weights = {
        language: len(test_df[test_df["language"] == language]) / len(test_df)
        for language in test_df["language"].unique()
    }

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label", "language"])
    # define the model
    model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_NAME, num_labels=2)

    training_args = TrainingArguments(
        evaluation_strategy="epoch",
        output_dir="./wandb",
        learning_rate=lr,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=train_epochs,
        weight_decay=weight_decay,
        push_to_hub=False,
        logging_steps=10,
        load_best_model_at_end=False,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_strategy="no",
        seed=SEED,
        data_seed=SEED,
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
    langs = test_dataset["language"]

    preds_df = pd.DataFrame({"preds": preds, "label": labels, "language": langs})
    f1 = evaluate.load("f1")
    precision_negative = evaluate.load("precision", pos_label=0)
    precision_positive = evaluate.load("precision", pos_label=1)
    recall_negative = evaluate.load("recall", pos_label=0)
    recall_positive = evaluate.load("recall", pos_label=1)
    results = {"language": [], "f1": [], "weight": []}

    weighted_f1 = 0
    avg_f1 = 0
    final_results = []
    for language in set(langs):
        df_lang = preds_df[preds_df["language"] == language]
        labels = df_lang["label"].tolist()
        preds = df_lang["preds"].tolist()
        f1_score = f1.compute(predictions=preds, references=labels, average="macro")["f1"]
        p_neg = precision_negative.compute(predictions=preds, references=labels, average="binary")["precision"]
        p_pos = precision_positive.compute(predictions=preds, references=labels, average="binary")["precision"]
        r_neg = recall_negative.compute(predictions=preds, references=labels, average="binary")["recall"]
        r_pos = recall_positive.compute(predictions=preds, references=labels, average="binary")["recall"]

        weight = len(df_lang) / len(preds_df)
        weighted_f1 += f1_score * weight
        avg_f1 += f1_score
        results["language"].append(language)
        results["f1"].append(f1_score)
        results["weight"].append(weight)

    # avg_f1 /= len(set(langs))
    print("Weighted F1", weighted_f1)
    final_results.append(results)


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
