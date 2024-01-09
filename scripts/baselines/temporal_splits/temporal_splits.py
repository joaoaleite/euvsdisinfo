import pandas as pd
from datasets import Dataset
import evaluate
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import numpy as np
import torch
import random
import yaml
import os

PRETRAINED_NAME = "bert-base-multilingual-cased"
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

metric = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels, average="macro")


tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_NAME)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)


data_path = os.path.join("data", "experiments", "euvsdisinfo.csv")
df = pd.read_csv(data_path)
configurations = [("Covid-19", "01-10-2019"), ("Invasion of Ukraine", "24-02-2022")]
results = {"Event": [], "Train Period": [], "Test Period": [], "F1 AVG by Language": [], "F1 Overall": []}
j = 0
for i, (scenario, cutoff_date) in enumerate(configurations):
    print("Scenario:", scenario)
    train_df = df[
        pd.to_datetime(df["published_date"], format="%d-%m-%Y") < pd.to_datetime(cutoff_date, format="%d-%m-%Y")
    ]
    test_df = df[
        pd.to_datetime(df["published_date"], format="%d-%m-%Y") >= pd.to_datetime(cutoff_date, format="%d-%m-%Y")
    ]

    config_path = os.path.join("data", "artefacts", "euvsdisinfo_config.yaml")
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    lr = config["lr"]["value"]
    weight_decay = config["weight_decay"]["value"]
    train_epochs = config["train_epochs"]["value"]

    final_results = []
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
    f1 = evaluate.load("f1")
    predictions = trainer.predict(test_dataset)
    preds = predictions.predictions.argmax(-1)
    labels = predictions.label_ids
    langs = test_dataset["language"]
    f1_overall = metric.compute(predictions=preds, references=labels, average="macro")["f1"]

    preds_df = pd.DataFrame({"preds": preds, "label": labels, "language": langs})
    precision_negative = evaluate.load("precision", pos_label=0)
    precision_positive = evaluate.load("precision", pos_label=1)
    recall_negative = evaluate.load("recall", pos_label=0)
    recall_positive = evaluate.load("recall", pos_label=1)

    weighted_f1 = 0
    avg_f1 = 0
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
        final_results.append(results)

    avg_f1 /= len(set(langs))
    results["Event"].append(scenario)
    results["Train Period"].append(f"<{cutoff_date}")
    results["Test Period"].append(f">={cutoff_date}")
    results["F1 Overall"].append(f1_overall)
    results["F1 AVG by Language"].append(avg_f1)


results = pd.DataFrame(results)
results = results.reset_index()

results_path = os.path.join("data", "results", "temporal_splits")
os.makedirs(results_path, exist_ok=True)

results.to_csv(os.path.join(results_path, "temporal_splits.csv"), index=False)
