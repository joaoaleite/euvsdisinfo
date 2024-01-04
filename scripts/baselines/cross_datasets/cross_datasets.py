import pandas as pd
from datasets import Dataset
import evaluate
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import StratifiedKFold
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


results = {"train_dataset": [], "test_dataset": [], "f1_lang_avg": [], "f1_overall": []}
for train_dataset in ["euvsdisinfo", "fakecovid", "mmcovid"]:
    for test_dataset in ["euvsdisinfo", "fakecovid", "mmcovid"]:
        if train_dataset == test_dataset:
            continue

        train_path = os.path.join("data", "experiments", f"{train_dataset}.csv")
        train_df = pd.read_csv(train_path)

        test_path = os.path.join("data", "experiments", f"{test_dataset}.csv")
        test_df = pd.read_csv(test_path)

        config_path = os.path.join("data", "artefacts", f"{train_dataset}_config.yaml")
        with open(config_path) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

        lr = config["lr"]["value"]
        weight_decay = config["weight_decay"]["value"]
        train_epochs = config["train_epochs"]["value"]

        language_weights = {
            language: len(test_df[test_df["language"] == language]) / len(test_df)
            for language in test_df["language"].unique()
        }

        train_ds = Dataset.from_pandas(train_df)
        test_ds = Dataset.from_pandas(test_df)
        train_ds = train_ds.map(tokenize_function, batched=True)
        test_ds = test_ds.map(tokenize_function, batched=True)
        train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label", "language"])
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
            train_dataset=train_ds,
            eval_dataset=test_ds,
            compute_metrics=compute_metrics,
        )

        # train the model
        trainer.train()

        # predict
        predictions = trainer.predict(test_ds)
        preds = predictions.predictions.argmax(-1)
        labels = predictions.label_ids
        langs = test_ds["language"]

        preds_df = pd.DataFrame({"preds": preds, "label": labels, "language": langs})
        f1 = evaluate.load("f1")
        f1_overall = f1.compute(predictions=preds, references=labels, average="macro")["f1"]
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
        avg_f1 /= len(set(langs))

        results["train_dataset"].append(train_dataset)
        results["test_dataset"].append(test_dataset)
        results["f1_lang_avg"].append(avg_f1)
        results["f1_overall"].append(f1_overall)


results = pd.DataFrame(results)
results.columns = ["Train Dataset", "Test Dataset", "F1 AVG by Language", "F1 Overall"]
results = results.reset_index()

results_path = os.path.join("data", "results", "cross_datasets")
os.makedirs(results_path, exist_ok=True)

results.to_csv(os.path.join(results_path, "cross_datasets.csv"), index=False)
