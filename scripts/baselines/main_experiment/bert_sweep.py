import wandb
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import pandas as pd
from datasets import Dataset
import numpy as np
import random
import evaluate
import yaml


def main():
    SEED = 42
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels, average="macro")

    with open("scripts/baselines/main_experiment/bert_sweep_config.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    wandb.init(config=config)

    train_df = pd.read_csv("data/train.csv")
    dev_df = pd.read_csv("data/dev.csv")

    train_dataset = Dataset.from_pandas(train_df)
    dev_dataset = Dataset.from_pandas(dev_df)
    train_dataset = train_dataset.map(
        lambda examples: tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512),
        batched=True,
    )
    dev_dataset = dev_dataset.map(
        lambda examples: tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512),
        batched=True,
    )
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label", "language"])

    learning_rate = wandb.config.lr
    weight_decay = wandb.config.weight_decay
    train_epochs = wandb.config.train_epochs

    training_args = TrainingArguments(
        evaluation_strategy="epoch",
        output_dir="./wandb",
        learning_rate=learning_rate,
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
    )

    # train the model
    trainer.train()

    # predict
    predictions = trainer.predict(dev_dataset)
    preds = predictions.predictions.argmax(-1)
    labels = predictions.label_ids
    langs = dev_dataset["language"]

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
        final_results.append(results)

    avg_f1 /= len(set(langs))

    f1_macro_overall = f1.compute(predictions=preds_df["preds"], references=preds_df["label"], average="macro")["f1"]
    # Log metrics to Weights and Biases
    wandb.log(
        {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "num_epochs": train_epochs,
            "Lang AVG. F1-Macro": avg_f1,
            "F1-Macro": f1_macro_overall,
            "precision_negative": p_neg,
            "precision_positive": p_pos,
            "recall_negative": r_neg,
            "recall_positive": r_pos,
        }
    )


if __name__ == "__main__":
    main()
