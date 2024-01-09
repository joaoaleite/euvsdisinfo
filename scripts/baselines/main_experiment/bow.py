"""
This script:
1- loads the train (90%) and dev (10%) splits

2- runs a grid search over the hyperparameters of a Multinomial Naive Bayes classifier and a Support Vector Machine
using the dev set to evaluate.

3- uses the best hyperparameters to train a classifier using cross validation with the 90% train set.

4- saves the results for each model as a csv file with results per language, and a json file with the
best hyperparameters.
"""

# %%
import pandas as pd
import scipy.sparse as sp
import os
from tqdm import tqdm

from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
import json


# %%
def run_gridsearch(model, param_grid, X_train, y_train, X_dev, y_dev, dev_languages):
    best_score = 0
    best_params = {}

    for params in ParameterGrid(param_grid):
        count_vectorizer = CountVectorizer()
        train_X_transformed = count_vectorizer.fit_transform(X_train)
        clf = model()
        clf.set_params(**params)
        clf.fit(train_X_transformed, y_train)
        dev_X_transformed = count_vectorizer.transform(X_dev)

        # Separate examples by their language. dict = {language: (features, labels)}
        examples_per_language = {}
        for i, language in enumerate(dev_languages):
            features = dev_X_transformed[i]
            label = y_dev[i]
            if language not in examples_per_language:
                examples_per_language[language] = (features, [label])
            else:
                examples_per_language[language] = (
                    sp.vstack((examples_per_language[language][0], features)),
                    examples_per_language[language][1] + [label],
                )

        # Score the classifier for each language and average the scores
        score = 0
        for language in examples_per_language:
            features, labels = examples_per_language[language]
            language_score = f1_score(labels, clf.predict(features), average="macro")
            score += language_score
        score /= len(examples_per_language)

        if score > best_score:
            best_score = score
            best_params = params

    print(best_score)
    return best_params


def score_best_model(model, best_params, X_train, y_train, X_test, y_test, test_languages):
    count_vectorizer = CountVectorizer()
    train_X_transformed = count_vectorizer.fit_transform(X_train)
    clf = model()
    clf.set_params(**best_params)
    clf.fit(train_X_transformed, y_train)
    test_X_transformed = count_vectorizer.transform(X_test)

    # Separate examples by their language. dict = {language: (features, labels)}
    examples_per_language = {}
    for i, language in enumerate(test_languages):
        features = test_X_transformed[i]
        label = y_test[i]
        if language not in examples_per_language:
            examples_per_language[language] = (features, [label])
        else:
            examples_per_language[language] = (
                sp.vstack((examples_per_language[language][0], features)),
                examples_per_language[language][1] + [label],
            )

    # Score the classifier for each language and average the scores
    results = {}
    avg = 0
    for language in examples_per_language:
        features, labels = examples_per_language[language]
        language_score = f1_score(labels, clf.predict(features), average="macro")
        results[language] = language_score
        avg += language_score
    avg /= len(examples_per_language)

    results["Average"] = avg
    return results


def main():
    results_path = os.path.join("data", "results", "main_experiment")
    os.makedirs(results_path, exist_ok=True)

    artefacts_path = os.path.join("data", "artefacts")
    os.makedirs(artefacts_path, exist_ok=True)

    data_path = os.path.join("data", "experiments")
    train_path = os.path.join(data_path, "euvsdisinfo.csv")
    dev_path = os.path.join(data_path, "euvsdisinfo_dev.csv")

    assert all(
        [os.path.exists(path) for path in [train_path, dev_path]]
    ), f"Data split files not found at '{data_path}'. Run the split_data.py script first."

    train = pd.read_csv(train_path)
    dev = pd.read_csv(dev_path)

    X_train = train["text"]
    y_train = train["label"]
    train_languages = train["language"]

    X_dev = dev["text"]
    y_dev = dev["label"]
    dev_languages = dev["language"]

    models = [MultinomialNB, SVC]
    params = [
        {"alpha": [0.01, 0.1, 1.0, 10.0]},
        {
            "C": [0.001, 0.1, 1.0, 5.0, 10.0],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"],
        },
    ]

    # The train set (90%) is now used to perform 10-fold cross-validation
    # Dev set is now discarded
    X_all = X_train
    y_all = y_train
    languages_all = train_languages
    for model, param_grid in zip(models, params):
        print(f"Running grid search for {model.__name__}")
        best_params = run_gridsearch(model, param_grid, X_train, y_train, X_dev, y_dev, dev_languages)
        print("Best parameters:", best_params)
        final_results = []
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        for i, (train_index, test_index) in tqdm(
            enumerate(skf.split(X_all, languages_all)), total=10, desc=f"Scoring best model for {model.__name__}"
        ):
            X_train, X_test = X_all.iloc[train_index].to_list(), X_all.iloc[test_index].to_list()
            y_train, y_test = y_all.iloc[train_index].to_numpy(), y_all.iloc[test_index].to_numpy()
            test_languages = languages_all.iloc[test_index].to_list()

            run_results = score_best_model(model, best_params, X_train, y_train, X_test, y_test, test_languages)
            final_results.append(run_results)

        # Create and format a dataframe with the results for each classifier
        final_results = pd.DataFrame(final_results).T
        final_results = final_results.stack().groupby(level=0).agg(["mean", "std"])
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
        final_results = final_results.loc[languages + ["Average"]]

        # Save results and best parameters
        csv_path = os.path.join(results_path, f"results_{model.__name__}.csv")
        final_results.to_csv(csv_path)
        params_path = os.path.join(artefacts_path, f"params_{model.__name__}.json")
        with open(params_path, "w") as f:
            json.dump(best_params, f)


# %%
main()
