# Appendix
This is the technical appendix for our paper "EUvsDisinfo: a Dataset for Multilingual Detection of Pro-Kremlin
Disinformation in News Articles" submitted to CIKM 2024.

## A - N-Grams Filters
Table A1 displays the N-grams referring to related debunk articles cited in EUvsDisinfo's response section. To obtain the N-grams, we lowercase, remove punctuations, and then sentence tokenise the text within the response section. Next, we use the <a href="https://radimrehurek.com/gensim/models/phrases.html">Gensim</a> Python library to compute 3-grams and 4-grams. We check the most frequent 3 and 4-grams within the sentences, and sort them decreasingly. Finally, we select the N-grams that refer to other debunks, such as "recurring pro-kremilin disinformation", or "see similar cases". Using this list of N-grams, we remove URLs that appear in the same sentence with an N-gram in the list.


| N-gram                                       | #Sentences |
|----------------------------------------------|------------|
| recurring pro-kremlin disinformation narrative | 3,123      |
| pro-kremlin disinformation narrative about   | 2,398      |
| disinformation narrative about               | 1,236      |
| see other examples of                       | 927        |
| a recurring pro-kremlin disinformation       | 848        |
| this is a recurring                         | 802        |
| disinformation cases alleging that           | 753        |
| similar cases claiming that                  | 736        |
| pro-kremlin disinformation narratives about | 697        |
| recurring pro-kremlin disinformation narratives | 681      |
| read more about the                         | 560        |
| read similar cases claiming                 | 525        |
| is a recurring pro-kremlin                  | 464        |
| other examples of similar                   | 453        |
| recurring pro-kremlin narrative about       | 447        |
| a recurring pro-kremlin narrative           | 441        |
| a recurring disinformation narrative        | 439        |
| earlier disinformation cases alleging       | 430        |
| see earlier disinformation cases            | 422        |
| disinformation narratives about             | 375        |
|----------------------------------------------|------------|
| recurring pro-kremlin disinformation        | 4,541      |
| pro-kremlin disinformation narrative        | 4,015      |
| disinformation narrative about              | 2,767      |
| a recurring pro-kremlin                    | 1,363      |
| see other examples                          | 1,145      |
| pro-kremlin disinformation narratives       | 1,114      |
| recurring pro-kremlin narrative             | 1,036      |
| other examples of                           | 1,008      |
| disinformation narratives about             | 952        |
| is a recurring                             | 898        |
| see similar cases                          | 731        |

[Table A1: N-grams associated to related debunk articles cited
in EUvsDisinfo’s response section, along with the number of
sentences in which the n-gram occurs. 4-grams are above the
dashed line, and 3-grams are below the dashed line.]


## B - Distribution of Classes per Language
Table B1 shows the detailed breakdown of disinformation and trustworthy articles per language for EUvsDisinfo. We indicate the languages used throughout the experiments using a dashed line.

| Language        | Total | Disinformation | Trustworthy |
|-----------------|-------|----------------|-------------|
| English         | 6,546 | 425            | 6,121       |
| Russian         | 5,825 | 5,356          | 469         |
| German          | 313   | 216            | 97          |
| French          | 292   | 165            | 127         |
| Spanish         | 287   | 243            | 44          |
| Georgian        | 156   | 146            | 10          |
| Czech           | 152   | 111            | 41          |
| Polish          | 147   | 44             | 103         |
| Italian         | 103   | 85             | 18          |
| Lithuanian      | 78    | 28             | 50          |
| Romanian        | 68    | 17             | 51          |
| Slovak          | 35    | 32             | 3           |
| Serbian         | 31    | 27             | 4           |
| Finnish         | 30    | 8              | 22          |
|-----------------|-------|----------------|-------------|
| Arabic          | 3,451 | 3,449          | 2           |
| Ukrainian       | 323   | 8              | 315         |
| Hungarian       | 147   | 144            | 3           |
| Armenian        | 87    | 83             | 4           |
| Azerbaijani     | 54    | 54             | 0           |
| Swedish         | 22    | 4              | 18          |
| Bulgarian       | 18    | 4              | 14          |
| Dutch           | 11    | 3              | 8           |
| Norwegian       | 9     | 0              | 9           |
| Estonian        | 8     | 0              | 8           |
| Indonesian      | 8     | 6              | 2           |
| Bosnian         | 7     | 6              | 1           |
| Latvian         | 6     | 3              | 3           |
| Croatian        | 6     | 4              | 2           |
| Greek           | 5     | 2              | 3           |
| Belarusian      | 5     | 0              | 5           |
| Afrikaans       | 3     | 3              | 0           |
| Macedonian      | 3     | 1              | 2           |
| Chinese         | 2     | 2              | 0           |
| Persian         | 2     | 1              | 1           |
| Filipino        | 2     | 0              | 2           |
| Turkish         | 1     | 0              | 1           |
| Norwegian (Nynorsk) | 1  | 0              | 1           |
| Japanese        | 1     | 0              | 1           |
| Danish          | 1     | 0              | 1           |
| Catalan         | 1     | 0              | 1           |
| Korean          | 1     | 1              | 0           |
| Portuguese      | 1     | 1              | 0           |

[Table B1: Class distribution per language for EUvsDisinfo. per language. Languages above the dashed line are used in the classification experiments.]

## C - Hyperparameters and Training Details

For the Multinomial Naive Bayes MNB baseline, we try four different values for alpha (the Laplace smoothing constant): $0.1$, $1.0$, and $10$. For the Support Vector Machine (SVM) baseline, we tune three hyperparameters: $C$ (the regularisation parameter), with values of $1e^{-3}$, $1e^{-1}$, $1.0$, $5.0$, and $10$, the kernel type, as either linear or using a radial basis function (rbf). When using the rbf kernel, we also tune the gamma parameter (kernel coefficient) trying both $\frac{1}{n_features \times \sigma^2(features)}$ and $\frac{1}{n_features}$, as defined in the Scikit-Learn framework documentation 1. The best alpha for the MNB baseline is 0.1, which achieves an $F_{macroAVG}$ of $0.63$ on the development set. For the SVM model, the best configuration uses a $C$ of $1.0$ and a linear kernel. This configuration achieves an $F_{macroAVG}$ of $0.74$ on the development set.

For the mBERT and XLM-RoBERTa baselines, we finetune the pre-trained bert-base-multilingual-cased ($110M$ parameters) and XLM-RoBERTa-base ($125M$ parameters), respectively. For each of them, we run $10$ randomly sampled hyperparameter configurations considering the number of training epochs ($3$, $5$, or $10$), values of learning rate between $1e^{-1}$ and $1e^{-7}$ and weight decay between $1e^{-1}$ and $1e^{-3}$. Both the values of learning rate and weight decay are sampled uniformly from a logarithmic distribution. We truncate and pad the sequences to a maximum of 512 tokens (the maximum allowed for these architectures). This is done because the input articles are generally extensive (averaging $6,346$ characters, as shown in Section 2 (Methodology). The batch size is fixated at $16$. This the highest factor of $2$ that fit into $40GB$ of VRAM. Other configurations are the default defined in the <a href="https://huggingface.co/docs/transformers/main_classes/trainer">HuggingFace deep learning framework</a>.

The best mBERT configuration uses a learning rate of $1.78e^{-5}$, a weight decay of $5.1e^{-3}$, and trains for $5$ epochs, achieving an $F_{macroAVG}$ of $0.79$ on the development set. The best XLM-R configuration uses a learning rate of $6.2e^{-6}$, a weight decay of $5.4e^{-3}$, and trains for $5$ epochs, achieving an $F_{macroAVG}$ of $0.77$ on the development set.

All experiments involving mBERT and XLM-RoBERTa are performed on a single Nvidia A100 40GB GPU

## D - Cross-Dataset Experiment
Lastly, we study the cross-dataset generalisation capabilities of the best performing mBERT model. In addition to the EUvsDisinfo dataset, the experiments make use of the two other inherently multilingual datasets presented in Table 1, namely MM-Covid and FakeCovid. Following the methodology described in Section 4, for each dataset $D_i$, we filter out the under-represented languages, and use 10\% of the dataset exclusively for mBERT hyperparameter search. The best hyperparameters for each dataset are described below**. These portions of the datasets are then discarded. Next, we train the mBERT model using the entirety of the dataset $D_i$, and score it with the entirety of each of the other datasets $D_{j\neq i}$. Similarly to the methodology in Perez Rosas et. al. 2018, we compare the cross-dataset scores against baselines scores obtained by training and testing the \texttt{mBERT} model on the same dataset. For these baseline scenarios, we perform a leave-one-out 10-fold cross-validation strategy. Table D1 displays the cross-dataset results using EUvsDisinfo, MM-Covid, and FakeCovid, for which the baseline $F_{macroAVG}$ scores are $0.83$, $0.96$, and $0.66$, respectively.

| Train         | Test        | $F_{macroAVG}$ |
|---------------|-------------|---------------------------|
| EUvsDisinfo   | FakeCovid   | 0.37 (↓41.3%)             |
| EUvsDisinfo   | MM-Covid    | 0.41 (↓57.3%)             |
| FakeCovid     | EUvsDisinfo | 0.44 (↓47.0%)             |
| FakeCovid     | MM-Covid    | 0.18 (↓81.3%)             |
| MM-Covid      | EUvsDisinfo | 0.55 (↓33.7%)             |
| MM-Covid      | FakeCovid   | 0.46 (↓27.0%)             |

[Table D1: Cross-Dataset results. The percentage decrease with respect to the baseline score for the test set (EUvsDisinfo=0.83, MM-Covid=0.96, FakeCovid=0.66) is shown within parenthesis.]

** The best configurations for FakeCovid and MM-Covid are, respectively, a learning rate of $3.6e^{-6}$ and $1.7e^{-5}$, weight decay of $1.7e^{-2}$ and $1.9e^{-3}$, and $10$ and $5$ training epochs, achieving $F_{macroAVG}$ scores of $0.59$ and $0.96$.