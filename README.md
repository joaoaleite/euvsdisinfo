## About
This repository contains the materials that allow reproducing the work introduced in the paper "EUvsDisinfo: A Dataset for Multilingual Detection of Pro-Kremlin Disinformation in News Articles". The EUvsDisinfo dataset contains around 18K articles labelled as either containing misinformation or not. The misinformation articles are sourced from pro-Kremilin outlets, while non-misinformation articles are sourced from credible / less biased outlets. The dataset is collected according to the URLs cited within the debunks made by the EUvsDisinfo organisation in [their website](https://euvsdisinfo.eu).


## Collect EuvsDisinfo
Use this repository to collect the EuvsDisinfo dataset described in our paper TBA.

## Setup python environment:
    conda create -n euvsdisinfo python=3.11.5
    conda activate euvsdisinfo
    pip install -r requirements.txt

## To collect the data:
1. Download the base data file in [Zenodo](https://zenodo.org/records/10514307).
2. Create a folder named ```data``` in the root directory.
3. Place the base data file inside the ```data``` folder.
4. Run ```python3 scripts/collect/collect.py```.
5. When finished, the script should save a file named ```euvsdisinfo.csv``` inside the ```data``` folder.

## To reproduce the experiments:
- **Data analysis**: open and run the eda.ipynb jupyter notebook.
- **Classification**: 
    1. Run the python script for the desired scenario inside baselines/.
    2. After finished, the script will save the results in a file named ```results_{scenario}.csv``` in the root folder.

## Supplementary material:
Please refer to [this file](https://github.com/JAugusto97/euvsdisinfo/blob/main/supplementary_material.md).

## License
The EUvsDisinfo dataset is licensed under a Creative Commons BY-SA 4.0 license. The code available for reproducing experiments is licensed under an Apache-2.0 license that can be found in the file LICENSE.txt.

## Citing:
Dataset: https://zenodo.org/records/10514307

Software: https://zenodo.org/records/10492913

Paper: 
    @inproceedings{10.1145/3627673.3679167,
    author = {Leite, Jo\~{a}o A. and Razuvayevskaya, Olesya and Bontcheva, Kalina and Scarton, Carolina},
    title = {EUvsDisinfo: A Dataset for Multilingual Detection of Pro-Kremlin Disinformation in News Articles},
    year = {2024},
    isbn = {9798400704369},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3627673.3679167},
    doi = {10.1145/3627673.3679167},
    abstract = {This work introduces EUvsDisinfo, a multilingual dataset of disinformation articles originating from pro-Kremlin outlets, along with trustworthy articles from credible / less biased sources. It is sourced directly from the debunk articles written by experts leading the EUvsDisinfo project. Our dataset is the largest to-date resource in terms of the overall number of articles and distinct languages. It also provides the largest topical and temporal coverage. Using this dataset, we investigate the dissemination of pro-Kremlin disinformation across different languages, uncovering language-specific patterns targeting certain disinformation topics. We further analyse the evolution of topic distribution over an eight-year period, noting a significant surge in disinformation content before the full-scale invasion of Ukraine in 2022. Lastly, we demonstrate the dataset's applicability in training models to effectively distinguish between disinformation and trustworthy content in multilingual settings.},
    booktitle = {Proceedings of the 33rd ACM International Conference on Information and Knowledge Management},
    pages = {5380–5384},
    numpages = {5},
    keywords = {classification, dataset, disinformation, news articles, pro-kremlin},
    location = {Boise, ID, USA},
    series = {CIKM '24}
    }
