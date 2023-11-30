# Collect EuvsDisinfo
Use this repository to collect the EuvsDisinfo dataset described in our paper TBA.

## Setup python environment:
    conda create -n euvsdisinfo python=3.11.5
    pip install -r requirements.txt

## To access the data:
You can contact TBA to ask for access, or you can collect it yourself using this repository.
### If you do not have permission:
1. download stratcom-data.zip and annotated_domains.csv.
2. set the diffbot api key in your environment: ```export DIFFBOT_API_KEY="your_key"```.
3. set up the necessary files:
```
    mkdir -p data/raw/
    unzip stratcom-data.zip -d data/raw/
    mv annotated_domains.csv data/
```
4. run ```python3 crawl.py```. you may want to set a different number of processes in the script to speed up scraping.
5. run ```python3 consolidate.py```.

 the dataset will be in data/euvsdisinfo.csv.

### If you were given permission:
1. run ```dvc pull```
2. authorise access with your gmail account.

## To reproduce the experiments:
- **data analysis**: open and run the eda.ipynb jupyter notebook.
- **classification**: run the python script for the desired scenario inside baselines/.

## Citing:
TBA