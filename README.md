# Collect EuvsDisinfo
Use this repository to collect the EuvsDisinfo dataset described in our paper TBA.

## Setup python environment:
    conda create -n euvsdisinfo python=3.11.5
    conda activate euvsdisinfo
    pip install -r requirements.txt

## To collect the data:
1. Download the base data file [euvsdisinfo_base.csv](https://zenodo.org/records/10514307).
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

## Citing:
TBA
