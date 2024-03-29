# Wudu Dataset

This dataset is submitted to ICIP 2024 as a dataset paper entitled "DATASET FOR ANTICIPATING FAST-CHANGING WUDU ACTIONS IN WATER TAP INTERACTIONS".

For any possible query regarding the datasets, please contact the paper's first author.

## To-do:
- [x] Data description
- [x] Data .h5 generation codes
- [ ] Torch data loader example
- [ ] Metrics generation codes
- [ ] Vizualization codes

## Dataset and Environment
### Download the datasets
The full datasets can be downloaded via:
https://doi.org/10.7910/DVN/HAJM3Y

If you need to download the preprocessed data:
https://drive.google.com/file/d/1Ea-kuAMeMQqoJrJK4Zo_yz-_TCrFUKia/view?usp=drive_link

### Clone the repo
```
git clone https://github.com/WahabF/Wudu-Dataset.git && cd Wudu-Dataset
```

### Environment setup
- Create a conda environment:
```
conda create -n wudu python=3.8
```
- Activate the conda environment:
```
conda activate wudu
```
- install the required packages:
```
pip install -r requirements.txt
```

## Data Files Content

After downloading the datasets, please transfer the "kfupm" and "kaust" directories into the data directory. These folders contain data sourced from two distinct locations. Within these directories, each subfolder corresponds to a unique session, comprising multiple Wudu experiments (samples). 

An example of a sample directory in kfupm session 1:
`/data/kfupm/1/sample 1`

### Sample File Contents
Each sample consists of two files: 
- `coordinates.mat`: Contains a sequence of body joint coordinates.
- `labels.mat`: Contains the corresponding sequence labels, ensuring synchronization between the coordinates and labels.

## Action Classes

### Original Label Set of 14 Classes
In the `labels.mat` file, an original set of 14 classes is used:
1. Non-Wudu action [tap off]
2. Hand washing [tap on]
3. Collect water for mouth & nose washing [tap on]
4. Mouth & nose washing [tap off]
5. Collect water for face washing [tap on]
6. Face washing [tap off]
7. Arm washing [tap on]
8. Collect water for head wiping [tap on]
9. Head wiping [tap off]
10. Foot washing [tap on] *(not used)
11. Collect water for arm washing [tap on] *(not used)
12. Arm washing after collecting water [tap off] *(not used)
13. Collect water for foot wiping [tap on]
14. Foot wiping [tap off]

Note that classes 10, 11, and 12 are ignored due to not having enough representative examples in the data.

### Default Label Set of 8 Classes
Focusing on the remaining, non-excluded classes, the classes can be mapped using our framework to 2, 5, 7, 8 (default), 10, or 11 classes, where similar actions are grouped. In the case of binary (2) classes, they are grouped based on whether they require the water tap to be on (class 1) or off (class 0).
The 8-class label set described in the paper is as follows:
1. Collect water [tap on]
2. Non-Wudu action [tap off]
3. Hand washing [tap on]
4. Mouth & nose washing [tap off]
5. Face washing [tap off]
6. Arm washing [tap on]
7. Head wiping [tap off]
8. Foot wiping [tap off]

Notes: 
- The numbering in code is from 0 to 7 not 1 to 8.
- The numbering used in the code is different from the order in the paper, the order in the paper is reorganized for better illustration.

## Generate .h5 data file
```
python data_get.py --data_root_dir /data --output_dir /data --classes 8 --window 20 --nextPredFrame 3
```


## Citation
@data{DVN/HAJM3Y_2024,
author = {Masood, Mudassir and Felemban, Abdulwahab and Almadani, Murad and Ahmed, Mohanad and Al-Naffouri, Tareq},
publisher = {Harvard Dataverse},
title = {{Wudu Dataset}},
UNF = {UNF:6:LMXjZ/Ed+PCOrV0hZEkQYA==},
year = {2024},
version = {V1},
doi = {10.7910/DVN/HAJM3Y},
url = {https://doi.org/10.7910/DVN/HAJM3Y}
}

