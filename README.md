# Wudu Dataset

This dataset is submitted ICIP as dataset paper entitled "DATASET FOR ANTICIPATING FAST-CHANGING WUDU ACTIONS IN WATER TAP INTERACTIONS".

For any possible query regarding the datasets, please contact the first author of the paper.

## How to download the datasets
The full datasets can be downloaded via:
https://doi.org/10.7910/DVN/HAJM3Y

If you need to download the preprocessed data:
https://drive.google.com/file/d/1Ea-kuAMeMQqoJrJK4Zo_yz-_TCrFUKia/view?usp=drive_link

## Dataset Structure

After downloading the datasets, please transfer the "kfupm" and "kaust" directories into the data directory. These folders contain data sourced from two distinct locations. Within these directories, each subfolder corresponds to a unique session, comprising multiple Wudu experiments (samples). 

An example of a sample directory in kfupm session 1:
/data/kfupm/1/sample 1


### Sample File Contents
Each sample consists of two files: 
- `coordinates.mat`: Contains a sequence of body joint coordinates.
- `labels.mat`: Contains the corresponding sequence labels, ensuring synchronization between the coordinates and labels.



## Structures of the datasets
After downloading the data, make sure to move the "kfupm" and "kaust" folders to the data folder.
"kfupm" and "kaust" folders contains data collected from two different locations. Each subfolder represent a single session of multiple Wudu experiments recorded. An example of a session directory:
/data/kfupm/1/
Each Wudu experiment is stored in a seperate sample file. An example of a sample directory:
/data/kfupm/1/sample 1
Each sample contains a coordinates.mat and labels.mat folder. A coordinates.mat includes a sequence of body joints coordinates recorded, with the an in sync corresponding sequence label in the labels.mat file.

## Samples with missing skeletons

## Action Classes

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

