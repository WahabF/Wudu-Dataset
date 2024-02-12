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

Note: In the code, the numbering is from 0 to 13 not 1 to 14, and from 0 to 7, not 1 to 8.
Note: The numbering used in the code and described above is different from the order in the paper; the order in the paper is reorganized for better illustration.

### Default labelset of 8 classes
Focusing on the rest of the non excluded classes, the classes can be mapped using our framework to 2, 5, 7, 8  (default), 10, or 11 classes. Where similar actions are grouped, in the case of binary (2) classes are grouped based on wither they require water tap on (class 1) or water tap off (class 0).
The 8 classes labelset decribed in the paper this is the numbering used in the coder:
1. Collect water [tap on]
2. Non Wudu action [tap off]
3. Hand washing [tap on]
4. Mouth & nose washing [tap off]
5. Face washing [tap off]
6. Arm washing [tap on]
7. Head wiping [tap off]
8. Foot wiping [tap off]
Note: The numbering in code is from 0 to 7 not 1 to 8.
Note: The numbering used in the code is different from the order in the paper, the order in the paper is reorganized for better illustration.

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

