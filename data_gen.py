
"""
# This code to create .h5 train, val, test data splits or load it if it exist
"""
from tqdm import tqdm
import gc
from pathlib import Path
from contextlib import contextmanager
import random
import h5py
import torch.distributions as tdist
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from scipy.io import loadmat
from os import listdir
import argparse


class wuduDataset(Dataset):
    def __init__(self, root_dir, split, windowSize, nextPredFrame, selected_dims, selected_joints, selected_joints_map, class_map, df_fileNames, stride=1):
        self.root_dir = root_dir
        self.fileNames = df_fileNames
        self.stride = stride
        self.windowSize = windowSize
        self.nextPredFrame = nextPredFrame + 1 # The +1 compensate for something in the code used later
        # self.binary = binary # set to false for multiclass dataset
        self.class_map = class_map
        # if binary:
        #     self.class_map = {0:0,1:0,2:0,3:1,4:0,5:1,6:0,7:0,8:1,12:0,13:1} #hand/arm washing is a collectin water class
        # else:
        #     self.class_map = {0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,12:9,13:10} #hand/arm washing is a collectin water class
        self.labels = [self.class_map[i] for i in self.class_map]
        self.selected_joints= selected_joints
        self.selected_dims = selected_dims
        self.selected_joints_map = selected_joints_map

    def __getitem__(self, index):
        fileName = self.fileNames.iloc[index][0]
        coor = loadmat( self.root_dir + "/" + fileName + "/coordinates.mat")['coordinates'][0]
        label = loadmat(self.root_dir + "/" + fileName + "/labels.mat")['labels'][0]
        seqs, target = process_seq_wuduDataset(coor, label, self.class_map, \
                                   fileName, self.selected_joints, self.selected_dims, self.selected_joints_map)
        return seqs, target

    def __len__(self):
        return len(self.fileNames)

def process_seq_wuduDataset(coor, label, class_map, fileName, selected_joints, selected_dims, selected_joints_map):
    mask = [validCoor(coor[i]) for i in range(len(coor))]
    coor = coor[mask]
    label = label[mask]
    mask = [label[i] in class_map for i in range(len(label))]
    coor = coor[mask]
    label = label[mask]
    label = [class_map[label[i]] for i in range(len(label))]
    coor = torch.tensor(np.array([x for x in coor]))
    label = torch.FloatTensor(label).long()
    coor.reshape
    coor = coor[:,selected_dims][:,:,selected_joints] # Shape should be [T,dims,joints]
    coor = coor.nan_to_num().transpose(1,2)

    for i in range(len(coor)): # Replace zero joints with mean of other joints
        joint_mask_i = [not torch.allclose(coor[i][j], torch.tensor([0.0, 0.0, 0.0])) for j in range(len(coor[i]))] # Selects non zero
        if False in joint_mask_i:
            for j in range(len(coor[i])):
               if torch.allclose(coor[i][j], torch.tensor([0.0, 0.0, 0.0])):
                   coor[i][j] = coor[i][joint_mask_i].mean(dim=0)
    coor = coor.transpose(1,2)

    # coor = pre_normalization(coor,selected_joints_map) # optional prenormalization
    return coor, label

class HDF5Dataset(Dataset):
    """Simple classification dataset saved in an HDF5 file."""
    def __init__(self, hdf5_path, split, windowSize, nextPredFrame, temporal_aug_p=0, temporal_aug_range = 0.25):
        self.hdf5_path = Path(hdf5_path)
        self.hdf5_file = h5py.File(self.hdf5_path, 'r')
        self.split = split
        self.windowSize = windowSize
        self.nextPredFrame = nextPredFrame
        self.temporal_aug_p = temporal_aug_p
        self.temporal_aug_range = temporal_aug_range #+- that value
        self.jitter_augmentation = jitter_aug(spatial_aug_p=0.1,
                                         jit_range=0.1)
        # self.normalization = input_normalize(mean=[-0.0206, -1.9192, -0.1158], std=[0.1593, 0.3789, 0.2708])
        # self.labels = tuple(self.hdf5_file.attrs['labels'])
    def __getitem__(self, index):
        instance1 = self.hdf5_file[f'{self.split}/{index}/features']
        features = torch.from_numpy(instance1[()])
        instance2 = self.hdf5_file[f'{self.split}/{index}/label']
        target = torch.from_numpy(instance2[()])
        # target = self.labels.index(instance.attrs['label'])
        if self.split=='train' and self.temporal_aug_p>0:
            seq_len = features.shape[0]
            # features.shape = [T,C,J]
            # Temporal interpolation/extrapolation augmentation:
            if random.random() < self.temporal_aug_p:
                factor=0
                while factor*seq_len < self.windowSize + self.nextPredFrame + 1:
                    # factor = (random.random()*(2-0.25)) + 0.25  # sampling with a factor uniformly distributied on the interval [0.25,2)
                    # factor = (random.random() * (2 - 0.25)) + 0.25
                    factor = (random.random() * (2*self.temporal_aug_range)) + (1-self.temporal_aug_range)
                    factor = round(factor * seq_len) / seq_len
                features, target = temporalResizeH5(features,target,factor)

        features, target = process_seqH5(features, target, self.windowSize, self.nextPredFrame)
        return features, target

    def __len__(self):
        return len(self.hdf5_file[self.split])


    @classmethod
    def create(cls, hdf5_path, labels, **metadata):
        """Create an HDF5 file for a classification dataset."""
        metadata['labels'] = tuple(labels)
        Path(hdf5_path).parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(hdf5_path, 'w-') as dataset:
            dataset.attrs.update(metadata)

    @classmethod
    @contextmanager
    def append(cls, hdf5_path, split, **metadata):
        """A context manager to update an HDF5 dataset file."""
        gc.collect()  # the hdf5 file must not be open
        # sometimes h5py.File gets stuck in memory
        # also, it is not thread-safe when writing
        with h5py.File(hdf5_path, 'r+') as dataset:
            labels = set(dataset.attrs['labels'])

            if split not in dataset:
                dataset.create_group(split)
            dataset[split].attrs.update(metadata)

            def append_(features, label, **metadata):
                # assert label in labels, f'{label} not in {labels}'
        #        metadata['label'] = label
                index = len(dataset[split])
                dataset[f'{split}/{index}/features'] = features
                dataset[f'{split}/{index}/label'] = label
                # dataset[f'{split}/{index}'].attrs.update(metadata)

            try:
                yield append_
            finally:
                pass

def process_seqH5(coor, label, windowSize, nextPredFrame):
    seq_len = coor.shape[0] - nextPredFrame - windowSize
    num_dims = coor.shape[1]
    num_joints = coor.shape[2]
    # if(seq_len<=0):
    #     raise ValueError('The sequence length is 0 or negative!\n With the current configeration of window size and look ahead label frame this file should be discarded, check the interpolation speed. \n File name= ' + fileName)
    seqs_coor = torch.zeros((seq_len, num_dims, windowSize, num_joints))
    seqs_labels = torch.zeros((seq_len))
    for i in range(seq_len):
        seqs_coor[i] = coor[i:i+windowSize].transpose(0,1) # output shape = (1026, 3, 20, 25), 20 is subsequence length in the sequence of 1026
        seqs_labels[i] = label[i+windowSize+nextPredFrame-1] # example: i=0, windowsSize=20, nextPredFrame=3 (targeting the 3rd next frame) results in 22
    return  seqs_coor, seqs_labels.long()

def temporalResizeH5(coor, label, factor):
    coor = coor.transpose(0, -1)
    label = label.float().unsqueeze(0).unsqueeze(0)
    m_coor = torch.nn.Upsample(scale_factor=factor, mode='linear', align_corners=True)
    m_label = torch.nn.Upsample(scale_factor=factor, mode='nearest')
    coor = m_coor(coor)
    label = m_label(label)
    coor = coor.transpose(0, -1)
    label = label.squeeze(0).squeeze(0).long()
    return coor, label

# Note: we can rewrite this to continue where it stopped if failed
def process_data_split(dataset_path, split, loader):
    global device
    dataset = loader if isinstance(loader, Dataset) else loader.dataset
    with HDF5Dataset.append(dataset_path, split) as append:
        for i, (frames, labels) in enumerate(tqdm(loader)):
            features = frames
            append(features, labels)

def pre_normalization(data, selected_joints_map):
    T,C,V = data.shape
    s = data.transpose(1,-1)
    skeleton=s
    assert skeleton.sum() != 0
    main_body_center = skeleton[:, selected_joints_map[20]:selected_joints_map[20]+1, :] # [all T, V=selected_joints_map[20], all C) this means that the spineShoudler joint is the center
    # for i_p, person in enumerate(skeleton):
    assert skeleton.sum() != 0
    mask = (skeleton.sum(-1) != 0).reshape(T, V, 1)
    s = (s - main_body_center) * mask
    skeleton=s
    assert skeleton.sum() != 0
    body_unit = skeleton[:, selected_joints_map[3]:selected_joints_map[3]+1, :]# This mean the distance between the spineshoulder and neck is 1 i.e. selected_joints_map[2]
    dist_factor = 1/torch.sqrt((body_unit**2).sum(dim=-1))
    dist_factor = dist_factor.unsqueeze(-1).repeat(1, s.shape[1], s.shape[2])
    assert skeleton.sum() != 0
    s = s*dist_factor

    # s = np.transpose(s, [0, 4, 2, 3, 1])
    s = s.transpose(1,-1)
    return s

class input_normalize(nn.Module):
    # can be useful to normalize the input of deep learning models
    def __init__(self, mean= [-0.0206, -1.9192, -0.1158], std= [0.1593, 0.3789, 0.2708]):
        super().__init__()
        self.normalize = T.Normalize(mean, std, inplace=True)
    def forward(self, features):
        b,c,t,j,n = (features.shape[0],features.shape[1],features.shape[2],features.shape[3],features.shape[4])
        features = self.normalize(features.transpose(1, 2).contiguous().view(-1, c, j, n))
        features = features.view(b,t,c,j,n).transpose(1,2)
        return features

class jitter_aug(nn.Module):
    # Spatial noise augmentation: (jitter from https://github.com/abduallahmohamed/Social-Implicit/blob/main/trajectory_augmenter.py#L39)
    def __init__(self, spatial_aug_p=0.2, jit_range=0.01):
        super().__init__()
        self.spatial_aug_p=spatial_aug_p
        self.jit_range=jit_range
    def forward(self, features):
        if random.random() < self.spatial_aug_p:
            u = tdist.uniform.Uniform(torch.Tensor([-self.jit_range, -self.jit_range, -self.jit_range]),
                                      torch.Tensor([self.jit_range, self.jit_range, self.jit_range]))
            features = features + u.sample(sample_shape=(features.shape[0], features.shape[-1], features.shape[2], features.shape[3])).transpose(1, -1).to(features.device)
        return features

def isInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def validCoor(c):
    if((np.absolute(c.max() != 0.0))):
        return True
    else:
        return False


def countData(path, class_map, excl_files):
    counter = 0
    dirs = listdir(path)
    for i in dirs:
        dir = i
        if (isInt(dir)):
            samples = listdir(path + "/" + dir)
            for j in range(len(samples)):
                if (isInt(samples[j][-1])):
                    if not (path[-6:]+ dir + "/" + samples[j] in excl_files):
                        fileName = path[-6:-1]  + "/" + dir + "/" + samples[j]
                        coor = loadmat(path + "/" + dir + "/" + samples[j] + "/coordinates.mat")['coordinates'][0]
                        labels = loadmat(path + "/" + dir + "/" + samples[j] + "/labels.mat")['labels'][0]
                        coor_mask = [validCoor(coor[i]) for i in range(len(coor))]
                        if (True in coor_mask):
                            labels = labels[coor_mask]
                            labels_mask = [labels[i] in class_map for i in range(len(labels))]
                            if (True in labels_mask):
                                counter += 1
    return counter

def data2df(path, df, counter, class_map, excl_files):
    dirs = listdir(path)
    for i in dirs:
        dir = i
        if (isInt(dir)):
            samples = listdir(path + "/" + dir)
            for j in range(len(samples)):
                if (isInt(samples[j][-1])):
                    if not (path[-6:] + "/" + dir + "/" + samples[j] in excl_files):
                        fileName = path[-6:]  + "/" + dir + "/" + samples[j]
                        coor = loadmat(path + "/" + dir + "/" + samples[j] + "/coordinates.mat")['coordinates'][0]
                        labels = loadmat(path + "/" + dir + "/" + samples[j] + "/labels.mat")['labels'][0]
                        coor_mask = [validCoor(coor[i]) for i in range(len(coor))]
                        if (True in coor_mask):
                            labels = labels[coor_mask]
                            labels_mask = [labels[i] in class_map for i in range(len(labels))]
                            if (True in labels_mask):
                                df['fileName'][counter] = fileName
                                counter += 1
    return counter

def get_data(config):
    if not Path(config.dataset_path).exists():
        print("Creating data h5 file...")
        train_path = config.data_root_dir + '/kfupm'
        test_path = config.data_root_dir + '/kaust'
        class_map = {i:i for i in range(config.classes)}
        trainDataCount = countData(path=train_path, class_map=class_map, excl_files=config.excl_files)
        testDataCount = countData(path=test_path, class_map=class_map, excl_files=config.excl_files)
        totalDataCount = trainDataCount + testDataCount

        df_train = pd.DataFrame({'fileName': [[]] * trainDataCount})
        df_test = pd.DataFrame({'fileName': [[]] * testDataCount})

        counter = 0
        counter = data2df(path=train_path, df=df_train, counter=counter, class_map=class_map,  excl_files=config.excl_files)
        assert trainDataCount == counter
        counter = 0
        counter = data2df(path=test_path, df=df_test, counter=counter, class_map=class_map, excl_files=config.excl_files)
        assert testDataCount == counter


        train_dataset = wuduDataset(root_dir=config.data_root_dir,
                                    windowSize=config.window,
                                    split='does not matter',
                                    nextPredFrame=config.nextPredFrame,
                                    selected_dims=config.dims,
                                    selected_joints=config.selected_joints,
                                    selected_joints_map=config.selected_joints_map,
                                    class_map=class_map,
                                    df_fileNames=df_train,
                                    stride=1)
        test_dataset = wuduDataset(root_dir=config.data_root_dir,
                                   windowSize=config.window,
                                   split='does not matter',
                                   nextPredFrame=config.nextPredFrame,
                                   selected_dims=config.dims,
                                   selected_joints=config.selected_joints,
                                   selected_joints_map=config.selected_joints_map,
                                   class_map=class_map,
                                   df_fileNames=df_test,
                                   stride=1)
        
        def collate_noBatchDim(batch):
            return batch[0]
        

        train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            collate_fn=collate_noBatchDim,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            collate_fn=collate_noBatchDim,
        )


        if not Path(config.dataset_path).exists():
            print("Creating data h5 file...")
            HDF5Dataset.create(config.dataset_path, train_dataset.labels)
            print("Processing train data")
            process_data_split(config.dataset_path, 'train', train_loader)
            print("Processing test data")
            process_data_split(config.dataset_path, 'test', test_loader)

        train_set = HDF5Dataset(config.dataset_path, 'train', config.window, config.nextPredFrame, temporal_aug_p=config.temporal_augmentation_p, temporal_aug_range=config.temporal_augmentation_range)
        test_set = HDF5Dataset(config.dataset_path, 'test', config.window, config.nextPredFrame)

        return train_set, val_set, test_set


def gen_dataset_path(config):
    return config.output_dir + '/wudu' \
        + '_w' + str(config.window) + '_dims' + config.selected_dims_name[str(config.dims)] \
            + '_classes' + str(config.classes) + '_nextPredFrame' + str(config.nextPredFrame) + '.h5'

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--temporal_augmentation_p', required=False, default=0.0)
    parser.add_argument('--temporal_augmentation_range', required=False, default=0.0)
    parser.add_argument('--spatial_augmentation_p', required=False, default=0.0)
    parser.add_argument('--spatial_augmentation_range', required=False, default=0.0)
    
    parser.add_argument('--classes', required=False, default=8)
    parser.add_argument('--window', required=False, default=20)
    parser.add_argument('--nextPredFrame', required=False, default=3)
    parser.add_argument('--dims', required=False, default=[0, 1, 2]) 
    parser.add_argument('--excl_files', required=False, default='/train/5/sample 77') #files to exclude

    parser.add_argument('--data_root_dir', required=True)
    
    parser.add_argument('--output_dir', required=True)


    config = parser.parse_args()

    config.selected_joints = [i for i in range(25)]
    config.selected_joints_map = {i: i for i in range(25)}
    config.selected_classes_dict = {2: {0: 0, 1: 1, 2: 1, 3: 0, 4: 1, 5: 0, 6: 1, 7: 1, 8: 0, 12: 1, 13: 0}, \
                             10: {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 12: 8, 13: 9}, \
                             11: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 12: 9, 13: 10}, \
                             7: {1: 1, 2: 0, 3: 2, 4: 0, 5: 3, 6: 4, 7: 0, 8: 5, 12: 0, 13: 6}, \
                             8: {0: 1, 1: 2, 2: 0, 3: 3, 4: 0, 5: 4, 6: 5, 7: 0, 8: 6, 12: 0, 13: 7}, \
                             5: {1: 0, 2: 0, 3: 1, 4: 0, 5: 2, 6: 0, 7: 0, 8: 3, 12: 0, 13: 4}}
    
    config.binaryClassMaps = {2: [0, 1], \
                       10: [1, 1, 0, 1, 0, 1, 1, 0, 1, 0], \
                       11: [0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0], \
                       7: [1, 1, 0, 0, 1, 0, 0], \
                       8: [1, 0, 1, 0, 0, 1, 0, 0], \
                       5: [1, 0, 0, 0, 0]}

    config.selected_dims_name = {"[0, 1, 2]": "xyz", "[0, 1]": "xy", "[0, 2]": "xz", "[1, 2]": "xyz", "[0]": "x", "[1]": "y",
                          "[2]": "z"}

    config.dataset_path = gen_dataset_path(config)

    get_data(config)

if __name__ == "__main__":
    main()