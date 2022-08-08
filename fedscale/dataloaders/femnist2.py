from __future__ import print_function

import csv
import json
import os
import os.path
import warnings

import numpy as np
import torch


class FEMNIST2():
    """
    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    classes = []

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, root, dataset='train', transform=None):
        self.root = root
        self.dataset = dataset # 'train','test'
        self.transform = transform

        self.path = os.path.join(self.processed_folder, self.dataset)
        self.data, self.targets = self.read_data_femnist2(self.path)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (json, target) where target is index of the target class.
        """
        cur_data = self.data[index]
        cur_data = torch.from_numpy(cur_data)
        cur_data = cur_data.to(torch.float32)
        cur_data = cur_data.reshape(1,28,28)
        
        return cur_data, int(self.targets[index])

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return self.root

    @property
    def processed_folder(self):
        return self.root

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder,self.dataset))) 
        
    def read_data_femnist2(self,data_dir):
        ret_data = []
        ret_label = []

        files = os.listdir(data_dir)        
        files = [f for f in files if f.endswith('.json') and f[0]!='_']

        for f in files:
            file_path = os.path.join(data_dir, f)
            my_data = np.array(json.load(open(file_path, 'r'))["records"])
            
            cur_data, cur_label = my_data[:,1:], my_data[:,0]

            for d in cur_data:
                ret_data.append(d)
            for l in cur_label:
                ret_label.append(l)

        return ret_data, ret_label