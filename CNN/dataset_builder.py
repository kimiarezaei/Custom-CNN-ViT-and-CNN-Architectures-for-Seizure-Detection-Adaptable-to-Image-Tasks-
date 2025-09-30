"""Customized dataset and related functions for preprocessing the data"""
import time
import os
from torch.utils.data import random_split
import torch
from torch.utils.data.dataloader import DataLoader, Dataset

# split train data into train and validation set
def split_dataset(dataset, params):
    # Calculate the number of training samples 
    train_size = int(len(dataset) * params.train_portion)
    val_size = len(dataset) - train_size
    return random_split(dataset, [train_size, val_size])


class MyDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir

        self.file_path = [
        os.path.join(folder_path, file)
        for folder in os.listdir(self.root_dir)
        if os.path.isdir(folder_path := os.path.join(self.root_dir, folder))
        for file in os.listdir(folder_path)
        if file.endswith('.pt')
        ]

        self.length = len(self.file_path)  # Store the length of the dataset
        self.spectrogram_all = []
        self.label_all = []
        self.name_all = []

        # getting the matrix and its label
        print('loading data...')
        start_time = time.time()
        for path in self.file_path:
            name = os.path.splitext(os.path.basename(path))[0]
            sample_data =  torch.load(path)
            self.spectrogram_all.append(sample_data['spectrogram'])
            self.label_all.append(sample_data['seizure'])
            self.name_all.append(name)

        self.spectrogramsT = torch.stack(self.spectrogram_all, dim=0)
        self.labelsT = torch.tensor(self.label_all)
        self.namesT = self.name_all

        end_time = time.time()
        total_time = end_time - start_time
        print('test data loaded in:', total_time, 'seconds')

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        spectrogram, label, name = self.spectrogramsT[idx] , self.labelsT[idx], self.namesT[idx]
        return spectrogram, label, name



# seperating data into batches
def MyDataLoader(train_ds, val_ds, test_ds, params):
    train_dl = DataLoader(train_ds, batch_size=params.batch_size, shuffle=True)
    validation_dl = DataLoader(val_ds, batch_size=params.batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=params.batch_size, shuffle=False)
    print('number of train samples:', len(train_ds), 'number of validation samples:', len(val_ds), 'number of test samples:', len(test_ds))
    return train_dl, validation_dl, test_dl


