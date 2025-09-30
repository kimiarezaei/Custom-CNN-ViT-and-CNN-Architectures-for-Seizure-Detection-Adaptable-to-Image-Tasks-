import numpy as np
import random
from datetime import date
import torch
from torchsummary import summary
import wandb
import argparse
import sys
import os
from sklearn.model_selection import KFold

from dataset_builder import MyDataset, split_dataset, MyDataLoader
from train import train, wandbinitialization
from test import test
from utils import save_models, Params, folder_creator, DeviceDataLoader
from model import MyCNNViT, init_weights


today = date.today()

# paths
PYTHON = sys.executable
parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', 
                    default=rf"your directory/spectrogram new", 
                    help='Directory containing data')

parser.add_argument('--info_dir', 
                    default=r'CNNViT\parameters/my_params.json',
                    help='Directory containing json file for hyperparameters')


parser.add_argument('--save_dir', 
                    default=rf'CNNViT\saved files\{today}',
                    help='Directory for saving the result')

args = parser.parse_args()

# Make a folder to save files
folder_creator(args.save_dir)

project_name = f'CNNViT_seizure'

# predefined parameters
params = Params(args.info_dir)

# Set random seeds
np.random.seed(params.random_seed)
random.seed(params.random_seed)
torch.manual_seed(params.random_seed)
torch.cuda.manual_seed(params.random_seed)
torch.cuda.manual_seed_all(params.random_seed)

# Read both folders into one dataset
train1= MyDataset(os.path.join(args.data_dir, 'ANSeR1'))
train2= MyDataset(os.path.join(args.data_dir, 'ANSeR2'))
train_total = torch.utils.data.ConcatDataset([train1, train2])
print(' dataset is initialized')

all_patient_ids = np.array(train1.namesT + train2.namesT)
cleaned_ids = [ "_".join(name.split("_")[:2]) for name in all_patient_ids]
unique_patients = np.unique(cleaned_ids)
print('unique patients:', unique_patients)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_pat_idx, test_pat_idx) in enumerate(kf.split(unique_patients), 1):
    print(f"========   Fold number{fold}   ============")
    new_path = os.path.join(args.save_dir, f'fold{fold}')
    folder_creator(new_path)

    train_patients = set(unique_patients[train_pat_idx])
    test_patients = set(unique_patients[test_pat_idx])
    print('test patients:', test_patients)
 
    # Get indices of samples belonging to those patients
    train_idx = [i for i, name in enumerate(all_patient_ids) if "_".join(name.split("_")[:2]) in train_patients]
    test_idx   = [i for i, name in enumerate(all_patient_ids) if "_".join(name.split("_")[:2]) in test_patients and 'noisy' not in name]  # augmented data is not for testing
   
    # Subsets without reloading
    train_subset = torch.utils.data.Subset(train_total, train_idx)
    test_subset   = torch.utils.data.Subset(train_total, test_idx)

    train_ds, val_ds = split_dataset(train_subset, params)
    print('split is done:',len(train_ds), len(val_ds))

    # seperating data into batches
    train_dl, validation_dl, test_dl = MyDataLoader(train_ds, val_ds, test_subset, params)

    # New model
    model = MyCNNViT(params)
    model.apply(init_weights)

    # move data to GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', device)

    train_loader = DeviceDataLoader(train_dl,device)
    validation_loader = DeviceDataLoader(validation_dl,device)
    test_loader = DeviceDataLoader(test_dl,device)

    model = model.to(device)

    # train and validate the model
    if params.use_wandb:
        wandb.login()
        wandbinitialization(project_name, params)

    # train the model
    model, best_epoch, best_model, df = train(model, params, train_loader, validation_loader, device)

    # save the model
    save_models(model, best_model, new_path)

    # test model on test set
    test(model, test_loader, device, new_path, df)

    if not params.apply_early_stop and best_epoch != params.epochs-1:
        test(best_model, test_loader, device, new_path, df)