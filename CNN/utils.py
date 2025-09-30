import json
import torch
import os

class Params():
    
    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)
            
    def save(self,json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    
    @property
    def dict(self):
        return self.__dict__
    


def folder_creator(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def save_models(model, best_model, save_dir):
    # save the last model
    torch.save(model.state_dict(), f'{save_dir}/model_parameters.pth')
    torch.save(model, f'{save_dir}/mymodel.pth')
    # save the best model
    torch.save(best_model.state_dict(), f'{save_dir}/bestmodel_parameters.pth')
    torch.save(best_model, f'{save_dir}/bestmodel.pth')



#function for moving data and model to a chosen device(gpu or cpu)
def to_device(data, device):    
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]    
    elif isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=True)
    else:
        # leave strings, ints, etc. untouched
        return data

#create a class to wrap our existing data loaders and move batches of data to the selected device.
class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    'Yield a batch of data after moving it to device'    
    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)
            
    def __len__(self):
        return len(self.dl)     #number of batches           
    

