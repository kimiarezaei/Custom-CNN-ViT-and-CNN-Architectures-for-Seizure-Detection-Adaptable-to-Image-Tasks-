# define CNN model
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Params
from torchsummary import summary

class MyCNN(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.network = nn.Sequential(
            # Block 1
            nn.Conv2d(1, params.out_channel, kernel_size=(5,5)),  
            nn.BatchNorm2d(params.out_channel), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            # Block 2
            nn.Conv2d(params.out_channel, params.out_channel*2, kernel_size=(3,3)),  
            nn.BatchNorm2d(params.out_channel*2),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=(2,2)),     
            # Block 3
            nn.Conv2d(params.out_channel*2, params.out_channel*4, kernel_size=(3,3)),  
            nn.BatchNorm2d(params.out_channel*4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            # Block 4
            nn.AdaptiveAvgPool2d((1,1)),  # global average pooling to change the features into a unit value
            # classification
            nn.Flatten(),  # flatten the output
            nn.Linear(1*1*params.out_channel*4, params.out_channel*2),  
            nn.BatchNorm1d(params.out_channel*2),
            nn.ReLU(),
            nn.Dropout(p=0.5),  # Dropout layer
            nn.Linear(params.out_channel*2, 2)  

        )

    def forward(self, x):     
        x = self.network(x)
        return x  

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        if hasattr(module,'weight'):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity='relu')
        if hasattr(module,'bias') and module.bias is not None:
            nn.init.zeros_(module.bias)

            

def main():
    ## check model output
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    params = Params(r'CNN/parameters/my_params.json')
    sample_input = torch.randn(10, 1, 113, 105) 
    model = MyCNN(params)
    model.apply(init_weights)
    model.to(device)
    output = model(sample_input.to(device))                # output_shape = batch_size,    number_of_patches = ((H//batch_size)*(W//batch_size)),     embedding dimension
    print(output.shape)
    print(summary(model, (1, 113, 105)))

if __name__ == '__main__':
    main() 