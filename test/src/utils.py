## Inference using trained ML model for the app of aerosol emulation 
## by Z.Bai, zhebai@lbl.gov

import torch
import torch.nn as nn

# MLP architecture
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.fc1 = nn.Linear(80, 256) 
        # self.dropout1 = nn.Dropout(0.2)
        # self.bn1 = nn.BatchNorm1d(256)  # BatchNorm layer after the 1st fully connected layer
        self.fc2 = nn.Linear(256, 384)  
        # self.dropout2 = nn.Dropout(0.2)
        # self.bn2 = nn.BatchNorm1d(16)   # BatchNorm layer after the 2nd fully connected layer
        self.fc3 = nn.Linear(384, 256)  
        # self.dropout3 = nn.Dropout(0.2)
        # self.bn2 = nn.BatchNorm1d(16)   
        self.fc4 = nn.Linear(256, 31)   

    def forward(self, x):
        x = self.fc1(x)
        act = nn.GELU()
        x = act(x)
        # x = self.dropout1(x)
        x = self.fc2(x)
        x = act(x)
        # x = self.dropout2(x)
        x = self.fc3(x)
        x = act(x)
        # x = self.dropout3(x)
        x = self.fc4(x)
        return x

class toTensor:
    # def __init__(self, device = 'cpu'):
    #     self.device = torch.device(device)
    def __call__(self, array):
        tensor = torch.tensor(array, dtype=torch.float64)
        return tensor
    
class powTransform:
    def __init__(self, par):
        self.par = par
    def __call__(self, sample):
        sample = odd_pow(sample, self.par)
        return sample
    
class normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std =std
    def __call__(self, sample):
        sample -= self.mean
        sample /= self.std
        return sample

def odd_pow(input, exponent):
    return input.sign() * input.abs().pow(exponent)

def transform_y(y, mean_y, std_y):
    y *= std_y
    y +=  mean_y
    return y

if __name__ == "__main__":
    main()