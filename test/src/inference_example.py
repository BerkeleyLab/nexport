## Inference using trained ML model for the app of aerosol emulation 
## by Z.Bai, zhebai@lbl.gov

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from torchvision import transforms
from time import perf_counter
from utils import *
import nexport

torch.manual_seed(42)
torch.backends.cudnn.benchmark = True
torch.get_num_threads()

with open('../model/scaler_X1_X4.pkl', 'rb') as f:
    scaler_X = pickle.load(f)
with open('../model/scaler_y1_y4.pkl', 'rb') as f:
    scaler_y = pickle.load(f)
mean_X = scaler_X.mean_
std_X = np.sqrt(scaler_X.var_)
mean_y = scaler_y.mean_
std_y = np.sqrt(scaler_y.var_)
 

class InferenceModel:
    def __init__(self, model_path, device='cpu'):
        """
        Initialize the InferenceModel class.
        
        Parameters:
            model_path (str): path to the trained model
            device(str): device to run the inference on ('cpu' or 'cuda')
        """
        
        self.device = torch.device(device)
        self.model = self.load_model(model_path)
        self.model.to(self.device)
        self.model.eval() # evaluation mode
        
        # define preprocessing transformation
        self.transform = transforms.Compose([
            # transforms.ToTensor(),
            toTensor(),
            powTransform(1.0/3),
            normalize(mean_X, std_X)
        ])
            
    def load_model(self, model_path):
        """
        Load the trained model 
        """
        model = MyNet().double()
        model.load_state_dict(torch.load(model_path, map_location = self.device))
        for param in model.parameters():
            param.requires_grad = False
        return model

    def preprocess(self, sample):
        sample = np.load(data_path)
        sample = self.transform(sample)
        return sample

    def postprocess(self, sample):
        sample = transform_y(sample, mean_y, std_y)
        sample = odd_pow(sample,3)
        return sample

    def predict(self, sample):
        sample = self.preprocess(sample)
        sample = sample.to(self.device)
        with torch.no_grad():
            output = self.model(sample)
        output = self.postprocess(output)
        return output

# Example
if __name__ == "__main__":
    model_path = '../model/trained_model.pth'
    data_path = '../test/X_test5.npy'
    t0_start = perf_counter()
    inference_model = InferenceModel(model_path, device='cuda' if torch.cuda.is_available() else 'cpu')
    nexport.export(inference_model.model,"json_exp",  include_metadata=True, model_name="E3SM", model_author="Zhe Bai", activation_function="gelu", using_skip_connections=False)
    # t0_stop = perf_counter()
    # print("Elapsed time for loading the inference model in seconds:", t0_stop-t0_start)
    # t1_start = perf_counter()
    # output = inference_model.predict(data_path)
    # t1_stop = perf_counter()
    # print("Elapsed time for predicting output in seconds:", t1_stop-t1_start)
    #
    # print(output)
