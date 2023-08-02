# nexport/test/test_xor.py

import torch
from torch import Tensor, tensor
from torch import nn
from torch.optim import Adam
import numpy as np
import nexport
Xs = Tensor(np.array([[0,1],[1,0],[1,1],[0,0]]))
Ys = Tensor(np.array([[1],[1],[0],[0]]))


class XORModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2,2),
            nn.Sigmoid(),
            nn.Linear(2,1),
            nn.Sigmoid()
        )
        self.optimizer = Adam(self.parameters())
        self.loss = nn.MSELoss()
        
    def forward(self,X):
        return self.layers(X)
    
    def fit(self,X,y_true):
        self.optimizer.zero_grad()
        y_pred = self.forward(X)
        loss = self.loss(y_true,y_pred)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    

xor_model = XORModel()
xor_model(Xs)
EPOCHS = 20_000
for i in range(EPOCHS):
    loss = xor_model.fit(Xs,Ys)
    if i % 2_000 == 0:
        print(loss)

print(xor_model(Xs))


class TestInfer():
    def test_truth_table_0_1(self):
        assert round(float(xor_model(tensor([0.0, 1.0])))) == 1
    
    def test_truth_table_1_0(self):
        assert round(float(xor_model(tensor([1.0, 0.0])))) == 1
    
    def test_truth_table_1_1(self):
        assert round(float(xor_model(tensor([1.0, 1.0])))) == 0
    
    def test_truth_table_0_0(self):
        assert round(float(xor_model(tensor([0.0, 0.0])))) == 0

