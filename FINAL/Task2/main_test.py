import torch
import pandas as pd
import torchvision
import time
import os
import shutil
from torchvision import datasets,transforms
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from dataset_rgb import *
import sys

"""Test additional dataset on ResNet 50"""

"""model is run on UCL server with 4 GPU, any 2 id from 0 to 3
   can be used here, depending on which 2 is available"""
os.environ["CUDA_VISIBLE_DEVICES"] = '0,3'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""Please use code below is CUDA out of memory occurs"""
#device = torch.device('cpu')

CATEGORY_INDEX = {
    "no_tumor": 0,
    "meningioma_tumor": 1,
    "pituitary_tumor":2,
    "glioma_tumor":3
}
root = './dataset/test/image/'
directory = './dataset/test/label.csv'


model = torch.load('./modelterm_best_res50_lr.pth')
model = model.to(device)
test_dataset = raw_dataset(root, directory)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=50, shuffle=True, num_workers=0)
test_correct = 0
test_total = 0
test_running_loss = 0

#model.eval(), no dropout
model.eval()
with torch.no_grad():
    for x,y in test_loader:
        #move data to device
        x = x.to(device)
        y = y.to(device)
        print("\ntest-origin",y)
        y_pred = model(x)
        print("\ntest-tensor",y_pred)
        #loss = loss_fn(y_pred,y)
        y_pred = torch.argmax(y_pred, dim=1)
        print("\ntest-result",y_pred)
        test_correct += (y_pred == y).sum().item()
        test_total += y.size(0)

test_acc = test_correct / test_total

print('\n')
print('test_accuracy: ',round(test_acc,4))

