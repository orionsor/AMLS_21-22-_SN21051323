import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision
import time
import os
import shutil
from torchvision import datasets,transforms
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataset_test import *
import sys


from pytorchtool import EarlyStopping


start=time.time()
os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'
device = torch.device("cuda")

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
        #test_running_loss += loss.item()


test_acc = test_correct / test_total

print('\n')
print('test_accuracy: ',round(test_acc,4))

