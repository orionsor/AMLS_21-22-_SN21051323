import random
import os
import sys
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
from model import CNNModel
import matplotlib.pyplot as plt
from dataset_preprocess import data, label
from sklearn.model_selection import train_test_split
import sklearn

"""#######################################
     parameter setting
    #######################################"""
batch_size = 32
"""#######################################
     import and load datasets
    #######################################"""

x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=42)
#torch.utils.data.DataLoader(src_data,batch_size=batch_size,shuffle=True,num_workers=0)




#if __name__ == '__main__':
#


