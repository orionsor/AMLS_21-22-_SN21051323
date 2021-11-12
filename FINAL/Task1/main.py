import random
import os
import sys
#import torch.backends.cudnn as cudnn
#import torch.optim as optim
#import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
from dataset_preprocess import *
from sklearn.model_selection import train_test_split
from sklearn import svm
import pickle

"""#######################################
     parameter setting
    #######################################"""
batch_size = 32
root = './dataset/'
data_root = "data.p"
label_root = "label.p"
"""#######################################
     import and load datasets
    #######################################"""
filepath_data = os.path.join(root,data_root)
filepath_label = os.path.join(root,label_root)
data = pickle.load(open(filepath_data, "rb"))
label = pickle.load(open(filepath_label, "rb"))
#for i  in range(6):
#    plt.subplot(2, 3, i+1)
#    plt.imshow(data[i])
#    plt.axis('off')

#plt.show()


x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=42)
#torch.utils.data.DataLoader(src_data,batch_size=batch_size,shuffle=True,num_workers=0)








#if __name__ == '__main__':
#


