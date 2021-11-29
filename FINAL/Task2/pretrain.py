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
from dataset_cnn import *
import sys

start=time.time()

CATEGORY_INDEX = {
    "no_tumor": 0,
    "meningioma_tumor": 1,
    "pituitary_tumor":2,
    "glioma_tumor":3
}
#root = './dataset/image/'
#directory = './dataset/label.csv'
device = torch.device('cpu')
#########trail
root = './dataset/image_trail/'
directory = './dataset/binary_trail.csv'




def fit(epoch,model,trainloader,testloder):

    correct = 0
    total = 0
    running_loss = 0

    model.train()
    for batch_idx, (x, y) in enumerate(trainloader):

        x = x.to(device)
        y = y.to(device)
        #print("\norigin-target",y)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        sys.stdout.write(
            '\r epoch: %d, [iter: %d / all %d], loss: %f' \
            % (epoch, batch_idx + 1, len(trainloader), loss.detach().numpy()))
        sys.stdout.flush()

        with torch.no_grad():

            y_pred = torch.argmax(y_pred, dim=1)
            print("\ntrain-result:",y_pred)

            correct += (y_pred == y).sum().item()
            total += y.size(0)
            running_loss += loss.item()




    epoch_loss = running_loss/len(trainloader.dataset)
    epoch_acc = correct/total

    test_correct = 0
    test_total = 0
    test_running_loss = 0


    model.eval()
    with torch.no_grad():
        for x,y in testloder:

            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            print("\n",y_pred)
            loss = loss_fn(y_pred,y)
            y_pred = torch.argmax(y_pred, dim=1)
            print(y_pred)
            test_correct += (y_pred == y).sum().item()
            test_total += y.size(0)
            test_running_loss += loss.item()

    epoch_test_loss = test_running_loss / len(test_loader.dataset)
    epoch_test_acc = test_correct / test_total
    print('\n')
    print('epoch: ',epoch,
          'train_loss: ',round(epoch_loss,3),
          'train_accuracy: ',round(epoch_acc,3),
          'test_loss: ',round(epoch_test_loss,3),
          'test_accuracy: ',round(epoch_test_acc,3)
              )

    return epoch_loss,epoch_acc,epoch_test_loss,epoch_test_acc


if __name__ == '__main__':


    model = torchvision.models.vgg16(pretrained=False)
    #print(model)
    model = model.to(device)

    #for p in model.features.parameters():

    #    p.requires_grad = False



    model.classifier[-1].out_features = 4
    model.features[0] = nn.Conv2d(1, 64, 3, 1, 1)

    # 只需要优化model.classifier的参数
    #optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=0.0001)
    #optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    loss_fn = nn.CrossEntropyLoss()

    #load dataset
    dataset_all = raw_dataset(root, directory)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset=dataset_all, lengths=[240, 61],
                                                                generator=torch.Generator().manual_seed(3))
    #train_dataset, test_dataset = torch.utils.data.random_split(dataset=dataset_all, lengths=[2100, 900],generator=torch.Generator().manual_seed(0))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=0)
    #load model
    #model = VGG_16()
    device = torch.device('cpu')
    model = model.to(device)
    #########parameter setting############
    n_epoch = 10
    learning_rate = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,betas=[0.9,0.999],eps=1e-08)
    #train_loss = []
    #test_accuracy = []
    best_accu = 0.0



    train_loss=[]
    train_acc=[]
    test_loss=[]
    test_acc=[]

    for epoch in range(n_epoch):
        epoch_loss,epoch_acc,epoch_test_loss,epoch_test_acc = fit(epoch,model,train_loader,test_loader)
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)
        test_loss.append(epoch_test_loss)
        test_acc.append(epoch_test_acc)



    end = time.time()
    print(end-start)

    plt.plot(range(1,n_epoch+1),train_loss,label='train_loss')
    plt.plot(range(1,n_epoch+1),test_loss,label='test_loss')
    plt.plot(range(1,n_epoch+1),train_acc,label='train_acc')
    plt.plot(range(1,n_epoch+1),test_acc,label='test_acc')



    plt.show()
