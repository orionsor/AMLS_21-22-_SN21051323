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
from dataset_rgb import *
import sys

start=time.time()

CATEGORY_INDEX = {
    "no_tumor": 0,
    "meningioma_tumor": 1,
    "pituitary_tumor":2,
    "glioma_tumor":3
}
root = './dataset/image/'
directory = './dataset/label.csv'
device = torch.device('cpu')
#########trail
#root = './dataset/image_trail/'
#directory = './dataset/binary_trail.csv'




def fit(epoch,model,trainloader,testloder):
    #下面三个是个数不是概率
    correct = 0
    total = 0
    running_loss = 0
    #Dropput在训练的时候回随机丢弃神经元(的输出),但预测的时候不会
    #model.train()是训练模式,想让Dropout发挥作用,对BN层也有用
    model.train()
    for batch_idx, (x, y) in enumerate(trainloader):
        #将训练数据也放到GPU上
        x = x.to(device)
        y = y.to(device)
        print("\norigin-target",y)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        # 梯度置为0
        optimizer.zero_grad()
        # 反向传播求解梯度
        loss.backward()
        #for name, param in model.named_parameters():
        #    print('\n层:', name, param.size())
        #    print('权值梯度', param.grad)
        #    print('权值', param)

        # 优化
        optimizer.step()
        # 不需要进行梯度计算

        with torch.no_grad():
            #torch.argmax将数字转换成真正的预测结果
            y_pred = torch.argmax(y_pred, dim=1)
            print("\ntrain-result:",y_pred)
            #计算个数
            correct += (y_pred == y).sum().item()
            total += y.size(0)
            running_loss += loss.item()

        sys.stdout.write(
            '\r epoch: %d, [iter: %d / all %d], loss: %f' \
            % (epoch, batch_idx + 1, len(trainloader), loss.detach().numpy()))
        sys.stdout.flush()


    #除以的是总样本数 trainloader.dataset是形参,实参是train_dl即train_dl.dataset
    #train_dl.dataset指向的是train_ds
    epoch_loss = running_loss/len(trainloader.dataset)
    epoch_acc = correct/total

    test_correct = 0
    test_total = 0
    test_running_loss = 0

    #model.eval()是预测模式,不让Dropout发挥作用,对BN层也有用
    model.eval()
    with torch.no_grad():
        for x,y in testloder:
            #将测试数据也要放到GPU上
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            print("\ntest-tensor",y_pred)
            loss = loss_fn(y_pred,y)
            y_pred = torch.argmax(y_pred, dim=1)
            print("\ntest-result",y_pred)
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
#分割线----------------------以上是定义学习函数的内容---------------------------

if __name__ == '__main__':


    model = torchvision.models.vgg16(pretrained=False)
    print(model)
    model = model.to(device)

    #for p in model.features.parameters():
    #    # 将卷积层部分的参数冻结
    #    p.requires_grad = False

    # model.classifier会返回线性层所有层
    # 将out_features=1000改为out_features = 4
    model.classifier[-1].out_features = 4
    #model.features[0] = nn.Conv2d(1, 64, 3, 1, 1)
    #print(model)

    #initialization
    #for m in model.parameters():
    #    nn.init.kaiming_normal_(m, a=0, mode='fan_out', nonlinearity='relu')



    # 只需要优化model.classifier的参数
    #optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=0.0001)
    #optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)


    #load dataset
    dataset_all = raw_dataset(root, directory)
    #train_dataset, test_dataset = torch.utils.data.random_split(dataset=dataset_all, lengths=[240, 61],
    #                                                            generator=torch.Generator().manual_seed(3))
    train_dataset, test_dataset = torch.utils.data.random_split(dataset=dataset_all, lengths=[2400, 600],generator=torch.Generator().manual_seed(0))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=0)
    #load model
    #model = VGG_16()
    device = torch.device('cpu')
    model = model.to(device)
    #########parameter setting############
    n_epoch = 30
    learning_rate = 0.0002
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,betas=[0.9,0.999],eps=1e-08,)
    loss_fn = nn.CrossEntropyLoss()
    #train_loss = []
    #test_accuracy = []
    best_accu = 0.0


    #便于随着训练的进行观察数值的变化
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
        if epoch_test_acc == max(test_acc):
            torch.save(model, '{0}/modelterm_best_vgg.pth'.format('./'))

    print("\nbest test accuracy:", max(test_acc))
    print("\nbest train accuracy:", max(train_acc))

    end = time.time()
    print(end-start)

    plt.figure(1)
    plt.subplot(121)
    plt.subplots_adjust(wspace=0.5)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(range(1, n_epoch + 1), train_loss, label='train_loss', color='k')
    plt.plot(range(1, n_epoch + 1), test_loss, label='test_loss', color='r')
    plt.legend()
    plt.title('loss')
    plt.subplot(122)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('accuracy')
    plt.plot(range(1, n_epoch + 1), train_acc, label='train_acc', color='k')
    plt.plot(range(1, n_epoch + 1), test_acc, label='test_acc', color='r')

    plt.legend()
    plt.savefig('./plot_vgg.jpg')
    plt.show()



    plt.show()
