import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import time
import os
import shutil
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from dataset_rgb import *
import sys
from pytorchtool import EarlyStopping

start=time.time()
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
root = './dataset/image/'
directory = './dataset/label.csv'


def fit(epoch,model,trainloader,testloder):
    """Model Training and Testing
               loss and accuracy of trainning and testng set are tracked and saved for curve graph"""
    correct = 0
    total = 0
    running_loss = 0

    model.train()
    for batch_idx, (x, y) in enumerate(trainloader):

        x = x.to(device)
        y = y.to(device)
        print("\norigin-target",y)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        # clear the gradient
        optimizer.zero_grad()
        # back propagation
        loss.backward()
        """code set to comment below is used to check gradient of each layer
                           for debug"""
        #for name, param in model.named_parameters():
        #    print('\nlayer:', name, param.size())
        #    print('gradient', param.grad)
        #    print('value', param)

        # 优化
        optimizer.step()


        with torch.no_grad():
            #convert torch to real target
            y_pred = torch.argmax(y_pred, dim=1)
            print("\ntrain-result:",y_pred)
            #calculate accuracy
            correct += (y_pred == y).sum().item()
            total += y.size(0)
            running_loss += loss.item()
        """Display realtime loss, iteration and epoch """
        sys.stdout.write(
            '\r epoch: %d, [iter: %d / all %d], loss: %f' \
            % (epoch, batch_idx + 1, len(trainloader), loss.cpu().detach().numpy()))
        sys.stdout.flush()



    epoch_loss = running_loss/len(trainloader.dataset)
    epoch_acc = correct/total

    test_correct = 0
    test_total = 0
    test_running_loss = 0

    #model.eval(), no dropout
    model.eval()
    with torch.no_grad():
        for x,y in testloder:
            #move data to device
            x = x.to(device)
            y = y.to(device)
            print("\ntest-origin",y)
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
    print('epoch: ',epoch+1,
          'train_loss: ',round(epoch_loss,5),
          'train_accuracy: ',round(epoch_acc,4),
          'test_loss: ',round(epoch_test_loss,5),
          'test_accuracy: ',round(epoch_test_acc,4)
              )

    return epoch_loss,epoch_acc,epoch_test_loss,epoch_test_acc


if __name__ == '__main__':


    model = torchvision.models.resnet18(pretrained=True)
    print(model)


    # change out_features from 1000 to 4
    model.fc.out_features = 4

    """For acceleration with multiple GPUS """
    device_id = [0, 1]
    model = nn.DataParallel(model, device_id)
    model.to(device)


    """Load Dataset"""
    dataset_all = raw_dataset(root, directory)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset=dataset_all, lengths=[2100, 900],generator=torch.Generator().manual_seed(0))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=True, num_workers=0)


    """parameter and key function setting """
    n_epoch = 50
    learning_rate = 0.0002
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,betas=[0.9,0.999],eps=1e-08,)
    loss_fn = nn.CrossEntropyLoss()
    #train_loss = []
    #test_accuracy = []
    best_accu = 0.0



    train_loss=[]
    train_acc=[]
    test_loss=[]
    test_acc=[]

    early_stopping = EarlyStopping(patience=7, verbose=True)
    for epoch in range(n_epoch):
        epoch_loss,epoch_acc,epoch_test_loss,epoch_test_acc = fit(epoch,model,train_loader,test_loader)
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)
        test_loss.append(epoch_test_loss)
        test_acc.append(epoch_test_acc)

        early_stopping(epoch_test_loss, model)
        """Apply early stopping mechanism to avoid over-fitting"""
        if early_stopping.early_stop:
            print("Early stopping")
            print("stop at epoch", epoch)
            break

    model.load_state_dict(torch.load('checkpoint.pt'))
    torch.save(model, '{0}/modelterm_best_res.pth'.format('./'))

    print("\nbest train accuracy:", max(train_acc))
    print("\nbest test accuracy:",max(test_acc))

    """Calculate computing time """
    end = time.time()
    print(end-start)

    """Plot learning curves graph """
    n = len(test_acc)
    plt.figure(1)
    plt.subplot(121)
    plt.subplots_adjust(wspace=0.5)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(range(1, n+1), train_loss, label='train_loss',color = 'k')
    plt.plot(range(1, n+1), test_loss, label='test_loss',color = 'r')
    plt.legend()
    plt.title('loss')
    plt.subplot(122)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('accuracy')
    plt.plot(range(1,n+1),train_acc,label='train_acc',color = 'k')
    plt.plot(range(1,n+1),test_acc,label='test_acc',color = 'r')

    plt.legend()
    plt.suptitle('Learning Curves for ResNet18')
    plt.savefig('./plot_res18.jpg')
    plt.show()










