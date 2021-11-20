import torch
from torch import nn
from dataset_preprocess import *
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import dataset, dataloader
from dataset_cnn import *
import sys



CATEGORY_INDEX = {
    "negative": 0,
    "positive": 1
}
root = './dataset/image/'
directory = './dataset/binary_label.csv'



class VGG_16(nn.Module):
    """#######################################
    VGG_16 CNN NETWORK
    input：
        one batch of training set
    output：
        1、predicted vector
    #######################################"""
    def __init__(self):
        super(VGG_16, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 113),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2, 113),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 57),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 2, 57),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 29),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 2, 29),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 2, 29),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 2, 15),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 2, 15),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 2, 15),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 2, 8),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 2, 8),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 2, 8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer6 = nn.Sequential(
            nn.Linear(7 * 7 * 512, 4096),
            nn.Linear(4096, 4096),
            nn.Linear(4096, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        output = self.layer6(x)


        return output


if __name__ == '__main__':

    #load dataset
    dataset_all = raw_dataset(root, directory)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset=dataset_all, lengths=[2100, 900],generator=torch.Generator().manual_seed(0))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=0)
    #load model
    model = VGG_16()
    device = torch.device('cpu')
    model = model.to(device)
    #########parameter setting############
    n_epoch = 30
    criterion = nn.CrossEntropyLoss()
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loss = []
    test_accuracy = []
    best_accu = 0.0
    #############train#####################
    for epoch in range(n_epoch):
        model.train()
        loss_aver = 0.0
        len_dataloader = len(train_loader)
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            loss_aver += loss.item()

            sys.stdout.write(
                '\r epoch: %d, [iter: %d / all %d], loss: %f' \
                % (epoch, batch_idx + 1, len_dataloader, loss.detach().numpy()))
            sys.stdout.flush()
            torch.save(model, '{0}/modelterm.pth'.format('./'))


        # val
        print('\n')
        loss_ = loss_aver.cpu().detach().numpy()
        train_loss.append(loss_ / len_dataloader)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, dim=1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        acc_val = correct / total
        test_accuracy.append(acc_val)
        print('Accuracy of current epoch: %f' % (acc_val))
        # save model
        if acc_val == max(test_accuracy):
            torch.save(model, '{0}/modelterm_best.pth'.format('./'))

        print("epoch = {},  loss = {},  acc_val = {}".format(n_epoch, loss_aver, acc_val))

    ########################################################
    x_epoch = np.arange(0, n_epoch, 1)
    plt.figure(1)
    plt.subplot(121)
    plt.subplots_adjust(wspace=0.5)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('train loss')
    plt.plot(x_epoch, train_loss)
    plt.subplot(122)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('test accuracy')
    plt.plot(x_epoch, test_accuracy)
    plt.show()























