import pandas as pd
import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
import pickle
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms as T
import torch



CATEGORY_INDEX = {
    "negative": 0,
    "positive": 1
}
root = './dataset/image/'
directory = './dataset/binary_label.csv'

class raw_dataset(Dataset):
    def __init__(self, root,directory):
        self.img, self.label = self.read_dataset(directory)
        self.root = root
        self.transforms = T.Compose([
            T.Resize([224,224]),
            T.CenterCrop(224),
            T.ToTensor(),
        ])

    def __getitem__(self, index):
        img_path = os.path.join(self.root,self.img[index])
        data = Image.open(img_path)
        if data.mode != "RGB":
            data = data.convert("RGB")
        image = self.transforms(data)
        labels = self.label[index]
        return image, labels

    def __len__(self):
        return len(self.img)

    def read_dataset(self,directory):
        CATEGORY_INDEX = {
            "negative": 0,
            "positive": 1}
        FileList = pd.read_csv(directory)
        filename = FileList['file_name'].values
        filename = (filename).tolist()
        raw_label = FileList['binary_label'].values
        raw_label = (raw_label).tolist()
        rawlabel = []
        for i in range(len(raw_label)):
            rawlabel.append(CATEGORY_INDEX[raw_label[i]])
        return filename, rawlabel






if __name__ == "__main__":
    train = raw_dataset(root,directory)
    src_loader = torch.utils.data.DataLoader(train, batch_size=30, shuffle=True, num_workers=0)
    for i, data in enumerate(src_loader):
        inputs, labels = data
        print(inputs.shape, labels)