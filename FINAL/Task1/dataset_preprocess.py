import pandas as pd
import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt

root = './dataset/image/'
CATEGORY_INDEX = {
    "negative": 0,
    "positive": 1
}
def make_raw_dataset(root,rawlist):
    """#######################################
    导入并解析源域可见光数据，生成原始数据文件
    input：

    output：

    note：

    #######################################"""

    dataset = []

    filename = rawlist['file_name'].values
    filename = (filename).tolist()
    raw_label = rawlist['binary_label'].values
    raw_label = (raw_label).tolist()
    label = []
    for i in range(len(raw_label)):
        label.append(CATEGORY_INDEX[raw_label[i]])

    for j in range(len(filename)):
        folder_path = os.path.join(root, filename[j])
        image = cv.imread(folder_path)
        dataset.append(image)
    return dataset, label







if __name__ == '__main__':

    FileList = pd.read_csv("./dataset/binary_label.csv")
    data,label = make_raw_dataset(root, FileList)





