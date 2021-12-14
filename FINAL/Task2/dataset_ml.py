import pandas as pd
import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
import pickle
from skimage import data, exposure, img_as_float

root = './dataset/image/'
CATEGORY_INDEX = {
    "no_tumor": 0,
    "meningioma_tumor": 1,
    "pituitary_tumor":2,
    "glioma_tumor":3
}

def make_raw_dataset(root, rawlist):
    """#######################################
    导入并解析源域可见光数据，生成原始数据文件
    input：

    output：

    note：

    #######################################"""

    dataset = []

    filename = rawlist['file_name'].values
    filename = (filename).tolist()
    raw_label = rawlist['label'].values
    raw_label = (raw_label).tolist()
    label = []
    #for i in range(len(raw_label)):
    #    label.append(CATEGORY_INDEX[raw_label[i]])

    for j in range(len(filename)):
        folder_path = os.path.join(root, filename[j])
        image = cv.imread(folder_path)
        image = Image.fromarray(np.array(image))
        image = image.convert("L")
        #if CATEGORY_INDEX[raw_label[j]] == 0:
        #    images,labels = data_aug(image,CATEGORY_INDEX[raw_label[j]])
        #    for i in range(len(images)):
        #        image = np.array(images[i],dtype=np.uint8).reshape((512, 512))
        #        dataset.append(image)
        #        label.append(labels[i])

        #else:
        image = np.array(image, dtype=np.uint8).reshape((512, 512))
        dataset.append(image)
        label.append(CATEGORY_INDEX[raw_label[j]])



    return dataset, label

def data_aug(im,label):
    images = []
    labels = []
    images.append(im)
    out1 = im.transpose(Image.ROTATE_180)
    images.append(out1)
    out2 = im.transpose(Image.FLIP_LEFT_RIGHT)
    images.append(out2)
    im = img_as_float(im)
    out3 = exposure.adjust_gamma(im, 1.5)  # 调暗
    images.append(out3)
    out4 = exposure.adjust_gamma(im, 0.75)  # 调亮
    images.append(out4)
    for num in range(len(images)):
        labels.append(label)
    return images, labels




if __name__ == "__main__":
    FileList = pd.read_csv("./dataset/label.csv")
    data,label = make_raw_dataset(root, FileList)

    pickle.dump(data, open("./dataset/data.p" , "wb"))
    pickle.dump(label, open("./dataset/label.p", "wb"))
