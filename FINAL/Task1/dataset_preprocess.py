import pandas as pd
import numpy as np
import os
import cv2 as cv
from PIL import Image
import pickle


root = './dataset/image/'
CATEGORY_INDEX = {
    "negative": 0,
    "positive": 1
}
def make_raw_dataset(root, rawlist):
    """Import images and labels, implement data augmentation to balance dataset,
       generate and save files of data and labels for later used in main module.
    """

    dataset = []

    filename = rawlist['file_name'].values
    filename = (filename).tolist()
    raw_label = rawlist['binary_label'].values
    raw_label = (raw_label).tolist()
    label = []
    #for i in range(len(raw_label)):
    #    label.append(CATEGORY_INDEX[raw_label[i]])

    for j in range(len(filename)):
        folder_path = os.path.join(root, filename[j])
        image = cv.imread(folder_path)
        image = Image.fromarray(np.array(image))
        image = image.convert("L")
        if CATEGORY_INDEX[raw_label[j]] == 0:
            images,labels = data_aug(image,CATEGORY_INDEX[raw_label[j]])
            for i in range(len(images)):
                image = np.array(images[i],dtype=np.uint8).reshape((512, 512))
                dataset.append(image)
                label.append(labels[i])

        else:
            image = np.array(image, dtype=np.uint8).reshape((512, 512))
            dataset.append(image)
            label.append(CATEGORY_INDEX[raw_label[j]])



    return dataset, label

def data_aug(im,label):
    """Data Augmentation:
    1.rotate image 180 degrees,
    2.flip image from left to right
    3.decrease image brightness
    4.equalize Histogram of image to increase contrast """
    images = []
    labels = []
    images.append(im)
    out1 = im.transpose(Image.ROTATE_180)
    images.append(out1)
    out2 = im.transpose(Image.FLIP_LEFT_RIGHT)
    images.append(out2)
    im = np.array(im, dtype=np.uint8)
    out3 = im * 0.5
    images.append(out3)
    out4 = cv.equalizeHist(im)
    images.append(out4)

    for num in range(len(images)):
        labels.append(label)
    return images, labels




if __name__ == "__main__":
    FileList = pd.read_csv("./dataset/binary_label.csv")
    data,label = make_raw_dataset(root, FileList)

    pickle.dump(data, open("./dataset/data.p" , "wb"))
    pickle.dump(label, open("./dataset/label.p", "wb"))
