import pandas as pd
import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
import pickle


root = './dataset/image/'
CATEGORY_INDEX = {
    "no_tumor": 0,
    "meningioma_tumor": 1,
    "pituitary_tumor":2,
    "glioma_tumor":3
}

def make_raw_dataset(root, rawlist):
    """Import images and labels, implement data augmentation to balance dataset,
           generate and save files of data and labels for later used in main module.
        """
    dataset = []

    filename = rawlist['file_name'].values
    filename = (filename).tolist()
    raw_label = rawlist['label'].values
    raw_label = (raw_label).tolist()
    label = []


    for j in range(len(filename)):
        folder_path = os.path.join(root, filename[j])
        image = cv.imread(folder_path)
        image = Image.fromarray(np.array(image))
        image = image.convert("L")
        image = np.array(image, dtype=np.uint8).reshape((512, 512))
        dataset.append(image)
        label.append(CATEGORY_INDEX[raw_label[j]])

    return dataset, label



if __name__ == "__main__":
    FileList = pd.read_csv("./dataset/label.csv")
    data,label = make_raw_dataset(root, FileList)

    pickle.dump(data, open("./dataset/data.p" , "wb"))
    pickle.dump(label, open("./dataset/label.p", "wb"))
