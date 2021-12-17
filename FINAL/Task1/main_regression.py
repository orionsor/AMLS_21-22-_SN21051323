import random
import os
import numpy as np
import matplotlib.pyplot as plt
from dataset_preprocess import *
from sklearn.model_selection import train_test_split
import pickle
from sklearn import preprocessing
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import SelectKBest, chi2
from pylab import *
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression



"""set path to saved files of data and labels"""
root = './dataset/'
data_root = 'data.p'
label_root = 'label.p'





def load_data(root):
    filepath_data = os.path.join(root, data_root)
    filepath_label = os.path.join(root, label_root)
    data = pickle.load(open(filepath_data, "rb"))
    label = pickle.load(open(filepath_label, "rb"))
    return data,label


def classifier():
    """Make pipeline of PCA and logistic regression"""
    pca = PCA(n_components=1500, whiten=True, random_state=42)
    LR = LogisticRegression()
    model = make_pipeline(pca, LR)
    return model


def show_samples(root):
    """not used in formal training
       used in train to check state of the dataset"""
    filepath_data = os.path.join(root, data_root)
    data = pickle.load(open(filepath_data, "rb"))
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(data[i])
        plt.axis('off')

    plt.show()


if __name__ == '__main__':
    data,label = load_data(root)
    print(len(data))
    x_train, x_test, y_train, y_test = train_test_split(data, label,test_size=0.3, random_state=0)
    print(len(x_train))
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    model = classifier()
    """Train model"""
    model.fit(preprocessing.scale(x_train.reshape(3371, -1)), y_train)
    y_predict = model.predict(preprocessing.scale(x_test.reshape(1445,-1)))
    """Performance report"""
    print(model.score(preprocessing.scale(x_test.reshape(1445,-1)),y_test))
    print(classification_report(y_test, y_predict))


