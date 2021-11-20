import random
import os
import numpy as np
import matplotlib.pyplot as plt
from dataset_preprocess import *
from sklearn.model_selection import train_test_split
from sklearn import svm
import pickle
from sklearn import preprocessing

"""#######################################
     parameter setting
    #######################################"""

root = './dataset/'
data_root = 'data.p'
label_root = 'label.p'
"""#######################################
     import and load datasets
    #######################################"""

#filepath_data = os.path.join(root,data_root)
#filepath_label = os.path.join(root,label_root)
#data = pickle.load(open(filepath_data, "rb"))
#label = pickle.load(open(filepath_label, "rb"))
#for i  in range(6):
#    plt.subplot(2, 3, i+1)
#    plt.imshow(data[i])
#    plt.axis('off')

#plt.show()




def load_data(root):
    #FileList = pd.read_csv("./dataset/binary_label.csv")
    #data, label = make_raw_dataset(root, FileList)
    filepath_data = os.path.join(root, data_root)
    filepath_label = os.path.join(root, label_root)
    data = pickle.load(open(filepath_data, "rb"))
    label = pickle.load(open(filepath_label, "rb"))
    return data,label


def classifier():
    clf = svm.SVC(C=1,
                  kernel='rbf',
                  decision_function_shape='ovo')
    #clf = svm.LinearSVC()
    return clf


def train(clf, x_train, y_train):

    clf.fit(x_train, y_train)


def show_accuracy(a, b, tip):
    acc = a == b
    print('%s Accuracy:%.3f' % (tip, np.mean(acc)))


def print_accuracy(clf, x_train, y_train, x_test, y_test):
    print('training prediction:%.3f' % (clf.score(x_train, y_train)))
    print('test data prediction:%.3f' % (clf.score(x_test, y_test)))
    show_accuracy(clf.predict(x_train), y_train, 'traing data')
    show_accuracy(clf.predict(x_test), y_test, 'testing data')






if __name__ == '__main__':
    data,label = load_data(root)
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=0)
    clf = classifier()
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    train(clf, preprocessing.scale(x_train.reshape(2100,-1)), y_train)
    print_accuracy(clf, preprocessing.scale(x_train.reshape(2100,-1)), y_train, preprocessing.scale(x_test.reshape(900,-1)), y_test)
    print('training prediction:%.3f' % (clf.score(preprocessing.scale(x_train.reshape(2100,-1)), y_train)))
    print('test data prediction:%.3f' % (clf.score(preprocessing.scale(x_test.reshape(900,-1)), y_test)))





