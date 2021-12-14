import random
import os
import numpy as np
import matplotlib.pyplot as plt
from dataset_preprocess import *
from sklearn.model_selection import train_test_split
from sklearn import svm
import pickle
from sklearn import preprocessing
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import SelectKBest, chi2
from pylab import *
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

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
    #clf = svm.SVC(C=1,
    #              kernel='rbf',
    #              decision_function_shape='ovo')
    pca = PCA(n_components=150, whiten=True, random_state=42)
    svc = svm.SVC(kernel='rbf', class_weight='balanced')
    model = make_pipeline(pca, svc)
    #clf = svm.LinearSVC()
    return model


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





#ax.scatter(Y[:,0][y == 0], Y[:,1][y == 0], marker='o', s=50, label='Bern')
#ax.scatter(Y[:,0][y == 1], Y[:,1][y == 1], marker='x', s=10, label='Perth')


#from sklearn.decomposition import PCA

#pca = PCA(n_components=2)
#1_2
#_component = pca.fit_transform(df.values)

#pca_1_2 = pd.DataFrame(np.dot(df.values, eigenvector[0:2].T))
#pca_1_2['categories'] = output.values

#sns.set(rc={'figure.figsize': (11, 8)})
#ax = sns.scatterplot(data=pca_1_2, x=0, y=1, hue='categories', s=40,
#                     palette={0: 'r', 1: 'b', 2: 'g', 3: 'purple', 4: 'orange'})



if __name__ == '__main__':
    data,label = load_data(root)
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=0)
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    model = classifier()
    print(model.get_params().keys())
    param_grid = {'svc__C': [1, 5, 10],
                  'svc__gamma': [0.0001, 0.0005, 0.001],
                  'svc__kernel': ['linear','poly','rbf']}
    grid = GridSearchCV(model, param_grid)
    grid.fit(preprocessing.scale(x_train.reshape(2100,-1)), y_train)
    print(grid.best_params_)
    model = grid.best_estimator_
    y_predict = model.predict(preprocessing.scale(x_test.reshape(900,-1)))
    #fig, ax = plt.subplots(4, 6)
    #for i, axi in enumerate(ax.flat):
    #    axi.imshow(x_test[i].reshape(512, 512), cmap='bone')
    #    axi.set(xticks=[], yticks=[])
    #    axi.set_ylabel(CATEGORY_INDEX[y_predict[i]],
    #                   color='black' if y_predict[i] == y_test[i] else 'red')
    #fig.suptitle('Predicted Names; Incorrect Labels in Red', size=14)
    #fig.show()



    print(classification_report(y_test, y_predict))



    #x_train = np.array(x_train)
    #x_test = np.array(x_test)
    #y_train = np.array(y_train)
    #y_test = np.array(y_test)
    #train(clf, preprocessing.scale(x_train.reshape(2100,-1)), y_train)
    #print_accuracy(clf, preprocessing.scale(x_train.reshape(2100,-1)), y_train, preprocessing.scale(x_test.reshape(900,-1)), y_test)
    #print('training prediction:%.3f' % (clf.score(preprocessing.scale(x_train.reshape(2100,-1)), y_train)))
    #print('test data prediction:%.3f' % (clf.score(preprocessing.scale(x_test.reshape(900,-1)), y_test)))
