import random
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
import pickle
from sklearn import preprocessing
from sklearn.decomposition import PCA
from pylab import *
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline


root = './dataset/'
data_root = 'data.p'
label_root = 'label.p'


def load_data(root):
    """Load data from saved files generated in 'dataset_preprocess.py' """
    filepath_data = os.path.join(root, data_root)
    filepath_label = os.path.join(root, label_root)
    data = pickle.load(open(filepath_data, "rb"))
    label = pickle.load(open(filepath_label, "rb"))
    return data,label


def classifier():
    """Make pipeline of PCA and kernel SVM"""
    pca = PCA(n_components=1500, whiten=True, random_state=42)
    svc = svm.SVC(kernel='rbf', class_weight='balanced')
    model = make_pipeline(pca, svc)
    return model




if __name__ == '__main__':
    data,label = load_data(root)
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=0)
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    model = classifier()
    print(model.get_params().keys())
    """Candidate list of hyper parameters"""
    param_grid = {'svc__C': [1, 5, 10,15,100],
                  'svc__gamma': [0.0001, 0.0005, 0.001,0.01,0.1,1]
                }
    grid = GridSearchCV(model, param_grid)
    grid.fit(preprocessing.scale(x_train.reshape(3371,-1)), y_train)
    print(grid.best_params_)
    model = grid.best_estimator_
    y_predict = model.predict(preprocessing.scale(x_test.reshape(1445,-1)))

    print(model.score(preprocessing.scale(x_train.reshape(3371, -1)), y_train))
    print(model.score(preprocessing.scale(x_test.reshape(1445, -1)), y_test))
    print(classification_report(y_test, y_predict))



