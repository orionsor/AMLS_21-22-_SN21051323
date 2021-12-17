from sklearn.model_selection import train_test_split
from sklearn import svm
import pickle
from sklearn import preprocessing
from sklearn.decomposition import PCA
from pylab import *
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
import os


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
    """Make pipeline of PCA and Linear SVM"""
    pca = PCA(n_components=1500, whiten=True, random_state=42)
    svc = svm.SVC(kernel='linear', class_weight='balanced')
    model = make_pipeline(pca, svc)
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
    param_grid = {'svc__C': [1, 5, 10,15,100]
                }
    """Traverse all possible parameter in the list to find the optimal one"""
    grid = GridSearchCV(model, param_grid)
    grid.fit(preprocessing.scale(x_train.reshape(3371,-1)), y_train)
    print(grid.best_params_)
    model = grid.best_estimator_
    y_predict = model.predict(preprocessing.scale(x_test.reshape(1445,-1)))

    """Performance reports"""
    print(model.score(preprocessing.scale(x_test.reshape(1445, -1)), y_test))
    print(classification_report(y_test, y_predict))

