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

def plot_svc_decision_function(model,ax=None,plot_support = True):
    "绘制二维的决定方程"
    if ax is None:
        ax = plt.gca() # 绘制一个子图对象
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    x = np.linspace(xlim[0],xlim[1],30)
    y = np.linspace(xlim[0],ylim[1],30)
    Y, X  = np.meshgrid(y,x)
    xy = np.vstack([X.ravel(),Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # 绘制决策边界
    ax.contour(X,Y,P,colors='k',levels=[-1,0,1],alpha=0.5,linestyles=['--','-','--'])

    # 绘制支持向量
    if plot_support:
        ax.scatter(model.support_vectors_[:,0],model.support_vectors_[:,1],s = 300,linewidth=1,facecolors='none')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)



if __name__ == '__main__':

    feature_ratio = np.linspace(0.5, 0.99, 20)
    # 保留数据的特征从50% 尝试到 99%

    x_shape = []  # 按照保存特征比例进行 PCA 降维之后，数据的维度保存在这个列表中
    scores = []  # 每次降维后的数据的评分保存在这里面
    data, label = load_data(root)


    for i in feature_ratio:
        pca = PCA(i)
        data = np.array(data)
        label = np.array(label)
        data = preprocessing.scale(data.reshape(3000, -1))
        x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=0)
        svc = svm.SVC(kernel='rbf', class_weight='balanced', C=10, gamma=0.001)
        svc.fit(x_train_reduced, y_train)
        y_predict = svc.predict(x_test_reduced)

        score = svc.score(x_test, y_test)

        x_shape.append(x_train_reduced.shape[1])

        scores.append(score)

    plt.plot(x_shape, scores)
    plt.xlabel('number of features')
    plt.ylabel('accuracy')
    plt.show()



    #pca = PCA(n_components=2, whiten=True, random_state=42)



    #plt.scatter(x_test_reduced[:, 0][y_test == 0], x_test_reduced[:, 1][y_test == 0], marker='o', s=50, cmap='negative')
    #plt.scatter(x_test_reduced[:, 0][y_test == 1], x_test_reduced[:, 1][y_test == 1], marker='x', s=50, cmap='positive')
    #ax = plt.gca()  # 获取当前的子图，如果不存在，创建新的子图
    #xlim = ax.get_xlim()
    #ylim = ax.get_ylim()
    # 要画决策边界，必须要有网格
    #axisx = np.linspace(xlim[0], xlim[1], 30)
    #axisy = np.linspace(ylim[0], ylim[1], 30)
    #axisy, axisx = np.meshgrid(axisy, axisx)
    # 将特征向量转换为特征矩阵的函数
    # 核心是将两个特征向量广播，获取y.shape * x.shape 这么多个坐标点的横坐标和纵坐标
    #xy = np.vstack([axisx.ravel(), axisy.ravel()]).T
    # ravel()是降维函数，vstack将多个结构一致的一维数组按行堆叠起来

    # 建模， fit计算相应的决策边界
    #P = svc.decision_function(xy).reshape(axisx.shape)
    # decision function 返回每个输入样本对应的到决策边界的距离
    # 然后将这个距离转为axisx的结果
    # 画决策边界和平行于决策边界的超平面
    #ax.contour(axisx, axisy, P, colors="k", levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"])

    #fig, ax = plt.subplots(4, 6)
    #for i, axi in enumerate(ax.flat):
    #    axi.imshow(x_test[i].reshape(512, 512), cmap='bone')
    #    axi.set(xticks=[], yticks=[])
    #    axi.set_ylabel(CATEGORY_INDEX[y_predict[i]],
    #                   color='black' if y_predict[i] == y_test[i] else 'red')
    #fig.suptitle('Predicted Names; Incorrect Labels in Red', size=14)
    #fig.show()

    #plt.show()


    #print(classification_report(y_test, y_predict))



    #x_train = np.array(x_train)
    #x_test = np.array(x_test)
    #y_train = np.array(y_train)
    #y_test = np.array(y_test)
    #train(clf, preprocessing.scale(x_train.reshape(2100,-1)), y_train)
    #print_accuracy(clf, preprocessing.scale(x_train.reshape(2100,-1)), y_train, preprocessing.scale(x_test.reshape(900,-1)), y_test)
    #print('training prediction:%.3f' % (clf.score(preprocessing.scale(x_train.reshape(2100,-1)), y_train)))
    #print('test data prediction:%.3f' % (clf.score(preprocessing.scale(x_test.reshape(900,-1)), y_test)))
