import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Perceptron(object):
    """
    Perceptron classifier.
    Parameters(参数)
    ------------
    eta : float
    Learning rate (between 0.0 and 1.0) 学习效率
    n_iter : int
    Passes over the training dataset(数据集).
    Attributes（属性）
    -----------
    w_ : 1d-array
    Weights after fitting.
    errors_ : list
    Number of misclassifications in every epoch（时间起点）.
    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
        
    def fit(self, X, y):
        '''
        Fit training data.
        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features] X的形式是列矩阵
        Training vectors, where n_samples is the number of samples
        and n_features is the number of features.
        y : array-like, shape = [n_samples]
        Target values.
        Returns
        -------
        self : object
        '''
        self.w_ = np.zeros(1 + X.shape[1])
        # zeros()创建了一个 长度为 1+X.shape[1] = 1+n_features 的 0数组
        #初始化权值为0
        # self.w_ 权向量
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update #更新权值，x0 =1
                errors += int(update != 0.0)
            self.errors_.append(errors) #每一步的累积误差
        return self

    def net_input(self, X):
        """Calculate net input"""
        return (np.dot(X, self.w_[1:])+self.w_[0])

    def predict(self, X):
        """return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)

def plot_decision_regions(X, y, classifier, resolution=0.02):
    #setup marker generator and colormap
    markers = ('o','x','s','^','v')
    colors = ('red','blue','lightgreen','gray','cyan')
    cmap = ListedColormap(colors[: len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:,0].min() -1, X[:,0].max()+1
    x2_min, x2_max = X[:,1].min() -1, X[:,1].max()+1
    # X[:,k] 冒号左边表示行范围，读取所有行，冒号右边表示列范围，读取第K列
    xx1, xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),
                           np.arange(x2_min,x2_max,resolution))
    #arange(start,end,step) 返回一个一维数组
    #meshgrid(x,y)产生一个以x为行，y为列的矩阵
    #xx1是一个(305*235)大小的矩阵 xx1.ravel()是将所有的行放在一个行里面的长度71675的一维数组
    #xx2同理
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T) #我们其实调用predict()方法预测了grid矩阵当中的每一个点
    #np.array([xx1.ravel(), xx2.ravel()]) 生成了一个 (2*71675)的矩阵
    # xx1.ravel() = (1,71675)
    #xx1.shape = (305,205) 将Z重新调整为(305,205)的格式
    Z = Z.reshape(xx1.shape) 

    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)

    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    print(np.unique(y))
    # idx = 0,1 cl = -1 1
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker = markers[idx],label = cl)

def main():

    filepath = '../DataSets/Iris.csv'
    df = pd.read_csv(filepath,header=None)
    # print(df.tail())

    #.iloc[0:100,4] 读取前100行的序号为4（第5列数据）
    y = df.iloc[0:100, 4].values # .values将dataframe中的值存进一个list中
    y = np.where(y=='Iris-setosa',-1,1) #如果是 Iris-setosa y=-1否则就是1 （二元分类）

    X = df.iloc[0:100,[0,2]].values
    #.iloc[0:100,[0:2]] 读取前100行的 前两列的数据

    plt.scatter(X[:50,0],X[:50,1],c='red',marker='o',label='setosa')
    plt.scatter(X[50:100,0],X[50:100,1],c='blue',marker='x',label='versicolor')
    plt.xlabel('sepal length')
    plt.ylabel('petal length')
    plt.legend(loc='upper left')
    plt.show()

    ppn = Perceptron(eta=0.1,n_iter=10)
    ppn.fit(X, y)
    plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker='o')
    plt.xlabel('Epoches')
    plt.ylabel('Number of misclassification')
    plt.xlim(1,10)
    plt.savefig('Number of misclassification-Epoches.png',bbox_inches='tight')
    plt.show()

    plot_decision_regions(X,y,classifier=ppn)
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc = 'upper left')
    plt.savefig(' decision_regions.png')
    plt.show()

if __name__ == "__main__":
    main()