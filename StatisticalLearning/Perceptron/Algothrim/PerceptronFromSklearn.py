from sklearn.linear_model import Perceptron
import numpy as np 

# 训练数据集
X_train = np.array([[3,3],[4,3],[1,1]])
y_train = np.array([1,1,-1])

# 构建perceptron对象,训练数据并输出结果
perceptron = Perceptron(penalty=None, alpha=0.0001, eta0=1, max_iter=5,tol=None)
# perceptron = Perceptron(penalty=None, alpha=0.0001, eta0=0.5, max_iter=5,tol=1e-3)
perceptron.fit(X_train,y_train)
print("w:",perceptron.coef_,"\n","b:",perceptron.intercept_,"\n","n_inter",perceptron.n_iter_)
    
#测试模型的准确率
res = perceptron.score(X_train,y_train)
print("correct rate:{:.0%}".format(res))
