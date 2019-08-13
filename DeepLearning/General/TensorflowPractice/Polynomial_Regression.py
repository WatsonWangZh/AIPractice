# 0. 导包
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (14,8)

# 1. 数据准备
n_observations = 100
xs = np.linspace(-3, 3, n_observations)
ys = np.sin(xs) + np.random.uniform(-0.5, 0.5, n_observations)
plt.scatter(xs, ys)
plt.show()

# 2.准备好placeholder
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

# 3.初始化参数/权重
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# 4.计算预测结果
Y_pred = tf.add(tf.multiply(X, W), b)
#添加高次项
W_2 = tf.Variable(tf.random_normal([1]), name='weight_2')
Y_pred = tf.add(tf.multiply(tf.pow(X, 2), W_2), Y_pred)
W_3 = tf.Variable(tf.random_normal([1]), name='weight_3')
Y_pred = tf.add(tf.multiply(tf.pow(X, 3), W_3), Y_pred)

# 5.计算损失函数值
sample_num = xs.shape[0]
loss = tf.reduce_sum(tf.pow(Y_pred - Y, 2)) / sample_num

# 6.初始化optimizer
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# 7.指定迭代次数，并在session里执行graph
n_samples = xs.shape[0]
init = tf.global_variables_initializer()
with tf.Session() as sess:
	# 初始化所有变量
    sess.run(init) 
    
    writer = tf.summary.FileWriter('./graphs/polynomial_reg', sess.graph)

    # 训练模型
    for i in range(1000):
        total_loss = 0
        for x, y in zip(xs, ys):
            # 通过feed_dic把数据灌进去
            _, l = sess.run([optimizer, loss], feed_dict={X: x, Y:y}) 
            total_loss += l
        if i%20 ==0:
            print('Epoch {0}: {1}'.format(i, total_loss/n_samples))

    # 关闭writer
    writer.close()
    # 取出w和b的值
    W, W_2, W_3, b = sess.run([W, W_2, W_3, b])

# 8. 绘制拟合结果
plt.plot(xs, ys, 'bo', label='Real data')
plt.plot(xs, xs*W + np.power(xs,2)*W_2 + np.power(xs,3)*W_3 + b, 'r', label='Predicted data')
plt.legend()
plt.show()
