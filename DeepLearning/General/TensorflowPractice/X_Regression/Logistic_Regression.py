# coding: utf-8

# 0. 导包
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

# 1. 数据读取
#使用tensorflow自带的工具加载MNIST手写数字集合
mnist = input_data.read_data_sets('./data/mnist', one_hot=True)
#查看一下数据维度
# print(mnist.train.images.shape)
#查看target维度
# print(mnist.train.labels.shape)

# 2.准备好placeholder
batch_size = 128
X = tf.placeholder(tf.float32, [batch_size, 784], name='X_placeholder') 
Y = tf.placeholder(tf.int32, [batch_size, 10], name='Y_placeholder')

# 3.准备好参数/权重
w = tf.Variable(tf.random_normal(shape=[784, 10], stddev=0.01), name='weights')
b = tf.Variable(tf.zeros([1, 10]), name="bias")

# 4.拿到每个类别的score
logits = tf.matmul(X, w) + b 

# 5.计算多分类softmax的loss function
# 求交叉熵损失
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y, name='loss')
# 求平均
loss = tf.reduce_mean(entropy)

# 6.准备好optimizer
learning_rate = 0.01
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# 7.在session里执行graph里定义的运算

#迭代总轮次
n_epochs = 30
init = tf.global_variables_initializer()
with tf.Session() as sess:
    # 在Tensorboard里可以看到图的结构
    writer = tf.summary.FileWriter('./graphs/logistic_reg', sess.graph)

    start_time = time.time()
    sess.run(init)	
    n_batches = int(mnist.train.num_examples/batch_size)
    for i in range(n_epochs): # 迭代这么多轮
        total_loss = 0

        for _ in range(n_batches):
            X_batch, Y_batch = mnist.train.next_batch(batch_size)
            _, loss_batch = sess.run([optimizer, loss], feed_dict={X: X_batch, Y:Y_batch}) 
            total_loss += loss_batch
        print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))

    print('Total time: {0} seconds'.format(time.time() - start_time))

    print('Optimization Finished!')

    # 测试模型
    preds = tf.nn.softmax(logits)
    correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
	
    n_batches = int(mnist.test.num_examples/batch_size)
    total_correct_preds = 0
 
    for i in range(n_batches):
        X_batch, Y_batch = mnist.test.next_batch(batch_size)
        accuracy_batch = sess.run([accuracy], feed_dict={X: X_batch, Y:Y_batch}) 
        total_correct_preds += accuracy_batch[0]
    print('Accuracy {0}'.format(total_correct_preds/mnist.test.num_examples))
    writer.close()
