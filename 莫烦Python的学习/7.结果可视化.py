# -*- coding: utf-8 -*-
__author__ = 'Mr.Finger'
__date__ = '2017/10/16 17:16'
__site__ = ''
__software__ = 'PyCharm'
__file__ = '7.结果可视化.py'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def add_layer(inputs, in_size, out_size, activation_fun=None):
    Weights = tf.Variable(tf.random_uniform([in_size, out_size]))  # 我们在这里定义的时候，如果变量是矩阵，我们就大写首字母
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_fun is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_fun(Wx_plus_b)
    return outputs


if __name__ == '__main__':
    x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
    noise = np.random.normal(0, 0.05, x_data.shape)
    y_data = np.square(x_data) - 0.5 + noise

    xs = tf.placeholder(tf.float32, [None, 1])  # 这里的None表示无论给出多少个样本都是可以的哦
    ys = tf.placeholder(tf.float32, [None, 1])

    l1 = add_layer(xs, 1, 10, activation_fun=tf.nn.relu)
    prediction = add_layer(l1, 10, 1, activation_fun=None)
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                        reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x_data, y_data)
    plt.ion()
    plt.show() # 在老版本的python上使用的是plt.show(block=False), 但是在新版的python我们使用的一般是plt.ion()

    for i in range(100000):
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i % 50 == 0:
            # print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass

            prediction_value = sess.run(prediction, feed_dict={xs: x_data})
            lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
            plt.pause(0.1)
