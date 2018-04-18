#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/2/4 15:42
# @Author  : Jasontang
# @Site    : 
# @File    : five_layer_network_use_collection.py
# @ToDo    : 使用集合来定义5层神经网络带L2正则化的损失函数

import tensorflow as tf
from numpy.random import RandomState


# 获取一层神经网络边上的权重，并将这个权重的L2正则化损失加入名称为losses的集合中
def get_weight(shape, lambda_):
    var = tf.Variable(tf.random_normal(shape))
    # add_to_collection函数将这个新生成变量的L2正则化损失项加入集合。
    tf.add_to_collection("losses", tf.contrib.layers.l1_regularizer(lambda_)(var))
    return var


x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))


rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
print(X)
Y = [[int(x1 + x2) < 1] for (x1, x2) in X]

batch_size = 8
# 定义了每一层网络节点的个数
layer_dimension = [2, 10, 10, 10, 1]
# 神经网络的层数
n_layers = len(layer_dimension)

# 这个变量维护前向传播时最深层的节点，开始的时候就是输入层。
cur_layer = x
# 当前层的结点个数
in_dimension = layer_dimension[0]

# 通过一个循环来生成5层全连接的神经网络结构
for i in range(1, n_layers):
    # layer_dimension[i]为下一层的结点个数
    out_dimension = layer_dimension[i]
    # 生成当前层中权重的变量，并将这个变量的L2正则化损失加入计算图上的集合
    weight = get_weight([in_dimension, out_dimension], 0.001)
    bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
    # 使用ReLu激活函数
    cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight) + bias)
    # 进入下一层之前将下一层的结点个数更新为当前层结点个数
    in_dimension = layer_dimension[i]

# 在定义神经网络向前传播的同时已经将所有的L2正则化损失加入了图上的集合，
# 这里只需要计算刻画模型在训练数据集上表现的损失函数
mse_loss = tf.reduce_mean(tf.square(y_ - cur_layer))

# 将均方误差损失函数加入损失集合
tf.add_to_collection("losses", mse_loss)
# 返回一个列表
loss = tf.get_collection("losses")
# 将列表元素进行相加
loss_add = tf.add_n(loss)

with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    print(sess.run(loss, feed_dict={x: X, y_: Y}))
    print(sess.run(loss_add, feed_dict={x: X, y_: Y}))
