#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/2/21 16:08
# @Author  : Jasontang
# @Site    : 
# @File    : mnist_train.py
# @ToDo    : 定义了神经网络的训练过程

import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import neural_network_learning.hand_writting_refactor.mnist_inference as mnist_inference

# 配置神经网络的参数
BATCH_SIZE = 100
LEARNING_REATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAING_STEPS = 2000
MOVING_AVERAGE_DECAY = 0.99
# 模型保存的路径和文件名
MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "model.ckpt"


def train(mnist):
    # 定义输入输出placeholder
    x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name="input-x")
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name="input-y")

    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    y = mnist_inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_average_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))
    learing_rate = tf.train.exponential_decay(LEARNING_REATE_BASE,
                                              global_step,
                                              mnist.train.num_examples / BATCH_SIZE,
                                              LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learing_rate).minimize(loss, global_step)

    with tf.control_dependencies([train_step, variables_average_op]):
        train_op = tf.no_op(name="train")

    # 初始化持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(TRAING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})

            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (i, loss_value))

                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main(argv=None):
    mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()