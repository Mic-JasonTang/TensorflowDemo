#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/2/21 16:32
# @Author  : Jasontang
# @Site    : 
# @File    : mnist_eval.py
# @ToDo    : 测试过程


import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import neural_network_learning.hand_writting_refactor.mnist_inference as mnist_inference
import neural_network_learning.hand_writting_refactor.mnist_train as mnist_train

# 每10s加载一次最新模型，并在测试数据上测试最新模型的正确率
EVAL_INTERVAL_SECS = 10


def evaluate(mnist):
    x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name="input-x")
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name="input-y")

    validate_feed = {x: mnist.validation.images,
                     y_: mnist.validation.labels}

    y = mnist_inference.inference(x, None)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # 每隔EVAL_INTERVAL_SECS秒调用一次计算正确率的过程以检测训练过程中正确率的变化
    stop_count = 0
    while True:
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
            # 停止条件 #
            stop_count += EVAL_INTERVAL_SECS
            if stop_count == mnist_train.TRAING_STEPS:
                return
            # 停止条件 #
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                # 通过文件名得到模型保存时迭代的轮数
                # 输出./model/model.ckpt-29001
                print(ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split("/")[-1].split("-")[-1]
                accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                print("After %s training step(s), validation accuracy is %g" % (global_step, accuracy_score))
            else:
                print("No checkpoint file found")
                return
        time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)
    evaluate(mnist)


if __name__ == '__main__':
    tf.app.run()