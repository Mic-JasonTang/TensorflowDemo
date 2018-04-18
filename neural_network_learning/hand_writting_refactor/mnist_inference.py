#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/2/20 19:43
# @Author  : Jasontang
# @Site    : 
# @File    : mnist_inference.py
# @ToDo    : 定义了前向传播的过程及神经网络的参数


import tensorflow as tf

# 定义神经网络结构相关的参数
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500


# 训练时会创建这些变量，测试时会通过保存的模型加载这些变量的取值
def get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))

    # 当使用正则化生成函数时,当前变量的正则化损失加入名字为losses的集合.
    # 自定义集合
    if regularizer:
        tf.add_to_collection("losses", regularizer(weights))
    return weights


# 前向传播过程
def inference(input_tensor, regularizer):
    # 声明第一层神经网络的变量并完成前向传播过程
    with tf.variable_scope("layer1"):
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    # 声明第二层圣经网络变量并完成前向传播过程
    with tf.variable_scope("layer2"):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases
    # 返回最后前向传播的结果
    return layer2
