#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/2/8 19:28
# @Author  : Jasontang
# @Site    : 
# @File    : MNIST_demo.py
# @ToDo    : MNIST数据集实验

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

# 载入数据
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print("loaded dataset!")

print("Training data size:", mnist.train.num_examples)

print("Validating data size:", mnist.validation.num_examples)

print("Testing data size:", mnist.test.num_examples)

print("Example training data:", mnist.train.images[0])

print("Example training label:", mnist.train.labels[0])

# 图像显示 数字7
image = mnist.train.images[0].reshape(28, 28)
plt.imshow(image, cmap="gray")
plt.show()

batch_size = 100
xs, ys = mnist.train.next_batch(batch_size)
# 从train的集合中选取batch_size个训练数据
print(type(xs))
print("X shape:", xs.shape)
print("Y shape:", ys.shape)
