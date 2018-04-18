#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/2/8 21:16
# @Author  : Jasontang
# @Site    : 
# @File    : MNIST_HandWritting.py
# @ToDo    : 手写数字体识别完整的例子


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# MNIST数据集相关常数的定义
INPUT_NODE = 784  # 输入层的结点数。这里就是图片的像素
OUTPUT_NODE = 10  # 输出层的结点数，这里等于类别的数目

# 配置神经网络的参数
LAYER1_NODE = 500  # 隐层结点数
BATCH_SIZE = 100  # 一个训练batch中的训练数据个数。

LEARNING_RATE_BASE = 0.8  # 初始学习率
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率
REGULARIZATION_RATE = 0.0001  # 描述模型复杂度的正则化项在损失函数中的系数（lambda)
TRAINING_STEPS = 30000  # 训练轮数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率


# 给定神经网络所有输入和所有参数，计算神经网络的前向传播结果。
# 定义了一个使用ReLU激活函数实现了去线性化。
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    # 当没有提供滑动平均类时，直接使用参数当前的取值
    if avg_class is None:
        # 计算隐层的前向传播结果，这里使用了ReLU激活函数。
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)

        # 计算输出层的前向传播结果。应为在计算损失函数时会一并计算sofmax函数
        # 所以这里不需要加入激活函数。而且不加入softmax不会影响预测结果。因为
        # 预测时使用的是不同类别对应结点输出值的相对大小，有没有softmax层对最后分类
        # 结果的计算没有影响。
        return tf.matmul(layer1, weights2) + biases2
    else:
        # 首先使用avg_class.average函数来计算得出变量的滑动平均值
        # 然后再计算相应的神经网络前向传播结果
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) +
                            avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)


# 通过tf.variable_scope和tf.get_variable函数来改进上述函数
def inference2(input_tensor, reuse=False):
    # 定义第一层神经网络的变量和前向传播过程
    with tf.variable_scope("layer1", reuse=reuse):
        # 根据传入的reuse来判断是创建新变量还是使用已经创建好的。
        # 在第一次构造网络时需要创建新的变量，以后每次调用这个函数都直接使用reuse=True就
        # 不需要每次讲变量传入了
        weights = tf.get_variable("weights",
                                  [INPUT_NODE, LAYER1_NODE],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases",
                                 [LAYER1_NODE],
                                 initializer=tf.constant_initializer)
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    # 类似地定义第二层神经网络的变量和前向传播过程
    with tf.variable_scope("layer2", reuse=reuse):
        weights = tf.get_variable("weights",
                                  [LAYER1_NODE, OUTPUT_NODE],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases",
                                 [OUTPUT_NODE],
                                 initializer=tf.constant_initializer)
        layer2 = tf.matmul(layer1, weights) + biases
    # 返回前向传播结果
    return layer2


# 训练模型
def train(mnist):
    x = tf.placeholder(tf.float32, shape=[None, INPUT_NODE], name="x-input")
    y_ = tf.placeholder(tf.float32, shape=[None, OUTPUT_NODE], name="y-output")

    # 生成隐层的参数
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

    # 生成输出层的参数
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 计算在当前参数下神经网络前向传播的结果。
    y = inference(x, None, weights1, biases1, weights2, biases2)

    # 如果使用了改进的方法，则这样生成y
    # y = inference2(x)

    # 定义存储训练轮数的变量，这个变量不需要计算滑动平均值，所以这个指定这个变量为不可
    # 训练的变量（trainable=False)。在使用Tensorflow训练圣经网络时，一般会将
    # 代表训练轮数的变量指定为不可训练的参数
    global_step = tf.Variable(0, trainable=False)

    # 给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类。
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

    # 在所有代表神经网络参数的变量上使用滑动平均，其他辅助变量（比如global_step)就不需要。
    # tf.trainable_variables犯规的就是图上集合GraphKeys.TRAINABLE_VERIABLES中的元素。
    # 这个集合的元素就是没有指定trainable=False的参数
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # 计算使用了滑动平均之后的前向传播结果。滑动平均不会改变原变量的值
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)
    # 这样调用即可
    # average_y = inference2(x, True)

    # 计算交叉熵作为刻画预测值和真实值之间差距的损失函数。这里使用TensorFlow中提供的
    # sparse_softmax_cross_entropy_with_logits函数来计算交叉熵。当分类问题只有一个正确答案时
    # 可以使用这个函数来加速交叉熵的计算。MNIST问题的图片中只包含0-9中的一个数字，所以可以使用这个函数来计算交叉熵
    # 损失。这个函数的第一个参数是神经网络不包括softmax层的前向传播结果，第二个是训练数据的正确答案。
    # 因为标准答案是一个长度为10的一维数组，而该函数需要提供一个正确答案的数组，所说义要使用tf.argmax函数
    # 来得到正确答案对应的类别编号。
    # 计算两个概率之间的差距
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    # 计算在当前batch中所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 计算L2正则化损失函数,下面会返回一个函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 计算模型的正则化损失。一般只计算神经网络边上权重的正则化损失，而不使用偏置项
    # 正则化是为了希望通过限制权重的大小，使得模型不能任意拟合训练数据中的随机噪音
    regularization = regularizer(weights1) + regularizer(weights2)
    # 总损失等于交叉熵损失和正则化损失的和
    loss = cross_entropy_mean + regularization  # 相当于目标函数

    # 设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,  # 基础学习率，随着迭代进行，学习率在此基础上衰减
        global_step,  # 当前迭代的轮数
        mnist.train.num_examples / BATCH_SIZE,  # 过完所有的训练数据需要的迭代次数
        LEARNING_RATE_DECAY)  # 学习率衰减速度

    # 使用tf.train.GradientDescentOptimizer优化算法来优化损失函数
    # 这里的损失函数包含了交叉熵损失和L2正则化损失
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)

    # 在训练神经网络模型时，每过一遍数据既需要通过反向传播来更新神经网络中的参数
    # 又要更新每一个参数的滑动平均值。为了一次完成多个操作，TensorFlow提供了tf.control_dependencies和
    # tf.group两种机制。
    # 参考：https://www.w3cschool.cn/tensorflow_python/tensorflow_python-mgxu2f1p.html
    # train_op = tf.group(train_step, variable_averages_op)
    with tf.control_dependencies([train_step, variable_averages_op]):
        # 只是对此上下文中的train_step和variable_averages_op进行更新操作。不作其他的操作(no_op)
        # 参考：http://blog.csdn.net/PKU_Jade/article/details/73498753
        train_op = tf.no_op(name="train")

    # 检验使用滑动平均模型的神经网络前向传播结果是否正确。tf.argmax(average_y, 1)
    # 计算每一个样例的预测答案。其中average_y是一个batch_size * 10的二维数组，每一行
    # 表示一个样例的前向传播结果。tf.argmax的第二个参数1表示取得最大值操作仅在第一个维度（行）中进行。
    # 即在每一行取最大值对应的下标。于是得到的结果是一个长度为batch的一维数组，这个一维数组中的值就表示
    # 了每一个样例对应的数字识别结果。tf.equal判断两个张量的每一维是否相等，如果相等返回True，否则False
    correct_predicting = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    # 这个运算首先将一个布尔型的数值转换为实数型，然后计算平均值。这个平均值就是模型在这一组数据上的正确率
    accuracy = tf.reduce_mean(tf.cast(correct_predicting, tf.float32))

    # 初始化会话并开始训练过程
    with tf.Session() as sess:
        tf.initialize_all_variables().run()

        # 准备验证数据。一般在神经网络的训练过程中会通过验证数据来大致判断停止的条件和评判训练的效果
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

        # 准备测试数据
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                # 计算滑动平均模型在验证数据上的效果。
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d traing step(s), validation accuracy using average model is %g" % (i, validate_acc))

            # 产生这一轮使用的一个batch的训训练数据
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        # 在训练结束后，在测试数据上检测神经网络模型的最终正确率
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After %d traings step(s), test accuracy using average model is %g" % (TRAINING_STEPS, test_acc))


# 主程序入口
def main(argv=None):
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    train(mnist)


# TensorFlow提供的一个主程序入口，tf.app.run会调用上面定义的main函数
if __name__ == '__main__':
    tf.app.run()
