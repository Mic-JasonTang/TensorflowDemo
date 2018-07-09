#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/8 17:20
# @Author  : Jasontang
# @Site    : 
# @File    : data_pre_with_tf.py
# @ToDo    : 数据增强


import tensorflow as tf
import matplotlib.pyplot as plt


image_raw_data = tf.gfile.FastGFile("cat.jpg", 'rb').read()

# print(image_raw_data)
with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    # print(img_data.eval())
    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
    # print(img_data.eval())
    resized = tf.image.resize_images(img_data, (300, 300), method=2)
    # print(resized.shape)
    print(resized.get_shape())

    # 调整图像大小，缩小时，裁剪，放大时，用0填充。
    croped = tf.image.resize_image_with_crop_or_pad(img_data, 1000, 1000)
    padded = tf.image.resize_image_with_crop_or_pad(img_data, 3000, 3000)

    # 按比例调整图像
    central_cropped = tf.image.central_crop(img_data, 0.5)

    # 图像翻转
    flipped = tf.image.flip_up_down(img_data)
    flipped2 = tf.image.flip_left_right(img_data)
    # 将图像沿对角线翻转
    transposed = tf.image.transpose_image(img_data)

    # 以一定概率上下翻转图像
    flipped3 = tf.image.random_flip_left_right(img_data)

    # 以一定概率左右翻转图像
    flipped4 = tf.image.random_flip_up_down(img_data)

    # 将图像亮度-0.5
    adjusted = tf.image.adjust_brightness(img_data, -0.5)
    # 将图像亮度+0.5
    adjusted2 = tf.image.adjust_brightness(img_data, 0.5)
    # 将[-1, 1)的范围随机调整图像的亮度
    adjusted3 = tf.image.random_brightness(img_data, 1)
    print(img_data.eval().max(), img_data.eval().min())

    # 调整图像对比度
    adjusted_contrast_1 = tf.image.adjust_contrast(img_data, -5)
    adjusted_contrast_2 = tf.image.adjust_contrast(img_data, 5)
    # lower 不能为负数，因为计算公式为 (x - mean) * contrast_factor + mean
    adjusted_contrast_3 = tf.image.random_contrast(img_data, lower=1, upper=10)

    # 调整图像色相
    adjusted_hue = tf.image.adjust_hue(img_data, 0.1)
    adjusted_hue2 = tf.image.adjust_hue(img_data, 0.6)
    # max_delta must be <= 0.5.
    adjusted_hue3 = tf.image.random_hue(img_data, max_delta=0.5)

    # 调整图像饱和度
    adjusted_saturation = tf.image.adjust_saturation(img_data, -5)
    adjusted_saturation2 = tf.image.adjust_saturation(img_data, 5)
    # lower must be non-negative
    adjusted_saturation3 = tf.image.random_saturation(img_data, lower=0, upper=5)

    # 均值变为0，方差变为1
    adjusted_standardization = tf.image.per_image_standardization(img_data)

    # 处理标注框
    # 将图像缩小，这样可视化能让标注框更清楚
    img_data = tf.image.resize_images(img_data, (180, 267), method=1)
    # tf.image.draw_bounding_boxes函数要求图像矩阵中的数字为实数，所说需要先讲图像矩阵
    # 转化为实数类型。tf.image.draw_bounding_boxes函数图像的输入时一个batch的，也就是
    # 多张图片组成的四维矩阵，所以需要将解码之后的图像矩阵加一维。
    print(img_data.eval().shape)
    # axis参数表示在四维矩阵的第几维加入一维
    batched = tf.expand_dims(tf.image.convert_image_dtype(img_data, tf.float32), axis=0)
    print(batched.eval().shape)
    # 给出每一张图像的所有标注框. 标注框有4个数字，分别代表[Ymin, Xmin, Ymax, Xmax]
    # 这里给出的数字都是图像的相对位置。
    # [0.35, 0.47, 0.5, 0.56] 代表了从(180*0.35=63, 267*0.47=125)到(9.5*180， 0.56*150)的图像。
    # 必须是三维的,shape=[1,2,4]
    boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])
    print(boxes.shape)
    result = tf.image.draw_bounding_boxes(batched, boxes)
    print(result.eval().shape)
    print(tf.shape(img_data).eval())
    # 可以通过提供标注框的方式来告诉随机截取图像的算法那些部分是“有信息量”的
    begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
        tf.shape(img_data), bounding_boxes=boxes, min_object_covered=0.1)
    print("begin:{}, size:{}, boxes:{}".format(begin.eval(), size.eval(), bbox_for_draw.eval()))
    # 通过标注框可视化随机截取得到的图像。
    batched = tf.expand_dims(tf.image.convert_image_dtype(img_data, tf.float32), 0)
    image_with_box = tf.image.draw_bounding_boxes(batched, bbox_for_draw)
    # 截取随机出来的图像。
    distorted_image = tf.slice(img_data, begin, size)
    print(distorted_image.eval().shape)
    plt.imshow(distorted_image.eval())
    # plt.subplot(221)
    # plt.imshow(img_data.eval())
    # plt.subplot(222)
    # plt.imshow(adjusted_standardization.eval())
    # plt.subplot(223)
    # 使用这个就不需要执行第20行的代码。
    # img_data = tf.image.rgb_to_grayscale(img_data)
    # wh = img_data.eval().shape[0] * img_data.eval().shape[1]
    # plt.hist(tf.reshape(img_data.eval(), (wh, -1)).eval())
    # plt.imshow(adjusted_saturation2.eval())
    # plt.subplot(224)
    # plt.hist(tf.reshape(adjusted_standardization.eval(), [wh, -1]).eval())
    # plt.imshow(adjusted_saturation3.eval())
    plt.show()


