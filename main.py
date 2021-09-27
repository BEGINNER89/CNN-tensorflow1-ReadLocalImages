# -*- coding: UTF-8 -*-
import numpy as np
from numpy import *
import os
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
import tensorflow as tf


def listall(dire):
    img_path, label_path = [], []
    for files in os.listdir(dire):
        if files[-1] == 'g':
            img_path.append(dire + files)
        else:
            label_path.append(dire + files)
    return img_path, label_path


def readData(direc):
    im, la = listall(direc)
    # imdata, ladata = [[0]*len(im) ], [0]
    imdata = []
    ladata = []
    for i in range(len(im)):
        x = Image.open(im[i])
        x = x.convert('L')
        tmpdata = np.asarray(x)
        tmpdata = np.reshape(tmpdata[0:101, 0:101], (1, 10201))  # 一行
        imdata.append(tmpdata)

    for j in range(len(la)):
        with open(la[j], 'r') as f:
            str = f.read()
            ladata.append(int(str[0]))

    return imdata, ladata


def get_Batch(data, label, batch_size):
    data = tf.cast(data, tf.string)
    label = tf.cast(label, tf.int32)
    input_queue = tf.train.slice_input_producer([data, label], shuffle=False, num_epochs=1)
    x_batch, y_batch = tf.train.batch(input_queue, batch_size=batch_size)

    y_batch = tf.reshape(y_batch, [batch_size])
    x_batch = tf.cast(x_batch, tf.float32)
    return x_batch, y_batch


def network(datas, labels, testdata, testlabel):
    labels = np.array(labels).reshape([100, 1])
    datas = np.array(datas).reshape([100, 101, 101, 1]) / 255
    testdata = np.array(testdata).reshape([25, 101, 101, 1])
    testlabel = np.array(testlabel).reshape([25, 1])
    batch_size = 5
    n_batchs = len(labels) // batch_size
    datas, labels = np.array(datas), np.array(labels)
    x = tf.placeholder(shape=[None, 101, 101, 1], dtype=tf.float32)
    y = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    w_conv1 = weight_variable([11, 11, 1, 32])
    b_conv1 = biases_vriable([32])
    datax = tf.reshape(x, shape=[-1, 101, 101, 1])

    h_conv1 = tf.nn.relu(conv2d(datax, w_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    w_conv2 = weight_variable([11, 11, 32, 64])
    b_conv2 = biases_vriable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    w_fc1 = weight_variable([26 * 26 * 64, 1024])
    b_fc1 = biases_vriable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, w_fc1.shape[0]])
    h_fc1 = tf.nn.tanh(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    w_fc2 = weight_variable([1024, 10])
    b_fc2 = biases_vriable([10])

    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(100):
            for batch in range(n_batchs):
                a, b = get_Batch(datas, labels, batch_size)
                sess.run(train_step, {x: datas, y: labels, keep_prob: 0.7})
            acc, l = sess.run([accuracy, loss], {x: testdata, y: testlabel, keep_prob: 1.0})
            print("Iter: " + str(epoch) + " Accuracy: " + str(acc) + " Loss: " + str(l))


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def biases_vriable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


if __name__ == '__main__':
    root_dir = r'C:/Users/Administrator/Desktop/job/Face Database/TrainImages/'
    ims, las = readData(root_dir)
    test_dir = r'C:/Users/Administrator/Desktop/job/Face Database/TestImages/'
    imtest, latest = readData(test_dir)
    network(ims, las, imtest, latest)
