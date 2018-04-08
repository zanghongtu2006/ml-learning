#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author Hongtu Zang
# @Time: 18-4-7 下午4:22
import tensorflow as tf

# sess = tf.Session()

# filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once("./blue_jay.jpg"))

filename_queue = tf.FIFOQueue(capacity=100, dtypes=[tf.string])
with tf.Session() as session:
    reader = tf.WholeFileReader()

    key, value = reader.read(filename_queue)
    print "key:", key, "; value:", value
    image = tf.image.decode_jpeg(value)
    flipImageUpDown = tf.image.encode_jpeg(tf.image.flip_up_down(image))
    flipImageLeftRight = tf.image.encode_jpeg(tf.image.flip_left_right(image))

    tf.global_variables_initializer().run(session=session)
    coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(coord=coord, sess=session)

    example = session.run(filename_queue.enqueue("./blue_jay.jpg"))

    # file = open("flippedUpDown.jpg", "wb+")
    # file.write(flipImageUpDown.eval(session=session))
    # file.close()

    file = open("flippedLeftRight.jpg", "wb+")
    file.write(flipImageLeftRight.eval(session=session))
    file.close()

# print "image:", image
# print "leftRight:" + flipImageLeftRight
#
# tf.global_variables_initializer().run(session=sess)
# coord = tf.train.Coordinator()
# threads = tf.train.start_queue_runners(coord=coord, sess=sess)
# example = sess.run(flipImageLeftRight)
# # print example
#
