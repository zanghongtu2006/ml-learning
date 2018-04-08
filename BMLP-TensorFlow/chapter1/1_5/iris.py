#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author Hongtu Zang
# @Time: 18-4-6 上午11:31
import tensorflow as tf

# sess = tf.InteractiveSession()

# filename_queue = tf.train.string_input_producer(
#     tf.train.match_filenames_once("./*.csv"),
#     shuffle=True
# )
filename_queue = tf.FIFOQueue(capacity=100, dtypes=[tf.string])
with tf.Session() as session:
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    record_defaults = [[0.], [0.], [0.], [0.], [""]]
    col1, col2, col3, col4, target = tf.decode_csv(value,
                                                   record_defaults=record_defaults)
    tf.initialize_all_variables().run(session=session)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=session)

    session.run(filename_queue.enqueue("iris.csv"))
    for i in range(100):
        print session.run(target)
        coord.request_stop()
        coord.join(threads)

    # tf.pack由tf.stack取代
    # features = tf.stack([col1, col2, col3, col4])
    # print features
    #
    # tf.initialize_all_variables 由tf.global_variables_initializer
    # tf.global_variables_initializer().run(session=sess)
    #
    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    #
    # for iteration in range(0, 5):
    #     example = sess.run([features])
    #     print(example)
    #     print coord.request_stop()
    #     print coord.join(threads)
