# -*- coding: utf-8 -*-

# Sample code to use string producer.

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """




    o_h = np.zeros(n)
    o_h[x] = 1.
    return o_h


num_classes = 3
batch_size = 10

# --------------------------------------------------
#
#       DATA SOURCE
#
# --------------------------------------------------

def dataSource(paths, batch_size):

    min_after_dequeue = 10
    capacity = min_after_dequeue + 3 * batch_size

    example_batch_list = []
    label_batch_list = []

    for i, p in enumerate(paths):
        filename = tf.train.match_filenames_once(p)
        filename_queue = tf.train.string_input_producer(filename, shuffle=False)
        reader = tf.WholeFileReader()
        _, file_image = reader.read(filename_queue)
        image, label = tf.image.decode_jpeg(file_image), one_hot(i, num_classes) #[float(i)]
        image=tf.image.rgb_to_grayscale(image, name=None)
        image = tf.image.resize_image_with_crop_or_pad(image, 80, 140)
        image = tf.reshape(image, [80, 140, 1])
        image = tf.to_float(image) / 255. - 0.5
        example_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity,
                                                          min_after_dequeue=min_after_dequeue)
        example_batch_list.append(example_batch)
        label_batch_list.append(label_batch)

    example_batch = tf.concat(values=example_batch_list, axis=0)
    label_batch = tf.concat(values=label_batch_list, axis=0)

    return example_batch, label_batch


# --------------------------------------------------
#
#       MODEL
#
# --------------------------------------------------

def myModel(X, reuse=False):
    with tf.variable_scope('ConvNet', reuse=reuse):
        o1 = tf.layers.conv2d(inputs=X, filters=32, kernel_size=3, activation=tf.nn.relu)
        o2 = tf.layers.max_pooling2d(inputs=o1, pool_size=2, strides=2)
        o3 = tf.layers.conv2d(inputs=o2, filters=64, kernel_size=3, activation=tf.nn.relu)
        o4 = tf.layers.max_pooling2d(inputs=o3, pool_size=2, strides=2)
        o5 = tf.layers.conv2d(inputs=o4, filters=128, kernel_size=3, activation=tf.nn.relu)
        o6 = tf.layers.max_pooling2d(inputs=o5, pool_size=2, strides=2)

        h = tf.layers.dense(inputs=tf.reshape(o6, [batch_size * num_classes, 8 * 15 * 128]), units=20, activation=tf.nn.relu)
        y = tf.layers.dense(inputs=h, units=num_classes, activation=tf.nn.softmax)
    return y

example_batch_train, label_batch_train = dataSource(["Dataset/train/0/*.jpg", "Dataset/train/1/*.jpg", "Dataset/train/2/*.jpg"], batch_size=batch_size)
example_batch_valid, label_batch_valid = dataSource(["Dataset/valid/0/*.jpg", "Dataset/valid/1/*.jpg", "Dataset/valid/2/*.jpg"], batch_size=batch_size)
example_batch_test, label_batch_test = dataSource(["Dataset/test/0/*.jpg", "Dataset/test/1/*.jpg", "Dataset/test/2/*.jpg"], batch_size=batch_size)

example_batch_train_predicted = myModel(example_batch_train, reuse=False)
example_batch_valid_predicted = myModel(example_batch_valid, reuse=True)
example_batch_test_predicted = myModel(example_batch_test, reuse=True)

cost = tf.reduce_sum(tf.square(example_batch_train_predicted - tf.cast(label_batch_train, tf.float32)))
cost_valid = tf.reduce_sum(tf.square(example_batch_valid_predicted - tf.cast(label_batch_valid, tf.float32)))
# cost = tf.reduce_mean(-tf.reduce_sum(label_batch * tf.log(y), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)
#optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
#optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9).minimize(cost)
# --------------------------------------------------
#
#       TRAINING
#
# --------------------------------------------------

# Add ops to save and restore all the variables.

saver = tf.train.Saver()

with tf.Session() as sess:

    file_writer = tf.summary.FileWriter('./logs', sess.graph)

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    error_valid = 0
    errors_valid = []
    previousError = 9999
    for epoch in range(200):#430
        sess.run(optimizer)
        error_valid = sess.run(cost_valid)
        errors_valid.append(error_valid)
        if epoch != 0 : previousError = errors_valid[len(errors_valid) - 2]
        if epoch % 20 == 0:
            print("Iter:", epoch, "---------------------------------------------")
            print(sess.run(label_batch_train))
            print(sess.run(example_batch_train_predicted))
            print("Error:", error_valid)
            print ("Error:", sess.run(cost))


        if np.absolute(error_valid - previousError) < (0.002):
            epoh = epoch
            break

    print("Final Epoch: ", epoh)

    plt.plot(errors_valid)
    plt.show()


    bad = 0
    re = []
    la = []
    cuenta = 0
    for epoch in range(9):
        result = sess.run(example_batch_test_predicted)
        lab = sess.run(label_batch_test)
        cuenta += len(lab)
        re.extend(result)
        la.extend(lab)
    for b, r in zip(la, re):
        if np.argmax(b) != np.argmax(r):
            print (b, "-->", r, " *** Se ha clasificado mal")
            bad = bad + 1
        else:
            print(b, "-->", r, " *** Se ha clasificado bien")


    print ("\nSe han clasificado mal", bad, "muestras de",cuenta ,".")


    save_path = saver.save(sess, "./tmp/model.ckpt")
    print("Model saved in file: %s" % save_path)
            
    coord.request_stop()
    coord.join(threads)
