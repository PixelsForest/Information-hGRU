#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import information_toolbox_tf as it
import scipy.io as sio
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.client import timeline
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
"""
this is the tensorflow version of the IB_main.py
this file was initially built to make the calculation of MI faster
However, the binning method became six times slower
"""

"""define some useful functions"""


def add_layer(inputs, in_size, out_size, activation_function=None):
    """
     inputs: input data matrix, one sample in a line
     in_size: length of the last layer
     out_size: length of this layer
     activation_function: sigmoid
     return: values in this layer
    """
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    bias = tf.Variable(tf.random_normal([1, out_size]))
    Wx_plus_b = tf.matmul(inputs, Weights)+bias
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


def shuffle(a, b):
    """Shuffle the arrays randomly"""
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

"""network"""

cfg = {}
cfg['SGD_LEARNINGRATE'] = 0.0004
cfg['NUM_EPOCHS'] = 100
cfg['ACTIVATION'] = 'tanh'
# cfg['ACTIVATION'] = 'relu'
# cfg['ACTIVATION'] = 'softsign'
# cfg['ACTIVATION'] = 'softplus'

# How many hidden neurons to put into each of the layers
cfg['LAYER_DIMS'] = [10, 7, 5, 4, 3]  # original IB network
cfg['NUM_BATCHS'] = 16
# kde/bin method
cfg['KDE'] = False
cfg['BIN'] = False
cfg['NUM_BINS'] = 30
cfg['VARIANCE'] = 0.001
# lists to store plot data
axis_1 = []
axis_2 = []
axis_3 = []
axis_4 = []
# remember epoch index
epoch_index = []

print ('Loading Data...')
data = sio.loadmat('datasets/var_u.mat')
F = data['F']  # 4096*12
y = data['y'].T  # 4096*1


with tf.device('/gpu:0'):
    # input data and labels
    input_layer = tf.placeholder(tf.float32, [None, F.shape[1]])
    labels = tf.placeholder(tf.float32, [None, y.shape[1]])

    # hidden layers
    l1 = add_layer(input_layer, F.shape[1], cfg['LAYER_DIMS'][0], activation_function=tf.nn.tanh)
    l2 = add_layer(l1, cfg['LAYER_DIMS'][0], cfg['LAYER_DIMS'][1], activation_function=tf.nn.tanh)
    l3 = add_layer(l2, cfg['LAYER_DIMS'][1], cfg['LAYER_DIMS'][2], activation_function=tf.nn.tanh)
    l4 = add_layer(l3, cfg['LAYER_DIMS'][2], cfg['LAYER_DIMS'][3], activation_function=tf.nn.tanh)
    l5 = add_layer(l4, cfg['LAYER_DIMS'][3], cfg['LAYER_DIMS'][4], activation_function=tf.nn.tanh)
    # output layer
    output_layer = add_layer(l5, cfg['LAYER_DIMS'][4], y.shape[1], activation_function=tf.nn.sigmoid)
    # loss function
    cross_entropy = -tf.reduce_sum(labels * tf.log(output_layer) + (1 - labels) * tf.log(1 - output_layer))
    loss = tf.reduce_mean(cross_entropy)

    train_step = tf.train.GradientDescentOptimizer(cfg['SGD_LEARNINGRATE']).minimize(loss)

    # Example code for calling functions from information_toolbox.py
    '''
    import information_toolbox as it
    l1_digitize = it.digitize(num_bins, l1)

    '''
    # I(X;T) in BIN METHOD
    I_XT1_BIN = it.bin_calc_information(input_layer, l1, cfg['NUM_BINS'])
    I_XT2_BIN = it.bin_calc_information(input_layer, l2, cfg['NUM_BINS'])
    I_XT3_BIN = it.bin_calc_information(input_layer, l3, cfg['NUM_BINS'])
    I_XT4_BIN = it.bin_calc_information(input_layer, l4, cfg['NUM_BINS'])
    I_XT5_BIN = it.bin_calc_information(input_layer, l5, cfg['NUM_BINS'])
    # I(Y;T) in BIN METHOD
    I_YT1_BIN = it.bin_calc_information(labels, l1, cfg['NUM_BINS'])
    I_YT2_BIN = it.bin_calc_information(labels, l2, cfg['NUM_BINS'])
    I_YT3_BIN = it.bin_calc_information(labels, l3, cfg['NUM_BINS'])
    I_YT4_BIN = it.bin_calc_information(labels, l4, cfg['NUM_BINS'])
    I_YT5_BIN = it.bin_calc_information(labels, l5, cfg['NUM_BINS'])

# training and print loss
init = tf.global_variables_initializer()
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))####################
sess.run(init)
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE) ########################
run_metadata = tf.RunMetadata()  #################################
# SGD
for epoch in range(cfg['NUM_EPOCHS']):
    # randomly shuffle the data
    F, y = shuffle(F, y)
    # pass through every sample in one training epoch
    for batch in range(cfg['NUM_BATCHS']):
        batch_size = F.shape[0]/cfg['NUM_BATCHS']
        sess.run(train_step, feed_dict={input_layer: F[batch*batch_size:(batch+1)*batch_size, :],
                                        labels: y[batch*batch_size:(batch+1)*batch_size, :]})
        if epoch % 20 == 0 and batch % 8 == 0:
            print("epoch: %s, batch: %s, loss: %s" % (epoch, batch, sess.run(loss, feed_dict={input_layer: F, labels: y})))
    if epoch % 20 == 0:
        # calculate mutual information
        epoch_index.append(float(epoch) / cfg['NUM_EPOCHS'])
        if cfg['BIN']:
            for j in range(1, len(cfg['LAYER_DIMS']) + 1, 1):
                exec ("x%s = sess.run(I_XT%s_BIN, feed_dict={input_layer: F, labels: y}, options=run_options, run_metadata=run_metadata)" % (j, j))  #######
                exec ("y%s = sess.run(I_YT%s_BIN, feed_dict={input_layer: F, labels: y}, options=run_options, run_metadata=run_metadata)" % (j, j))  ##########
            # store MI
            axis_1.append([x1, x2, x3, x4, x5])
            axis_2.append([y1, y2, y3, y4, y5])
        if cfg['KDE']:
            for j in range(1, len(cfg['LAYER_DIMS']) + 1, 1):
                exec ("T%s = sess.run(l%s, feed_dict={input_layer: F})" % (j, j))
                exec ("xu%s, xl%s, yu%s, yl%s = it.kde_calc_information(F, y, T%s, cfg['VARIANCE'])" % (j, j, j, j, j))
            # store MI
            axis_3.append([(xu1+xl1)/2, (xu2+xl2)/2, (xu3+xl3)/2, (xu4+xl4)/2, (xu5+xl5)/2])
            axis_4.append([(yu1+yl1)/2, (yu2+yl2)/2, (yu3+yl3)/2, (yu4+yl4)/2, (yu5+yl5)/2])


#######################
# Create the Timeline object, and write it to a json
tl = timeline.Timeline(run_metadata.step_stats)
ctf = tl.generate_chrome_trace_format()
with open('timeline.json', 'w') as f:
    f.write(ctf)

# report the final accuracy
prediction_class = np.round(sess.run(output_layer, feed_dict={input_layer: F}))
accuracy = float(np.sum(y == prediction_class))/F.shape[0]
print('the final accuracy of DNN is: %s' % accuracy)

# plot
font = {'family': 'serif',
        'color': 'black',
        'weight': 'normal',
        'size': 16}
cmap = plt.cm.get_cmap('gnuplot')

if cfg['BIN']:
    fig = plt.figure()
    for i in range(len(axis_1)):
        plt.plot(axis_1[i], axis_2[i], marker='o', linestyle='-', markersize=12,
                 markeredgewidth=0.01, linewidth=0.2, color=cmap(epoch_index[i]))

    ax = plt.gca()
    ax.set_xticks(np.linspace(1, 13, 13))
    ax.set_yticks(np.linspace(0, 1, 6))
    # size of xytick font
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    ax.set_xlabel('I(X;T)', fontdict=font)
    ax.set_ylabel('I(Y;T)', fontdict=font)
    plt.title('IB_BIN')
    plt.show()
if cfg['KDE']:
    fig = plt.figure()
    for i in range(len(axis_3)):
        plt.plot(axis_3[i], axis_4[i], marker='o', linestyle='-', markersize=12,
                 markeredgewidth=0.01, linewidth=0.2, color=cmap(epoch_index[i]))

    ax = plt.gca()
    ax.set_xticks(np.linspace(1, 13, 13))
    ax.set_yticks(np.linspace(0, 1, 6))
    # size of xytick font
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    ax.set_xlabel('I(X;T)', fontdict=font)
    ax.set_ylabel('I(Y;T)', fontdict=font)
    plt.title('IB_KDE')
    plt.show()


# save the data
if cfg['BIN']:
    axis_1 = np.array(axis_1)
    axis_2 = np.array(axis_2)
    epoch_index = np.array(epoch_index)
    np.save('IB_IXT_BIN.npy', axis_1)
    np.save('IB_IYT_BIN.npy', axis_2)
    np.save('IB_epoch_color_BIN', epoch_index)
if cfg['KDE']:
    axis_3 = np.array(axis_3)
    axis_4 = np.array(axis_4)
    epoch_index = np.array(epoch_index)
    np.save('IB_IXT_KDE.npy', axis_3)
    np.save('IB_IYT_KDE.npy', axis_4)
    np.save('IB_epoch_color_KDE', epoch_index)