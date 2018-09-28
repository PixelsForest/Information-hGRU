"""
This file is used to do experimental plotting for hGRU data.
Finding the fitting value for MI calculation parameters is time-consuming.
You can use KDE & binning method to plot each layer's info trajectory individually.
"""
import tensorflow as tf
import numpy as np
import os
from information_toolbox import *
import matplotlib.pyplot as plt
import h5py
import time

os.environ["CUDA_VISIBLE_DEVICES"]=""

DO_MODE = "kde"
axis_1 = []
axis_2 = []
#step_list = np.arange(1, 28000, 3000) # np.unique(np.logspace(np.log2(1.), np.log2(float(28000)), 100, dtype=int, base=2)) - 1
a = np.arange(1, 3000, 300)
b = np.arange(3000, 12000, 1000)
c = np.arange(12000, 14000, 300)
d = np.arange(14000, 16000, 30)
e = np.arange(16000, 18000, 200)
f = np.arange(18000, 56000, 6000)
g = np.arange(18301, 24001, 300)
h = np.arange(24301, 27901, 300)

list_step = np.concatenate([a, b, c, d, e, f, g, h])
list_step = np.sort(list_step)

for step in list_step: # [1, 27301]
    print("start dealing with data from step %s ..." % step)

    t0 = time.time()
    with h5py.File('/media/data_cifs/xinhao/hgru-data-compresseddims-2epochs-4timesteps/step_%s_val_images.h5' % step, 'r') as hdf:
        images = np.array(hdf.get('images')[1:10])
        #print(images.shape) 16384*150*150*1
    t1 = time.time()
    print("loading and transforming images spent: %s s" % (t1 - t0))

    with h5py.File('/media/data_cifs/xinhao/hgru-data-compresseddims-2epochs-4timesteps/step_%s_val_labels.h5' % step, 'r') as hdf:
        labels = np.array(hdf.get('labels'))
    t2 = time.time()
    print("loading and transforming labels spent: %s s" % (t2 - t1))

    with h5py.File('/media/data_cifs/xinhao/hgru-data-compresseddims-2epochs-4timesteps/step_%s_val_l1.h5' % step, 'r') as hdf:
        l1 = np.array(hdf.get('l1'))
        #print("l1", np.max(l1), np.min(l1))
        l1_ub = np.max(l1)
        l1_lb = np.min(l1)
    t3 = time.time()
    print("loading and transforming l1 spent: %s s" % (t3 - t2))

    with h5py.File('/media/data_cifs/xinhao/hgru-data-compresseddims-2epochs-4timesteps/step_%s_val_hgru.h5' % step, 'r') as hdf:
        hgru = np.array(hdf.get('hgru'))
        # print("hgru", np.max(hgru), np.min(hgru))
        hgru_ub = np.max(hgru)
        hgru_lb = np.min(hgru)
    t4 = time.time()
    print("loading and transforming hgru spent: %s s" % (t4 - t3))

    with h5py.File('/media/data_cifs/xinhao/hgru-data-compresseddims-2epochs-4timesteps/step_%s_val_output.h5' % step, 'r') as hdf:
        output = np.array(hdf.get('output'))
        print(output.shape)
        output_ub = np.max(output)
        output_lb = np.min(output)
    t5 = time.time()
    print("loading and transforming output spent: %s s" % (t5 - t4))


    _t = time.time() - t0
    print("step %s: loading and transforming data spent %s s" % (step, _t))

    if DO_MODE == "bin":
        t0 = time.time()
        MI_XT1 = bin_calc_information(images, l1, 0, 80000, 400)
        MI_YT1 = bin_calc_information(labels, l1, 0, 80000, 400)

        MI_XT2 = bin_calc_information(images, hgru, -2400, 1500, 40)
        MI_YT2 = bin_calc_information(labels, hgru, -2400, 1500, 40)

        MI_XT3 = bin_calc_information(images, output, -12, 12, 60)
        MI_YT3 = bin_calc_information(labels, output, -12, 12, 60)

        _t = time.time() - t0
        print("step %s: calculating MI spent %s s" % (step, _t))

        axis_1.append([MI_XT1, MI_XT2, MI_XT3])  # MI_XT3
        axis_2.append([MI_YT1, MI_YT2, MI_YT3])  # MI_YT3

    if DO_MODE == "kde":
        # rescale the data because kde method will be influenced by data range
        l1 = l1 / 80000.0
        hgru = hgru / 3900.0
        output = output / 24.0

        t0 = time.time()
        MI_XT1_upper, MI_XT1_lower, MI_YT1_upper, MI_YT1_lower = kde_calc_information(images, labels, l1,
                                                                                      2*1e-8)# 
        MI_XT2_upper, MI_XT2_lower, MI_YT2_upper, MI_YT2_lower = kde_calc_information(images, labels, hgru,
                                                                                      1*1e-5)# 1*e-5
        MI_XT3_upper, MI_XT3_lower, MI_YT3_upper, MI_YT3_lower = kde_calc_information(images, labels, output,
                                                                                      0.0004)
        _t = time.time() - t0
        print("step %s: calculating MI spent %s s" % (step, _t))

        axis_1.append(
            [(MI_XT1_upper + MI_XT1_lower) / 2, (MI_XT2_upper + MI_XT2_lower) / 2, (MI_XT3_upper + MI_XT3_lower) / 2])
        axis_2.append(
            [(MI_YT1_upper + MI_YT1_lower) / 2, (MI_YT2_upper + MI_YT2_lower) / 2, (MI_YT3_upper + MI_YT3_lower) / 2])

    print("step %s finished." % step)


np.save('/media/data_cifs/xinhao/hgru-plot/axis_1.npy', np.array(axis_1))
np.save('/media/data_cifs/xinhao/hgru-plot/axis_2.npy', np.array(axis_2))

# print(axis_1)
# print(axis_2)

font = {'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 16}
cmap = plt.cm.get_cmap('gnuplot')

fig = plt.figure()
for i in range(len(axis_1)):
    plt.plot(axis_1[i], axis_2[i], marker='o', linestyle='-', markersize=12,
                     markeredgewidth=0.01, linewidth=0.2, color=cmap(float(i)/len(axis_1)))

ax = plt.gca()
#ax.set_xticks(np.linspace(0, 50, 11))
#ax.set_yticks(np.linspace(0, 1, 6))
# size of xytick font
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

ax.set_xlabel('I(X;T)', fontdict=font)
ax.set_ylabel('I(Y;T)', fontdict=font)
plt.title('IB-' + DO_MODE)
# plt.savefig("IB_BIN_ITERATION%s(loss:%s).png" % (iteration, sess.run(loss, feed_dict={input_layer: F, labels: y})))
plt.show()





