# coding=utf-8
"""
This file is used to generate one of the pictures in the pdf file
The figure with nine subplots
"""
from matplotlib import pyplot as plt
import numpy as np


def get_data(timestep, PLOT_MODE):
    x = np.load("/media/cifs-serrelab/xinhao/hgru-plot/axis_1-%stimesteps-allsamples-plotstepmessy.npy" % timestep)
    y = np.load("/media/cifs-serrelab/xinhao/hgru-plot/axis_2-%stimesteps-allsamples-plotstepmessy.npy" % timestep)

    x_l1 = x[:, 0]
    y_l1 = y[:, 0]

    x_hgru = x[:, 1]
    y_hgru = y[:, 1]

    x_output = x[:, 2]
    y_output = y[:, 2]

    if PLOT_MODE == "l1":
	axis_1 = x_l1
	axis_2 = y_l1
    elif PLOT_MODE == "hgru":
	axis_1 = x_hgru
	axis_2 = y_hgru
    elif PLOT_MODE == "output":
	axis_1 = x_output
	axis_2 = y_output
    elif PLOT_MODE == "output-hgru":
	axis_1 = x[:, 1:3]
	axis_2 = y[:, 1:3]
    else:
	axis_1 = x
	axis_2 = y
    return axis_1, axis_2

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

font = {'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 18}
cmap = plt.cm.get_cmap('gnuplot')
plt.style.use('ggplot')

fig, axes = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=0.05, hspace=0.15)




axis_1, axis_2 = get_data(4, "output")
for i in range(len(axis_1)):
    axes[0, 0].plot(axis_1[i], axis_2[i], marker='o', linestyle='-', markersize=6,
                     markeredgewidth=0.01, linewidth=0.2, color=cmap(float(list_step[i]/56000.0)))
axes[0,0].axis([0, 14.5, -0.02, 1.02])
axes[0,0].set_xticks(np.linspace(0, 14, 8))
axes[0,0].set_yticks(np.linspace(0, 1, 6))
axes[0,0].set_title('$Output$ \n \n Four timesteps')
axes[0,0].yaxis.set_label_text('$I(Y;T)$')

axis_1, axis_2 = get_data(4, "hgru")
for i in range(len(axis_1)):
    axes[0, 1].plot(axis_1[i], axis_2[i], marker='o', linestyle='-', markersize=6,
                     markeredgewidth=0.01, linewidth=0.2, color=cmap(float(list_step[i]/56000.0)))
axes[0,1].set_title('$hgru$ \n \n ')



axis_1, axis_2 = get_data(4, "l1")
for i in range(len(axis_1)):
    axes[0, 2].plot(axis_1[i], axis_2[i], marker='o', linestyle='-', markersize=6,
                     markeredgewidth=0.01, linewidth=0.2, color=cmap(float(list_step[i]/56000.0)))
axes[0,2].set_title('$l1$ \n \n ')

axis_1, axis_2 = get_data(6, "output")
for i in range(len(axis_1)):
    axes[1, 0].plot(axis_1[i], axis_2[i], marker='o', linestyle='-', markersize=6,
                     markeredgewidth=0.01, linewidth=0.2, color=cmap(float(list_step[i]/56000.0)))
axes[1,0].yaxis.set_label_text('$I(Y;T)$')
axes[1,0].set_title('Six timesteps')

axis_1, axis_2 = get_data(6, "hgru")
for i in range(len(axis_1)):
    axes[1, 1].plot(axis_1[i], axis_2[i], marker='o', linestyle='-', markersize=6,
                     markeredgewidth=0.01, linewidth=0.2, color=cmap(float(list_step[i]/56000.0)))

axis_1, axis_2 = get_data(6, "l1")
for i in range(len(axis_1)):
    axes[1, 2].plot(axis_1[i], axis_2[i], marker='o', linestyle='-', markersize=6,
                     markeredgewidth=0.01, linewidth=0.2, color=cmap(float(list_step[i]/56000.0)))

axis_1, axis_2 = get_data(8, "output")
for i in range(len(axis_1)):
    axes[2, 0].plot(axis_1[i], axis_2[i], marker='o', linestyle='-', markersize=6,
                     markeredgewidth=0.01, linewidth=0.2, color=cmap(float(list_step[i]/56000.0)))
axes[2,0].yaxis.set_label_text('$I(Y;T)$')
axes[2,0].xaxis.set_label_text('$I(X;T)$')
axes[2,0].set_title('Eight timesteps')


axis_1, axis_2 = get_data(8, "hgru")
for i in range(len(axis_1)):
    axes[2, 1].plot(axis_1[i], axis_2[i], marker='o', linestyle='-', markersize=6,
                     markeredgewidth=0.01, linewidth=0.2, color=cmap(float(list_step[i]/56000.0)))
axes[2,1].xaxis.set_label_text('$I(X;T)$')

axis_1, axis_2 = get_data(8, "l1")
for i in range(len(axis_1)):
    axes[2, 2].plot(axis_1[i], axis_2[i], marker='o', linestyle='-', markersize=6,
                     markeredgewidth=0.01, linewidth=0.2, color=cmap(float(list_step[i]/56000.0)))
axes[2,2].xaxis.set_label_text('$I(X;T)$')

plt.show()
