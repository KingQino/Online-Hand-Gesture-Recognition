# -*- coding: utf-8 -*-
# @Time    : 2020/8/12 6:50 PM
# @Author  : Yinghao Qin
# @Email   : y.qin@hss18.qmul.ac.uk
# @File    : utils5.py
# @Software: PyCharm
import json
import os

import numpy as np
import matplotlib.pyplot as plt


#######################################################################################
# 'utils5' is used to plot accuracy and loss figures.                                 #
#######################################################################################


def plot_accuracy_curve(show=False, save=True, path='results/detector_acc.png'):

    path1 = './results/detector/run-resnet10-8-v2_Jul23_18-01-52_usher-tag-Train_Accuracy.json'
    path2 = './results/detector/run-resnet10-8-v2_Jul23_18-01-52_usher-tag-Val_Accuracy.json'
    with open(path1, 'r') as load_f:
        data1 = json.load(load_f)
    with open(path2, 'r') as load_f:
        data2 = json.load(load_f)

    data1 = [data1[i][2] for i in range(len(data1))]
    data2 = [data2[i][2] for i in range(len(data2))]
    max_train = max(data1)
    max_test = max(data2)

    num = 100
    x_axis = np.linspace(1, num, num, endpoint=True)
    plt.plot(x_axis, data1, 'r', label='Train accuracy')
    plt.plot(x_axis, data2, 'b', label='Test accuracy')
    plt.legend()
    plt.title('Accuracy - ResNet-10')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    if save:
        if not os.path.exists('results'):
            os.mkdir('results')
        else:
            pass
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()
    plt.close()

    return max_train, max_test


def plot_loss_curve(show=False, save=True, path='results/detector_loss.png'):
    path_loss_1 = './results/detector/run-resnet10-8-v2_Jul23_18-01-52_usher-tag-Train_Loss.json'
    path_loss_2 = './results/detector/run-resnet10-8-v2_Jul23_18-01-52_usher-tag-Val_Loss.json'
    with open(path_loss_1, 'r') as load_f:
        data1 = json.load(load_f)
    with open(path_loss_2, 'r') as load_f:
        data2 = json.load(load_f)

    data1 = [data1[i][2] for i in range(len(data1))]
    data2 = [data2[i][2] for i in range(len(data2))]

    num = 100
    x_axis = np.linspace(1, num, num, endpoint=True)
    plt.plot(x_axis, data1, 'r', label='Train loss')
    plt.plot(x_axis, data2, 'b', label='Test loss')
    plt.legend()
    plt.title('Loss - ResNet-10')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    if save:
        if not os.path.exists('results'):
            os.mkdir('results')
        else:
            pass
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()
    plt.close()


def plot_multiple_accuracy_curve(show=True, save=True, path='results/classifier_acc.png'):
    num = 100
    x_axis = np.linspace(1, num, num, endpoint=True)

    path_1 = './results/c3d-16/run-c3d-16_Jul20_02-18-06_phobos-tag-Train_Accuracy.json'
    path_2 = './results/c3d-32/run-c3d-32_Jul20_02-26-06_luna-tag-Train_Accuracy.json'
    path_3 = './results/resnext101-16/run-resnext101-16_Jul31_22-05-06_deimos-tag-Train_Accuracy.json'
    path_4 = './results/resnext101-32/run-resnext101-32_Jul31_22-10-07_deimos-tag-Train_Accuracy.json'
    with open(path_1, 'r') as load_f:
        data1 = json.load(load_f)
    with open(path_2, 'r') as load_f:
        data2 = json.load(load_f)
    with open(path_3, 'r') as load_f:
        data3 = json.load(load_f)
    with open(path_4, 'r') as load_f:
        data4 = json.load(load_f)

    data1 = [data1[i][2] for i in range(len(data1))]
    data2 = [data2[i][2] for i in range(len(data2))]
    data3 = [data3[i][2] for i in range(len(data3))]
    data4 = [data4[i][2] for i in range(len(data4))]

    plt.plot(x_axis, data1, 'r', label='Train: C3D-16')
    plt.plot(x_axis, data2, 'y', label='Train: C3D-32')
    plt.plot(x_axis, data3, 'b', label='Train: ResNeXt101-16')
    plt.plot(x_axis, data4, 'g', label='Train: ResNeXt101-32')

    path_1 = './results/c3d-16/run-c3d-16_Jul20_02-18-06_phobos-tag-Val_Accuracy.json'
    path_2 = './results/c3d-32/run-c3d-32_Jul20_02-26-06_luna-tag-Val_Accuracy.json'
    path_3 = './results/resnext101-16/run-resnext101-16_Jul31_22-05-06_deimos-tag-Val_Accuracy.json'
    path_4 = './results/resnext101-32/run-resnext101-32_Jul31_22-10-07_deimos-tag-Val_Accuracy.json'
    with open(path_1, 'r') as load_f:
        data1 = json.load(load_f)
    with open(path_2, 'r') as load_f:
        data2 = json.load(load_f)
    with open(path_3, 'r') as load_f:
        data3 = json.load(load_f)
    with open(path_4, 'r') as load_f:
        data4 = json.load(load_f)

    data1 = [data1[i][2] for i in range(len(data1))]
    data2 = [data2[i][2] for i in range(len(data2))]
    data3 = [data3[i][2] for i in range(len(data3))]
    data4 = [data4[i][2] for i in range(len(data4))]
    max1 = max(data1)
    max2 = max(data2)
    max3 = max(data3)
    max4 = max(data4)
    maxl = [max1, max2, max3, max4]

    plt.plot(x_axis, data1, 'r--', label='Test: C3D-16')
    plt.plot(x_axis, data2, 'y--', label='Test: C3D-32')
    plt.plot(x_axis, data3, 'b--', label='Test: ResNeXt101-16')
    plt.plot(x_axis, data4, 'g--', label='Test: ResNeXt101-32')

    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    if save:
        if not os.path.exists('results'):
            os.mkdir('results')
        else:
            pass
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()
    plt.close()

    return maxl


def plot_multiple_loss_curve(show=True, save=True, path='results/classifier_loss.png'):
    num = 100
    x_axis = np.linspace(1, num, num, endpoint=True)

    path_1 = './results/c3d-16/run-c3d-16_Jul20_02-18-06_phobos-tag-Train_Loss.json'
    path_2 = './results/c3d-32/run-c3d-32_Jul20_02-26-06_luna-tag-Train_Loss.json'
    path_3 = './results/resnext101-16/run-resnext101-16_Jul31_22-05-06_deimos-tag-Train_Loss.json'
    path_4 = './results/resnext101-32/run-resnext101-32_Jul31_22-10-07_deimos-tag-Train_Loss.json'
    with open(path_1, 'r') as load_f:
        data1 = json.load(load_f)
    with open(path_2, 'r') as load_f:
        data2 = json.load(load_f)
    with open(path_3, 'r') as load_f:
        data3 = json.load(load_f)
    with open(path_4, 'r') as load_f:
        data4 = json.load(load_f)

    data1 = [data1[i][2] for i in range(len(data1))]
    data2 = [data2[i][2] for i in range(len(data2))]
    data3 = [data3[i][2] for i in range(len(data3))]
    data4 = [data4[i][2] for i in range(len(data4))]

    plt.plot(x_axis, data1, 'r', label='Train: C3D-16')
    plt.plot(x_axis, data2, 'y', label='Train: C3D-32')
    plt.plot(x_axis, data3, 'b', label='Train: ResNeXt101-16')
    plt.plot(x_axis, data4, 'g', label='Train: ResNeXt101-32')

    path_1 = './results/c3d-16/run-c3d-16_Jul20_02-18-06_phobos-tag-Val_Loss.json'
    path_2 = './results/c3d-32/run-c3d-32_Jul20_02-26-06_luna-tag-Val_Loss.json'
    path_3 = './results/resnext101-16/run-resnext101-16_Jul31_22-05-06_deimos-tag-Val_Loss.json'
    path_4 = './results/resnext101-32/run-resnext101-32_Jul31_22-10-07_deimos-tag-Val_Loss.json'
    with open(path_1, 'r') as load_f:
        data1 = json.load(load_f)
    with open(path_2, 'r') as load_f:
        data2 = json.load(load_f)
    with open(path_3, 'r') as load_f:
        data3 = json.load(load_f)
    with open(path_4, 'r') as load_f:
        data4 = json.load(load_f)

    data1 = [data1[i][2] for i in range(len(data1))]
    data2 = [data2[i][2] for i in range(len(data2))]
    data3 = [data3[i][2] for i in range(len(data3))]
    data4 = [data4[i][2] for i in range(len(data4))]

    plt.plot(x_axis, data1, 'r--', label='Test: C3D-16')
    plt.plot(x_axis, data2, 'y--', label='Test: C3D-32')
    plt.plot(x_axis, data3, 'b--', label='Test: ResNeXt101-16')
    plt.plot(x_axis, data4, 'g--', label='Test: ResNeXt101-32')

    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    if save:
        if not os.path.exists('results'):
            os.mkdir('results')
        else:
            pass
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()
    plt.close()


plot_accuracy_curve(show=True)
plot_loss_curve(show=True)
plot_multiple_accuracy_curve()
plot_multiple_loss_curve()

# max_detector_acc_train, max_detector_acc_test = plot_accuracy_curve(show=True)
# print(max_detector_acc_train)
# print(max_detector_acc_test)
# plot_loss_curve(show=True)
# max_list = plot_multiple_accuracy_curve()
# # print(max_list)
# plot_multiple_loss_curve()
