# -*- coding: utf-8 -*-
# @Time    : 2020/8/4 10:18 AM
# @Author  : Yinghao Qin
# @Email   : y.qin@hss18.qmul.ac.uk
# @File    : online_test.py
# @Software: PyCharm
import json
import math
import pdb
import timeit

import cv2 as cv
import torch
from torch import nn

from Networks.c3d import C3D
from Networks.resnet import resnet10
from Networks.resnext import resnext101
from utils4 import online_video_processing


#######################################################################################
# You should run the file on Terminal (Command Line Window),                          #
# like "python online_test.py"                                                        #
# The parameters that you are possible to change are listed below                     #
#   path_detector: the path to the detector                                           #
#   path_classifier: the path to the classifier                                       #
#######################################################################################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

# load detector and classifier
path_detector = '/Users/apple/PycharmProjects/C3D/runs/resnet10-8-v2/resnet10-Jester_epoch-99.pth.tar'
path_detector_label_dic = './annotation_jester/label_dic_binary_gesture.json'
detector = resnet10(num_classes=2, sample_size=112, sample_duration=8)
detector.to(device)
checkpoint = torch.load(path_detector, map_location=lambda storage, loc: storage)
detector.load_state_dict(checkpoint['state_dict'])
detector.eval()

path_classifier = '/Users/apple/PycharmProjects/C3D/runs/resnext101-16/resnext101-Jester_epoch-99.pth.tar'
path_classifier_label_dic = './annotation_jester/label_dic.json'
clr_frame_nb = 16
classifier = resnext101(num_classes=27, sample_size=112, sample_duration=clr_frame_nb)
classifier.to(device)
checkpoint_ = torch.load(path_classifier, map_location=lambda storage, loc: storage)
classifier.load_state_dict(checkpoint_['state_dict'])
classifier.eval()

# load classes dictionary
category_index_classifier = {}
try:
    json_file = open(path_classifier_label_dic, 'r')
    class_dict = json.load(json_file)
    category_index_classifier = {v: k for k, v in class_dict.items()}
except Exception as e:
    print(e)
    exit(-1)


def camera(window_name, camera_id, save_dir, cycle_limit, window_size, window_queue):
    """

    :param window_name: the name of the display window
    :param camera_id: id of the opened video capturing device. 0 is the local camera
    :param save_dir: the save directory of the video
    :param cycle_limit: it is a cycle parameter to help store frames using the cycle approach
    :param window_size: the frame number in the window
    :param window_queue: a list with windows,
    """
    cv.namedWindow(window_name)
    cap = cv.VideoCapture(camera_id)

    num = 0
    startSystem = False
    redundantFrames = window_size + window_queue - 1

    # Initialize some redundant frames for the normal run of the system. For example, for a classifier with window size
    # of 32 and window queue of 5, the system should have 36 redundant frames.
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        image_name = '%s%d.jpg' % (save_dir, num % cycle_limit)
        cv.imwrite(image_name, frame)
        cv.imshow(window_name, frame)

        num += 1
        if num == redundantFrames:
            print('Start the system ...')
            startSystem = True
            break

    # real-time capture frames and run the system.
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        image_name = '%s%d.jpg' % (save_dir, num % cycle_limit)
        cv.imwrite(image_name, frame)

        cv.imshow(window_name, frame)
        c = cv.waitKey(1)
        if c & 0xFF == ord('q'):
            break

        if startSystem and (num % 4 == 0):
            if 0 <= num < redundantFrames:
                index = num + cycle_limit - redundantFrames
            else:
                index = num - redundantFrames
            run_system(index)

        num = (num + 1) % cycle_limit

    cap.release()
    cv.destroyAllWindows()
    print('Finished.')


def run_system(index):
    # run the detector
    detection_start = timeit.default_timer()
    activateClassifier = run_detector(index)
    detection_stop = timeit.default_timer()

    classification_start = 0
    classification_stop = 0
    if activateClassifier:
        print("Activate Classifier: " + str(activateClassifier))
        print("index: " + str(index))
        classification_start = timeit.default_timer()
        run_classifier(index)
        classification_stop = timeit.default_timer()

    det_dura = detection_stop - detection_start
    cla_dura = classification_stop-classification_start
    # print("Duration: detection {:.2f} s | classification {:.2f} s".format(det_dura, cla_dura))


def run_detector(index, d_stride=1, queue_size=4, max_num=200):
    existGesture = False

    window_queue = [(index + i * d_stride) % max_num for i in range(queue_size)]
    dataloader = online_video_processing(batch_size=queue_size, window_queue=window_queue)

    with torch.no_grad():
        for idx, inputs in enumerate(dataloader):
            inputs = inputs.to(device)
            outputs = detector(inputs)

            probs = nn.Softmax(dim=1)(outputs)
            preds = torch.max(probs, 1)[1]
            preds = preds.to("cpu").numpy()

            # if all of the predictions are 'True' in the queue, the system claim that there is a gesture
            if preds.all():
                existGesture = True

    return existGesture


def run_classifier(index, stride=1, queue_size=1, max_num=200):
    window_queue = [(index + i * stride) % max_num for i in range(queue_size)]
    dataloader = online_video_processing(batch_size=queue_size, window_queue=window_queue, frame_nb=clr_frame_nb)

    # when you change a diffrent window, you should change the corresponding 'queue_size'!
    # one window
    with torch.no_grad():
        for idx, inputs in enumerate(dataloader):
            inputs = inputs.to(device)
            outputs = classifier(inputs)

            probs = nn.Softmax(dim=1)(outputs)
            pred = torch.max(probs, 1)[1]

            pred = pred.to("cpu").numpy()
            pred_name = [category_index_classifier[i] for i in pred]
            print(pred_name)


if __name__ == "__main__":
    camera('myCamera', 0, './video_expr/Online_Buffer/', 200, clr_frame_nb, 4)

