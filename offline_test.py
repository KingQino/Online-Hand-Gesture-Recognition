# -*- coding: utf-8 -*-
# @Time    : 2020/8/2 11:49 PM
# @Author  : Yinghao Qin
# @Email   : y.qin@hss18.qmul.ac.uk
# @File    : offline_test.py
# @Software: PyCharm
import json
import math
import os
import pdb
from pathlib import Path

import torch
from torch import nn

from Networks.c3d import C3D
from Networks.resnet import resnet10
from Networks.resnext import resnext101
from utils3 import offline_video_processing, proposals_processing, proposal_data_collection
from utils4 import online_video_processing
from utils6 import LevenshteinDistance

#######################################################################################
# For running the file, you should ensure that there are video frames in the directory#
# - "video_expr/Offline_Buffer".                                                      #
# For example,                                                                        #
#     1. parse the experiment video 'VideoExp-1.avi' into frames placed in            #
#     'Offline_Buffer' by running 'video.py'                                          #
#     2. run this file                                                                #
#######################################################################################
#######################################################################################
# The parameters that you are possible to change are listed below                     #
#   path_detector | the path to the detector "resnet10-8"                             #
#   path_classifier | the path to the classifier                                      #
#   clr_frame_nb = 16 or 32  | when you use different classifier                      #
#   amendment_factor  | when you use 32-frame classifier, it is set as -6; when you   #
#                       when you use 16-frame classifier, it is set as 4              #
#   item (the bottom) | it is the name of video, which is used to get the ground truth#
#######################################################################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)


# load detector and classifier
path_detector = './run/resnet10-8-v2/resnet10-Jester_epoch-99.pth.tar'
path_detector_label_dic = './annotation_jester/label_dic_binary_gesture.json'
detector = resnet10(num_classes=2, sample_size=112, sample_duration=8)
detector.to(device)
checkpoint = torch.load(path_detector, map_location=lambda storage, loc: storage)
detector.load_state_dict(checkpoint['state_dict'])

path_classifier = './run/resnext101-16/resnext101-Jester_epoch-99.pth.tar'
path_classifier_label_dic = './annotation_jester/label_dic.json'
clr_frame_nb = 16
# classifier = resnext101(num_classes=27, sample_size=112, sample_duration=clr_frame_nb)
classifier = C3D(num_classes=27, sample_duration=clr_frame_nb)
classifier.to(device)
checkpoint = torch.load(path_classifier, map_location=lambda storage, loc: storage)
classifier.load_state_dict(checkpoint['state_dict'])


# load data
buffer_dir = "./video_expr/Offline_Buffer/"
frame_files = list(Path(Path(buffer_dir)).glob(f"*.{'jpg'}"))
stream_len = len(frame_files)
dataloader = offline_video_processing(buffer_dir=buffer_dir, stream_length=stream_len, num_workers=0)

# parameters
intersection_threshold = 4
amendment_factor = 4
# amendment_factor = -6
label_dir = "./video_expr/Exp_label.json"


print("####################################################################")
print("############################ Detection #############################")
print("####################################################################")

# read class_indict
category_index_detector = {}
try:
    json_file = open(path_detector_label_dic, 'r')
    class_dict = json.load(json_file)
    category_index_detector = {v: k for k, v in class_dict.items()}
except Exception as e:
    print(e)
    exit(-1)

# detection
proposals = []
detector.eval()
with torch.no_grad():
    for index, inputs in enumerate(dataloader):
        inputs = inputs.to(device)
        outputs = detector(inputs)

        probs = nn.Softmax(dim=1)(outputs)
        preds = torch.max(probs, 1)[1]

        preds = preds.to("cpu").numpy()
        preds_name = [category_index_detector[i] for i in preds]

        print(preds_name)

        if preds_name == ['Gesture', 'Gesture', 'Gesture', 'Gesture']:
            proposals += [(index * 4, index * 4 + 8)]

print("The original detection results: " + str(proposals))

# do some processing and amendments to original detection results
proposals = proposals_processing(proposals, intersection_threshold, amendment_factor, border=stream_len,
                                 clr_frame_nb=clr_frame_nb)
print("The 'Gesture' proposal area: " + str(proposals))

print("####################################################################")
print("########################## Classification ##########################")
print("####################################################################")

# collect frames in the proposal areas, and storage them into a directory
# dir_proposals = proposal_data_collection(buffer_dir, proposals, clr_frame_nb)

# read class_indict
class_int_name = {}
class_name_int = {}
try:
    json_file = open(path_classifier_label_dic, 'r')
    class_name_int = json.load(json_file)
    class_int_name = {v: k for k, v in class_name_int.items()}
except Exception as e:
    print(e)
    exit(-1)

# classification
# prediction = []
# for i in range(len(proposals)):
#     path_proposal = os.path.join(dir_proposals, str(proposals[i][0]))
#     classifier_dataloader = offline_video_processing(batch_size=1, buffer_dir=path_proposal, frame_nb=clr_frame_nb,
#                                                      stream_length=1+clr_frame_nb, num_workers=0)
#
#     classifier.eval()
#     with torch.no_grad():
#         for input in classifier_dataloader:
#             input = input.to(device)
#             output = classifier(input)
#
#             prob = nn.Softmax(dim=1)(output)
#             pred = torch.max(prob, 1)[1]
#
#             pred = pred.to("cpu").numpy()
#             prediction.append(pred[0])
#             pred_name = [class_int_name[i] for i in pred]
#             print(pred_name)


classifier.eval()
prediction1 = []
queue_size = 1
stride = 1
print("***************************** 1 window *****************************")
for k in range(len(proposals)):
    index = proposals[k][0]
    window_queue = [index + i * stride for i in range(queue_size)]
    dataloader = online_video_processing(batch_size=queue_size, window_queue=window_queue, frame_nb=clr_frame_nb,
                                         buffer_dir="./video_expr/Offline_Buffer/", cycle_para=240)

    # one window
    with torch.no_grad():
        for idx, inputs in enumerate(dataloader):
            inputs = inputs.to(device)
            outputs = classifier(inputs)

            probs = nn.Softmax(dim=1)(outputs)
            pred = torch.max(probs, 1)[1]

            pred = pred.to("cpu").numpy()
            prediction1.append(pred[0])
            pred_name = [class_int_name[i] for i in pred]
            print(pred_name)

print("***************************** 3 windows ****************************")
queue_size = 3
prediction2 = []
for k in range(len(proposals)):
    index = proposals[k][0]
    window_queue = [index + i * stride for i in range(queue_size)]
    dataloader = online_video_processing(batch_size=queue_size, window_queue=window_queue, frame_nb=clr_frame_nb,
                                         buffer_dir="./video_expr/Offline_Buffer/", cycle_para=240)

    # three windows
    weights = [0.3, 0.4, 0.3]
    with torch.no_grad():
        for idx, inputs in enumerate(dataloader):
            inputs = inputs.to(device)
            outputs = classifier(inputs)

            probs = nn.Softmax(dim=1)(outputs)
            weightprobs = [0 for i in range(27)]
            for i in range(queue_size):
                weightprobs += probs[i].numpy() * weights[i]

            weightprobs_dic = {weightprobs[i]: i for i in range(27)}
            weightprobs_sort = sorted(weightprobs, reverse=True)
            max1 = weightprobs_sort[0]
            max2 = weightprobs_sort[1]

            pred = 25  # No gesture
            if max1 - max2 >= 0.6:
                pred = weightprobs_dic[max1]
                print("Early Detection")
            elif max1 >= 0.2:
                pred = weightprobs_dic[max1]
                print("Late Detection")

            prediction2.append(pred)
            pred_name = class_int_name[pred]
            print(pred_name)

print("***************************** 5 windows ****************************")
queue_size = 5
prediction3 = []
for k in range(len(proposals)):
    index = proposals[k][0]
    window_queue = [index + i * stride for i in range(queue_size)]
    dataloader = online_video_processing(batch_size=queue_size, window_queue=window_queue, frame_nb=clr_frame_nb,
                                         buffer_dir="./video_expr/Offline_Buffer/", cycle_para=240)

    # five windows
    weights = [0.11 * math.cos((math.pi/4) * i - (math.pi/2)) + 0.15 for i in range(5)]
    with torch.no_grad():
        for idx, inputs in enumerate(dataloader):
            inputs = inputs.to(device)
            outputs = classifier(inputs)

            probs = nn.Softmax(dim=1)(outputs)
            weightprobs = [0 for i in range(27)]
            for i in range(queue_size):
                weightprobs += probs[i].numpy() * weights[i]

            weightprobs_dic = {weightprobs[i]: i for i in range(27)}
            weightprobs_sort = sorted(weightprobs, reverse=True)
            max1 = weightprobs_sort[0]
            max2 = weightprobs_sort[1]

            pred = 25  # No gesture
            if max1 - max2 >= 0.6:
                pred = weightprobs_dic[max1]
                print("Early Detection")
            elif max1 >= 0.2:
                pred = weightprobs_dic[max1]
                print("Late Detection")

            prediction3.append(pred)
            pred_name = class_int_name[pred]
            print(pred_name)


print("************************** Levenshtein accuracy ***************************")
print("The prediction (1 window)  is " + str(prediction1))
print("The prediction (3 windows) is " + str(prediction2))
print("The prediction (5 windows) is " + str(prediction3))
ground_truth = []
label = {}
try:
    json_file = open(label_dir, 'r')
    label = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

item = 'VideoExp-20'
ground_truth = [class_name_int[label[item][i]] for i in range(len(label[item]))]
print("The Ground truth sequence is  " + str(ground_truth))
distance = LevenshteinDistance(ground_truth, prediction1)
LevenshteinAccuracy = 1 - distance/len(ground_truth)
print("The Levenshtein 1 accuracy is " + str(LevenshteinAccuracy))

distance = LevenshteinDistance(ground_truth, prediction2)
LevenshteinAccuracy = 1 - distance/len(ground_truth)
print("The Levenshtein 3 accuracy is " + str(LevenshteinAccuracy))

distance = LevenshteinDistance(ground_truth, prediction3)
LevenshteinAccuracy = 1 - distance/len(ground_truth)
print("The Levenshtein 5 accuracy is " + str(LevenshteinAccuracy))

