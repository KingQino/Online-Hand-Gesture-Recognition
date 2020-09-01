# -*- coding: utf-8 -*-
# @Time    : 2020/7/20 1:41 AM
# @Author  : Yinghao Qin
# @Email   : y.qin@hss18.qmul.ac.uk
# @File    : predict.py
# @Software: PyCharm
import json

import torch
from torch import nn

from Networks.c3d import C3D
from utils2 import data_processing

# Use GPU if available else revert to CPU
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

frame_nb = 32
path_model = './run/run_0/models/C3D-Jester_epoch-99.pth.tar'
path_label_dic = './annotation_jester/label_dic.json'

# create model
model = C3D(num_classes=27, sample_duration=frame_nb)

model.to(device)
checkpoint = torch.load(path_model, map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint['state_dict'])

# read class_indict
category_index = {}
try:
    json_file = open(path_label_dic, 'r')
    class_dict = json.load(json_file)
    category_index = {v: k for k, v in class_dict.items()}
except Exception as e:
    print(e)
    exit(-1)

test_dataloader = data_processing(batch_size=1, frame_nb=frame_nb, csv_file='./annotation_jester/annotation_train.csv',
                                  video_dir="jester_data/20bn-jester-v1")

model.eval()
with torch.no_grad():
    for inputs, labels in test_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)

        probs = nn.Softmax(dim=1)(outputs)
        preds = torch.max(probs, 1)[1]

        preds = preds.to("cpu").numpy()
        preds_name = [category_index[i] for i in preds]
        print(preds_name)

