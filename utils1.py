# -*- coding: utf-8 -*-
# @Time    : 2020/7/19 10:58 PM
# @Author  : Yinghao Qin
# @Email   : y.qin@hss18.qmul.ac.uk
# @File    : utils1.py
# @Software: PyCharm
import json

import pandas as pd


#######################################################################################
# 'utils1' has three functions - get_labels, convert_labels, split_data_gesture_exist #
# 1. Read labels from the original data and convert them to the corresponding number  #
# 2. Get a subset of Jester database for training detector, 'Gesture' and 'No gesture'#
#######################################################################################

def get_labels(path_labels, save_dir="./annotation_jester/label_dic.json"):
    """
    Loads the data frame labels from a csv and creates dictionary to convert the string labels to int.
    :param path_labels: path to the csv containing the labels
    :param save_dir: the save directory of the label dictionary
    :return: the label dict.
    """
    labels_df = pd.read_csv(path_labels, names=['label'])
    # Extracting list of labels from the dataframe
    labels = [str(label[0]) for label in labels_df.values]
    n_labels = len(labels)
    # Create dictionaries to convert label to int and backwards
    label_to_int = dict(zip(labels, range(n_labels)))

    with open(save_dir, 'w') as file_obj:
        json.dump(label_to_int, file_obj)

    return label_to_int


def convert_labels(path_csv, path_label_dic, save_dir):
    """
    Convert the 'name' label to 'ID' label. (video_id, label_name) --> (video_id, label_id)
    :param path_csv: path to the csv containing the videos
    :param path_label_dic: path to the label dict
    :param save_dir: the directory of storing the return file
    """
    with open(path_label_dic, 'r') as load_f:
        label_dict = json.load(load_f)

    df = pd.read_csv(path_csv, sep=';', names=['video_id', 'label'])
    list_video_id = df['video_id'].values
    list_label_id = [label_dict[label] for label in df['label'].values]
    annotation = dict(zip(list_video_id, list_label_id))

    with open(save_dir, 'w') as f:
        [f.write('{0},{1}\n'.format(key, value)) for key, value in annotation.items()]


def split_data_gesture_exist(path_csv, save_dir_label, save_dir_annotation):
    """
    The function is used to split the data into 'Gesture' and 'No gesture' two classes. First, select all of the data in
     the 'No gesture' class and then randomly pick the same number of data from the other classes.
    :param path_csv: path to the csv containing the videos
    :param save_dir_label: The save directory of the label dict file.
    :param save_dir_annotation: The save directory of the generated annotation file.
    :return:
    """
    df = pd.read_csv(path_csv, sep=';', names=['video_id', 'label'])
    count_no_gesture = sum(df['label'].values[df.index] == "No gesture")
    print("The total data number of 'No gesture' class: %d " % count_no_gesture)
    list_video_id = []
    list_label_id = []
    for i in df.index:
        if df['label'].values[i] == 'No gesture':
            list_video_id.append(df['video_id'].values[i])
            list_label_id.append(0)
            # list_label_id.append(label_dict[df['label'].values[i]])
    for i in df.index:
        if count_no_gesture == 0:
            break
        if df['label'].values[i] != 'No gesture':
            list_video_id.append(df['video_id'].values[i])
            list_label_id.append(1)
            # list_label_id.append(label_dict[df['label'].values[i]])
            count_no_gesture -= 1
    annotation = dict(zip(list_video_id, list_label_id))

    label_dic_gesture = {'No gesture': 0, 'Gesture': 1}
    with open(save_dir_label, 'w') as file_obj:
        json.dump(label_dic_gesture, file_obj)

    with open(save_dir_annotation, 'w') as f:
        [f.write('{0},{1}\n'.format(key, value)) for key, value in annotation.items()]


# get_labels(path_labels='../jester_data/jester-v1-labels.csv',
#            save_dir='./annotation_jester/label_dic.json')
# convert_labels(path_csv='/dataset/jester-v1-train.csv',
#                path_label_dic='./annotation_jester/label_dic.json',
#                save_dir='./annotation_jester/annotation_train.csv')
# convert_labels(path_csv='/dataset/jester-v1-validation.csv',
#                path_label_dic='./annotation_jester/label_dic.json',
#                save_dir='./annotation_jester/annotation_val.csv')
# split_data_gesture_exist(path_csv='/dataset/jester-v1-train.csv',
#                          save_dir_label='./annotation_jester/label_dic_binary_gesture.json',
#                          save_dir_annotation='./annotation_jester/annotation_binary_gesture_train.csv')
# split_data_gesture_exist(path_csv='/dataset/jester-v1-validation.csv',
#                          save_dir_label='./annotation_jester/label_dic_binary_gesture.json',
#                          save_dir_annotation='./annotation_jester/annotation_binary_gesture_val.csv')
