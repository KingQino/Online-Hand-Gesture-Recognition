# -*- coding: utf-8 -*-
# @Time    : 2020/8/19 8:46 AM
# @Author  : Yinghao Qin
# @Email   : y.qin@hss18.qmul.ac.uk
# @File    : utils2.py
# @Software: PyCharm
from torch.utils.data import DataLoader
from torchvideotransforms.video_transforms import Resize, Normalize, Compose
from torchvideotransforms.volume_transforms import ClipToTensor

from jesterdataset import JesterDataset


#######################################################################################
# 'utils2' is used to process data to produce data loader                             #
#######################################################################################


def data_processing(batch_size=4, channel_nb=3, frame_nb=16, frame_size=(112, 112),
                    csv_file='./annotation_jester/annotation_train.csv', video_dir='./jester_data/20bn-jester-v1',
                    frame_select_strategy=JesterDataset.FrameSelectStrategy.RANDOM,
                    frame_padding=JesterDataset.FramePadding.REPEAT_END, shuffle=True, num_workers=4):
    """
    Process the video data. 1. transform data and convert them to tensor; 2. read the csv file, such as 'train', 'label'
    :param batch_size: the size of batch
    :param channel_nb: the number of channels
    :param frame_nb: the number of frames
    :param frame_size: the size of frame in video, e.g., (100, 120)
    :param csv_file: read the csv file, such as train, validation, test, and labels
    :param video_dir: the directory of the video
    :param frame_select_strategy: FROM_BEGINNING = 0, FROM_END = 1, RANDOM = 2
    :param frame_padding: REPEAT_END = 0, REPEAT_BEGINNING = 2
    :param shuffle: whether to shuffle the data
    :param num_workers: the number of workers
    :return: data loader, the batch in it is (batch_size, channel_nb, frame_nb, height, width), e.g., (4,3,16,100,120)
    """
    video_transform_list = [
        Resize(frame_size),
        ClipToTensor(channel_nb=channel_nb),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    video_transform = Compose(video_transform_list)
    dataset = JesterDataset(csv_file=csv_file, video_dir=video_dir, number_of_frames=frame_nb,
                            video_transform=video_transform,
                            frame_select_strategy=frame_select_strategy,
                            frame_padding=frame_padding)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader
