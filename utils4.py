# -*- coding: utf-8 -*-
# @Time    : 2020/8/3 9:58 PM
# @Author  : Yinghao Qin
# @Email   : y.qin@hss18.qmul.ac.uk
# @File    : utils4.py
# @Software: PyCharm
from torch.utils.data import DataLoader
from torchvideotransforms.video_transforms import Resize, Normalize, Compose
from torchvideotransforms.volume_transforms import ClipToTensor

from jesterdataset import Online_Video_Process


#######################################################################################
# 'utils4' is used to process online data                                             #
#######################################################################################


def online_video_processing(batch_size=5, channel_nb=3, frame_nb=8, frame_size=(112, 112), window_queue=[],
                            buffer_dir='./video_expr/Online_Buffer', cycle_para=200, shuffle=False, num_workers=4):
    """
    It is used to process the online video stream.
    :param batch_size: the number of frames for the whole video.
    :param channel_nb: the number of channels
    :param frame_nb: for each batch, the number of frames
    :param frame_size: the size of frame in video, e.g., (120, 120)
    :param window_queue: a list, used to store the window index
    :param buffer_dir: it is a buffer directory storing all of the video frames
    :param cycle_para: it is a cycle parameter to help store frames using the cycle approach
    :param shuffle: whether to shuffle the data
    :param num_workers: the number of workers
    :return: data loader, the batch in it is (batch_size, channel_nb, frame_nb, height, width), e.g., (4,3,8,112,112)
    """
    video_transform_list = [
        Resize(frame_size),
        ClipToTensor(channel_nb=channel_nb),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    video_transform = Compose(video_transform_list)
    dataset = Online_Video_Process(window_queue=window_queue, video_dir=buffer_dir, number_of_frames=frame_nb,
                                   cycle_para=cycle_para, video_transform=video_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return dataloader

