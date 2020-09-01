# -*- coding: utf-8 -*-
# @Time    : 2020/8/3 10:41 AM
# @Author  : Yinghao Qin
# @Email   : y.qin@hss18.qmul.ac.uk
# @File    : online_video_stream.py
# @Software: PyCharm
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class Online_Video_Process(Dataset):
    def __init__(self, window_queue=[], video_dir="./video_expr/Online_Buffer", frame_file_ending="jpg",
                 number_of_frames=16, cycle_para=200, video_transform=None):
        """

        :param window_queue: a list, used to store the window index, such as [18,19,20,21].
        :param video_dir: a directory storing all of the video frames
        :param frame_file_ending: file ending of the video images, eg. "jpg" for file names like "1.jpg"
        :param number_of_frames: the number of frames to get from one video.
        :param cycle_para: it is a cycle parameter to help store frames using the cycle approach
        :param video_transform: the transform approach to video frames.
        """
        self.data_description = window_queue
        self.video_dir = video_dir
        self.file_ending = frame_file_ending
        self.number_of_frames = number_of_frames
        self.cycle_para = cycle_para
        self.video_transform = video_transform

    def __len__(self):
        return len(self.data_description)

    def __getitem__(self, index):
        window_id = self.data_description[index]
        video_directory = Path(self.video_dir)
        frame_files = []
        for frame in range(window_id, window_id + self.number_of_frames):
            frame_files += list(Path(video_directory).glob(f"{frame % self.cycle_para}.{self.file_ending}"))
        print("The input video clip -- [" + str(window_id) + ", " + str(window_id + self.number_of_frames) + ")")
        if len(frame_files) == 0:
            raise FileNotFoundError(
                f"Could not find any frames. There should be at least one frame in the directory "
                f"{video_directory}")
        frames = [Image.open(frame_file).convert('RGB') for frame_file in frame_files]
        if self.video_transform:
            frames = self.video_transform(frames)
        return frames


