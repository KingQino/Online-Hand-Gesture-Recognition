# -*- coding: utf-8 -*-
# @Time    : 2020/7/23 10:35 AM
# @Author  : Yinghao Qin
# @Email   : y.qin@hss18.qmul.ac.uk
# @File    : video_stream.py
# @Software: PyCharm
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path


class Offline_Video_Process(Dataset):
    """
    The class is used to process offline video, i.e. prerecorded video.
    """
    def __init__(self, video_dir="./video_expr/Offline_Buffer/", frame_file_ending="jpg", number_of_frames=8,
                 stream_length=180, video_transform=None):
        """

        :param video_dir: Path to the directory containing the video frames.
        :param frame_file_ending: File ending of the video images, eg. "jpg" for file names like "1.jpg"
        :param number_of_frames: The number of frames to get from one video.
        :param stream_length: The total number of the frames in the video.
        :param video_transform: The transform approach to video frames.
        """
        self.file_ending = frame_file_ending
        self.video_dir = video_dir
        self.number_of_frames = number_of_frames
        self.stream_length = stream_length
        self.video_transform = video_transform

    def __len__(self):
        """
        For the video stream, we need to give it a clip redundancy.
        """
        return self.stream_length - self.number_of_frames

    def __getitem__(self, idx):
        video_directory = Path(self.video_dir)
        frame_files = []
        print("The predicted video clip -- [" + str(idx) + ", " + str(idx + self.number_of_frames) + ")")
        for _ in range(idx, idx+self.number_of_frames):
            frame_files += list(Path(video_directory).glob(f"{_}.{self.file_ending}"))
        if len(frame_files) == 0:
            raise FileNotFoundError(
                f"Could not find any frames. There should be at least one frame in the directory "
                f"{video_directory}")
        frames = [Image.open(frame_file).convert('RGB') for frame_file in frame_files]
        if self.video_transform:
            frames = self.video_transform(frames)
        return frames
