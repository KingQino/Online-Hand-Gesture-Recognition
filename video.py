# -*- coding: utf-8 -*-
# @Time    : 2020/7/29 1:50 PM
# @Author  : Yinghao Qin
# @Email   : y.qin@hss18.qmul.ac.uk
# @File    : video.py
# @Software: PyCharm
import cv2 as cv

#######################################################################################
# You should run the file on Terminal (Command Line Window), like "python video.py"   #
#######################################################################################


def capture_video(window_name, camera_id=0, save_dir='./video_expr/MySaveVideo.avi', fps=12, duration=3):
    """
    It is used to capture the video.
    :param window_name: the name of the display window
    :param camera_id: id of the opened video capturing device. 0 is the local camera
    :param save_dir: the save directory of the video
    :param fps: the number of frames per second
    :param duration: the duration of the video, (seconds)
    :return:
    """
    cv.namedWindow(window_name)
    cap = cv.VideoCapture(camera_id)

    frame_size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
    videoWriter = cv.VideoWriter(save_dir, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, frame_size)

    numFrame = fps * duration

    while cap.isOpened() and numFrame > 0:
        ok, frame = cap.read()
        if not ok:
            break

        videoWriter.write(frame)
        numFrame -= 1

        cv.imshow(window_name, frame)
        c = cv.waitKey(1)
        if c & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
    print('Finished.')


def read_video(video_dir, window_name, save_dir):
    """
    It is used to read video.
    :param video_dir: the directory of the video.
    :param window_name: the window name
    :param save_dir: the save directory used to store the frames in the video
    :param fps: it is used to set the waitKey.
    :return: the number of frames in the video
    """
    cv.namedWindow(window_name)
    cap = cv.VideoCapture(video_dir)

    num = 0

    while cap.isOpened():
        ok, frame = cap.read()

        image_name = '%s%d.jpg' % (save_dir, num)
        cv.imwrite(image_name, frame)

        num += 1

        cv.imshow(window_name, frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

    return num+1


if __name__ == '__main__':
    # print('Open the camera and capture video')
    # capture_video(window_name='myCamera', save_dir='./video_expr/VideoExp-21.avi', duration=20)

    print('Parse video')
    _ = read_video("./video_expr/VideoExp-20.avi", "Display", "./video_expr/Offline_Buffer/")

