# Online Hand Gesture Recognition Using 3D Convolutional Neural Networks
## MSc Final Dissertation

- [Dissertation](./Dissertation.pdf)  (&rarr; June 2020)

- This work is supervised by [Dr. Tijana Timotijevic](https://www.qmul.ac.uk/eecs/people/profiles/timotijevictijana.html) :heart:

  > - Although there have been a lot of deep learning researches about dynamic gesture recognition, most of them just focus on how to learn more representative spatiotemporal features in pre-segmented video clips. However, few researchers put their sights on the real-world application of hand gesture recognition. This paper is an attempt to bridge the gap between them.
  > - In this paper, an online hand gesture recognition system is proposed, which is able to localize gestures in real-time video stream and recognize what these gestures are. All of the models in my project are trained on Jester database. For the overall performance of the system, the best group can respond within three seconds and reach 37.5% Levenshtein accuracy on the homemade dataset.
  > - Finally, the project achieved a mark of 80%, i.e. distinction.

## Demo
![](https://github.com/KingQino/Online-Hand-Gesture-Recognition/blob/master/demo.gif)

## Notice!
For running the system, you need to install some packages first. There are some key packages are listed below, the other 
ones you can install according to the system prompt. If you want to train your models on Jester database on your own, 
you need to download the entire database from the official website and ensure the path to the database is right. 
Alternatively, you can also download the pre-trained models from the links below. In particular, there is a mini version 
of pre-trained models just with a detector and a classifier, which is the group with the best performance in my 
experiment. Besides, the model should be trained on the GPU server, but you can run the program on CPU processor (In 
fact, I debug this program on Mac). Finally, all parameters that you may change are listed in the corresponding file. 

## Requirements:
* Python 3
* Pytorch 1.5
* torch_videovision 
```pip install git+https://github.com/hassony2/torch_videovision```
* opencv-python 4.2


## Directory:
```
.
├── Networks: The directory of models.
├── jesterdataset: A pytorch data loader to load the 20BN-JESTER dataset.
├── annotation_jester: The annotaion files of Jester dataset.
├── run: The directory is used to store the running results of model training.
├── log: The log directory stores the real-time running records on model training.
├── results: The experimental data after simple processing.
├── README.md
├── offline_test.py: The offline running file.
├── online_test.py: The online running file. 
├── predict.py: The predict file used to predict single video data.
├── train.py: The training file used to train models on Jester database.
├── utils1.py: Process the labels and annnotations of Jester database.
├── utils2.py: Produce data loader by invoking files in 'jesterdataset'. 
├── utils3.py: Process offline data and give processing about proposals.
├── utils4.py: Process online data.
├── utils5.py: Plot accuracy and loss figures.
├── utils6.py: Calculate Levenshtein distance. 
├── video.py: Capture the video and parse video into frames.
└── video_expr: The directory is used for some video experiments and evaluation on the entire system.
    ├── Offline_Buffer: It is used to store the offline running files.
    ├── Online_Buffer: It is used to store the online running files.
    ├── Exp_label.json: The label file corresponding to 'VideoExp-*.avi'
    ├── VideoExp-1.avi
    ├── ...
    └── VideoExp-20.avi

Jester database
.
├── 20bn-jester-v1
├── jester-v1-labels.csv
├── jester-v1-train.csv
├── jester-v1-validation.csv
└── jester-v1-test.csv

```

## Pretrained Model:
* [Mini-version(554MB)](https://drive.google.com/file/d/1pSAFpAtd4W1cPleYNFkx10mpcWbiHqf8/view?usp=sharing): 
The mini version consists of resnet10-8 and resnext101-16, which are used to run the system with the best performance.
Backup Download (link:https://pan.baidu.com/s/1laQjMqSINk9cKABgtSN-eA  code:b3z9).
* [All-models(2.48GB)](https://drive.google.com/file/d/1mV4BURsSMJLpUotV9yWUGWHPdf6EKVcY/view?usp=sharing): 
It contains all of the pre-trained models. 
Backup Download (link:https://pan.baidu.com/s/1oTJ2HAN56eP8mAO8KLq1PA  code:mc58).
```
You can download pre-trained models from Google Drive or Baidu Netdisk.
The downloaded file is proposed to place in the directory 'run'.
```


 
## Dataset
### Jester
* Download Jester database by following [the official site](https://20bn.com/datasets/jester).
### Homemade dataset
* This dataset is used for evaluating the entire system.
* Place these data into the directory 'video_expr'.
* [Download link(580MB)](https://drive.google.com/file/d/1tXxBegbT4co12bKSvo0mV5IJ6TZbM-b7/view?usp=sharing)
* Backup Download link (link:https://pan.baidu.com/s/1sWol30_9PKZY6ot8q68-1g  code:9dal)

## Running the code
### Train
```
- If you wanna train your own models, you need to make sure you have the database downloaded.
- Then change the path of database in 'train.py'.
- Run 'train.py' on GPU.
- Alternatively, you can download the pre-trained models from links above. 
``` 
### Hand-Gesture-Recognition
* online test
```
The online system consists of the detector of ResNet10-8 and the classifier of ResNeXt101-16,
achieving the best performance on the whole.
- Make sure the paths of detecor and classifier are correct.
- Make sure the computer or laptop has a camera.
- Run 'online_test.py' on the command line interface (CML) or Terminal.
```
* offline test
```
- To run the offline version, please first parse the video into 'Offline_ Buffer' by running 'video.py' 
- Make sure parameters are correctly set.
- Run 'offline_test.py' on the command line interface (CML) or Terminal.
```
 
## Reference:
* https://github.com/FabianHertwig/pytorch_20BN-JESTER_Dataset
* https://github.com/jfzhang95/pytorch-video-recognition
* https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/Test5_resnet
* https://github.com/DavideA/c3d-pytorch
