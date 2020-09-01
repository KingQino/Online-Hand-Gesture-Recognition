# -*- coding: utf-8 -*-
# @Time    : 2020/7/20 12:11 AM
# @Author  : Yinghao Qin
# @Email   : y.qin@hss18.qmul.ac.uk
# @File    : train.py
# @Software: PyCharm
#######################################################################################
# The parameters that need to be changed are listed below when switching models       #
# Video_dir: The path to the database                                                 #
# Resnet-10:                                                                          #
#   nb_classes = 2                                                                    #
#   learning_rate = 1e-3                                                              #
#   Frame_nb = 8                                                                      #
#   Annotation_train = './annotation_jester/annotation_binary_gesture_train.csv'      #
#   Annotation_val = './annotation_jester/annotation_binary_gesture_val.csv'          #
#   model_name = 'resnet10'                                                           #
#   weight_decay=5e-4                                                                 #
#   step_size=10                                                                      #
# C3D: (16-frame, 32-frame)                                                           #
#   nb_classes = 27                                                                   #
#   learning_rate = 1e-3                                                              #
#   Frame_nb = 16 or 32                                                               #
#   Annotation_train = './annotation_jester/annotation_train.csv'                     #
#   Annotation_val = './annotation_jester/annotation_val.csv'                         #
#   model_name = 'C3D'                                                                #
#   weight_decay=5e-4                                                                 #
#   step_size=10                                                                      #
# ResNeXt: (16-frame, 32-frame)                                                       #
#   nb_classes = 27                                                                   #
#   learning_rate = 1e-2                                                              #
#   Frame_nb = 16 or 32                                                               #
#   Annotation_train = './annotation_jester/annotation_train.csv'                     #
#   Annotation_val = './annotation_jester/annotation_val.csv'                         #
#   model_name = 'resnext101'                                                         #
#   weight_decay=1e-3                                                                 #
#   step_size=15                                                                      #
#######################################################################################
import glob
import os
import timeit
import socket
from datetime import datetime

import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from tqdm import tqdm

from Networks.c3d import C3D
from Networks.resnet import resnet10
from Networks.resnext import resnext101
from utils2 import data_processing

# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

nEpochs = 100  # Number of epochs for training
resume_Epoch = 0  # Default is 0, change if want to resume
useTest = True  # See evolution of the test set when training
nTestInterval = 20  # Run on test set every nTestInterval epochs
snapshot = 50  # Store a model every snapshot epochs
nb_classes = 27  # Number of classes
learning_rate = 1e-3  # Learning rate
Frame_nb = 16  # The number of frames selected in a video
Frame_size = 112  # The size of the frame
Batch_size = 20  # The size of batch
Annotation_train = './annotation_jester/annotation_train.csv'  # annotation for train data
Annotation_val = './annotation_jester/annotation_val.csv'  # annotation for test data
Video_dir = '/import/scratch-01/yq300/20bn-jester-v1'  # database

save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]
if resume_Epoch != 0:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) if runs else 0
else:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0
save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))

# Select model and dataset
model_name = 'C3D'  # Options: C3D or ResNeXt-101
dataset_name = 'Jester'  # Options: Jester or EgoGesture


def train_model(modelName=model_name, dataset=dataset_name, save_dir=save_dir, num_epochs=nEpochs,
                resume_epoch=resume_Epoch, useTest=useTest, test_interval=nTestInterval, save_epoch=snapshot,
                num_classes=nb_classes, lr=learning_rate, frame_nb=Frame_nb, frame_size=Frame_size,
                batch_size=Batch_size, annotation_train=Annotation_train, annotation_val=Annotation_val,
                video_dir=Video_dir):
    """
    Training the model.
    :param modelName: the name of the model selected
    :param dataset: the dataset selected
    :param save_dir: the save directory
    :param num_epochs: the number of training epochs
    :param resume_epoch: the resume epoch when resume training from a model
    :param useTest: whether we should use Test
    :param test_interval: run on test set every nTestInterval epochs
    :param save_epoch: when it reaching every save_epoch, the model state will be stored
    :param num_classes: the number of classes in the dataset
    :param lr: learning rate
    :param frame_nb: the number of frames selected in a video
    :param frame_size: the size of the frame
    :param batch_size: the size of the batch
    :param annotation_train: the csv file containing the annotation of the train data
    :param annotation_val: the csv file containing the annotation of the validation data
    :param video_dir: path to the video
    :return:
    """
    if modelName == 'C3D':
        model = C3D(num_classes=num_classes, sample_duration=frame_nb)
    elif modelName == 'resnet10':
        model = resnet10(num_classes=num_classes, sample_size=frame_size, sample_duration=frame_nb)
    elif modelName == 'resnext101':
        model = resnext101(num_classes=num_classes, sample_size=frame_size, sample_duration=frame_nb)
    else:
        print("Wrong model is selected!")
        exit(0)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    # the scheduler divides the lr by 10 every 10 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    saveName = modelName + '-' + dataset
    if resume_epoch == 0:
        print("Training {} from scratch...".format(modelName))
    else:
        checkpoint = torch.load(
            os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar'),
            map_location=lambda storage, loc: storage)  # Load all tensors onto the CPU
        print("Initializing weights from: {}...".format(
            os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar')))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.to(device)
    criterion.to(device)

    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)

    print('Training model on {} dataset...'.format(dataset))
    train_dataloader = data_processing(batch_size=batch_size, frame_nb=frame_nb,
                                       csv_file=annotation_train, video_dir=video_dir)
    validation_dataloader = data_processing(batch_size=batch_size, frame_nb=frame_nb,
                                            csv_file=annotation_val, video_dir=video_dir)
    train_size = len(train_dataloader.dataset)
    val_size = len(validation_dataloader.dataset)

    for epoch in range(resume_epoch, num_epochs):
        # each epoch has a training and validation step
        start_time = timeit.default_timer()

        # train phrase
        # reset the running loss and corrects
        running_loss = 0.0
        running_corrects = 0.0

        scheduler.step()
        model.train()
        for inputs, labels in tqdm(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            probs = nn.Softmax(dim=1)(outputs)
            preds = torch.max(probs, 1)[1]

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / train_size
        epoch_acc = running_corrects.double() / train_size

        writer.add_scalar('Train/Loss', epoch_loss, epoch + 1)
        writer.add_scalar('Train/Accuracy', epoch_acc, epoch + 1)

        print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format('train', epoch + 1, num_epochs, epoch_loss, epoch_acc))

        # validation phrase
        # reset the running loss and corrects
        running_loss = 0.0
        running_corrects = 0.0

        model.eval()
        for inputs, labels in tqdm(validation_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with torch.no_grad():
                outputs = model(inputs)
            probs = nn.Softmax(dim=1)(outputs)
            preds = torch.max(probs, 1)[1]
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / val_size
        epoch_acc = running_corrects.double() / val_size

        writer.add_scalar('Val/Loss', epoch_loss, epoch + 1)
        writer.add_scalar('Val/Accuracy', epoch_acc, epoch + 1)

        print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format('val', epoch + 1, nEpochs, epoch_loss, epoch_acc))
        stop_time = timeit.default_timer()
        print("Execution time: " + str(stop_time - start_time) + "\n")

        # Save the model when epoch is divided by save_epoch with no remainder
        if epoch % save_epoch == (save_epoch - 1):
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'opt_dict': optimizer.state_dict(),
            }, os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar'))
            print("Save model at {}\n".format(
                os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar')))

    writer.close()


if __name__ == "__main__":
    train_model()
