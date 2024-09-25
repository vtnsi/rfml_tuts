# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 17:35:42 2024

Pytorch CNN model class for 1-dimensional convolutional neural network

@author: scadams21
"""

import torch.nn as nn


class CNN_RF(nn.Module):
    '''
    CNN model class for RF modulation classification

    The feature extraction layers use 1D convolution layers.
    The convolutional layers are followed by full connected layers.
    '''

    def __init__(self, num_classes):

        '''
        Initialize the CNN_RF class

        Args:
            num_classes: the number of modulation classess

        Returns:
            CNN_RF: an instance of the CNN_RF class
        '''
        super(CNN_RF, self).__init__()

        self.num_classes = num_classes

        # convolutional layers for feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv1d(2, 128, 8),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(128, 256, 8),
            nn.ReLU(),
            nn.MaxPool1d(4))

        # fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(3328, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_classes),
            nn.ReLU(),
        )

    def forward(self, x):
        '''
        forward function for the model

        Args:
            x: Pytorch tensor, dimensions [2, sequence length]
                the first element of the dimension represents the real and
                    imaginary components of the signal
                the secon element is the sequence length of the signal

        Returns:
            softmax_outputs: Tensor of softmax outputs of the model, dimensions
            [num_classes, ]
        '''
        x_conv = self.conv_layers(x).flatten(1)
        x_fc = self.fc(x_conv)
        return nn.functional.softmax(x_fc, dim=1)
