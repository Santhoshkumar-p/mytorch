from nn.flatten import *
from nn.Conv1d import *
from nn.linear import *
from nn.activation import *
from nn.loss import *
import numpy as np
import os
import sys

sys.path.append('mytorch')


class CNN_SimpleScanningMLP():
    def __init__(self):
        # self.conv1 = ???
        # self.conv2 = ???
        # self.conv3 = ???
        # ...
        # <---------------------
        self.conv1 = Conv1d(in_channels= 24, out_channels = 8, kernel_size = 8, stride=4)
        self.conv2 = Conv1d(in_channels= 8, out_channels = 16, kernel_size = 1, stride=1)
        self.conv3 = Conv1d(in_channels= 16, out_channels = 4, kernel_size = 1, stride=1)
        self.layers = [self.conv1, ReLU(), self.conv2, ReLU(), self.conv3, Flatten()] # TODO: Add the layers in the correct order

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN

        w1, w2, w3 = weights
        self.conv1.conv1d_stride1.W = np.transpose(w1.T.reshape(8, 8, 24),  (0, 2, 1))
        self.conv2.conv1d_stride1.W = np.transpose(w2.T.reshape(16, 1, 8),  (0, 2, 1))
        self.conv3.conv1d_stride1.W = np.transpose(w3.T.reshape(4, 1, 16),  (0, 2, 1))

    def forward(self, A):
        """
        Do not modify this method

        Argument:
            A (np.array): (batch size, in channel, in width)
        Return:
            Z (np.array): (batch size, out channel , out width)
        """

        Z = A
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Do not modify this method

        Argument:
            dLdZ (np.array): (batch size, out channel, out width)
        Return:
            dLdA (np.array): (batch size, in channel, in width)
        """

        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA


class CNN_DistributedScanningMLP():
    def __init__(self):
        self.conv1 = Conv1d(in_channels = 24, out_channels = 2, kernel_size = 2, stride = 2)
        self.conv2 = Conv1d(in_channels = 2, out_channels = 8, kernel_size = 2, stride = 2)
        self.conv3 = Conv1d(in_channels = 8, out_channels = 4, kernel_size = 2, stride = 1)
        self.layers = [self.conv1, ReLU(), self.conv2, ReLU(), self.conv3, Flatten()]  # TODO: Add the layers in the correct order

    def __call__(self, A):
        return self.forward(A)

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN

        w1, w2, w3 = weights
        w1 = w1[:48, :2]
        w2 = w2[:4,  :8]
        self.conv1.conv1d_stride1.W = np.transpose(w1.T.reshape(2, 2, 24), (0, 2, 1))
        self.conv2.conv1d_stride1.W = np.transpose(w2.T.reshape(8, 2, 2),  (0, 2, 1))
        self.conv3.conv1d_stride1.W = np.transpose(w3.T.reshape(4, 2, 8),  (0,2,1))

    def forward(self, A):
        """
        Do not modify this method

        Argument:
            A (np.array): (batch size, in channel, in width)
        Return:
            Z (np.array): (batch size, out channel , out width)
        """

        Z = A
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Do not modify this method

        Argument:
            dLdZ (np.array): (batch size, out channel, out width)
        Return:
            dLdA (np.array): (batch size, in channel, in width)
        """
        dLdA = dLdZ
        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA
