from nn.flatten import *
from nn.Conv1d import *
from nn.linear import *
from nn.activation import *
from nn.loss import *
import numpy as np
import os
import sys

sys.path.append('mytorch')


class CNN(object):

    """
    A simple convolutional neural network

    Here you build implement the same architecture described in Section 3.3
    You need to specify the detailed architecture in function "get_cnn_model" below
    The returned model architecture should be same as in Section 3.3 Figure 3
    """

    def __init__(self, input_width, num_input_channels, num_channels, kernel_sizes, strides,
                 num_linear_neurons, activations, conv_weight_init_fn, bias_init_fn,
                 linear_weight_init_fn, criterion, lr):
        """
        input_width           : int    : The width of the input to the first convolutional layer
        num_input_channels    : int    : Number of channels for the input layer
        num_channels          : [int]  : List containing number of (output) channels for each conv layer
        kernel_sizes          : [int]  : List containing kernel width for each conv layer
        strides               : [int]  : List containing stride size for each conv layer
        num_linear_neurons    : int    : Number of neurons in the linear layer
        activations           : [obj]  : List of objects corresponding to the activation fn for each conv layer
        conv_weight_init_fn   : fn     : Function to init each conv layers weights
        bias_init_fn          : fn     : Function to initialize each conv layers AND the linear layers bias to 0
        linear_weight_init_fn : fn     : Function to initialize the linear layers weights
        criterion             : obj    : Object to the criterion (SoftMaxCrossEntropy) to be used
        lr                    : float  : The learning rate for the class

        You can be sure that len(activations) == len(num_channels) == len(kernel_sizes) == len(strides)
        """

        self.train_mode = True
        self.nlayers = len(num_channels)

        self.activations = activations
        self.criterion = criterion

        self.lr = lr

        self.convolutional_layers = [
            Conv1d(num_input_channels, num_channels[0], kernel_sizes[0], strides[0]),
            Conv1d(num_channels[0],    num_channels[1], kernel_sizes[1], strides[1]),
            Conv1d(num_channels[1],    num_channels[2], kernel_sizes[2], strides[2])
        ]
        self.flatten = Flatten()
        w = input_width
        for i in range(3):
            w = (w - kernel_sizes[i]) // strides[i] + 1
        #w = ((w - kernel_sizes[i]) // strides[i] + 1 for i in range(3))[-1]
        self.linear_layer = Linear(num_channels[2] * w, num_linear_neurons)
        self.layers = [self.convolutional_layers[0], 
                       self.activations[0],
                       self.convolutional_layers[1],
                       self.activations[1],
                       self.convolutional_layers[2],
                       self.activations[2],
                       self.flatten,
                       self.linear_layer
                       ]
        
    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, num_input_channels, input_width)
        Return:
            Z (np.array): (batch_size, num_linear_neurons)
        """

        # Iterate through each layer
        # <---------------------

        # Save output (necessary for error and loss)
        self.Z = A
        for layer in self.layers:
            self.Z = layer.forward(self.Z)
        return self.Z


    def backward(self, labels):
        """
        Argument:
            labels (np.array): (batch_size, num_linear_neurons)
        Return:
            grad (np.array): (batch size, num_input_channels, input_width)
        """

        m, _ = labels.shape
        self.loss = self.criterion.forward(self.Z, labels).sum()
        grad = self.criterion.backward()

        # Iterate through each layer in reverse order
        # <---------------------
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    

    def zero_grads(self):
        for i in range(self.nlayers):
            self.convolutional_layers[i].conv1d_stride1.dLdW.fill(0.0)
            self.convolutional_layers[i].conv1d_stride1.dLdb.fill(0.0)

        self.linear_layer.dLdW.fill(0.0)
        self.linear_layer.dLdb.fill(0.0)

    def step(self):
        for i in range(self.nlayers):
            self.convolutional_layers[i].conv1d_stride1.W = (self.convolutional_layers[i].conv1d_stride1.W -
                                                             self.lr * self.convolutional_layers[i].conv1d_stride1.dLdW)
            self.convolutional_layers[i].conv1d_stride1.b = (self.convolutional_layers[i].conv1d_stride1.b -
                                                             self.lr * self.convolutional_layers[i].conv1d_stride1.dLdb)

        self.linear_layer.W = (
            self.linear_layer.W -
            self.lr *
            self.linear_layer.dLdW)
        self.linear_layer.b = (
            self.linear_layer.b -
            self.lr *
            self.linear_layer.dLdb)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False
