import numpy as np
from mytorch.nn.resampling import *
from mytorch.functional_1 import *
from mytorch.functional_2 import *

class Conv1D_stride1():
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 autograd_engine,
                 weight_init_fn=None,
                 bias_init_fn=None):
        # Do not modify this method
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size

        self.autograd_engine = autograd_engine

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0,
                                      (out_channel, in_channel, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, A):
        return self.forward(A)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channel, input_size)
        Return:
            Z (np.array): (batch_size, out_channel, output_size)
        """
        self.A = A

        batch_size, input_channel, input_size = A.shape

        output_size = input_size - self.kernel_size + 1

        Z = np.zeros((batch_size, self.out_channel, output_size))

        # TODO: Perform convolution (similar to HW2P1)
        for idx in range(output_size):

            window = A[:, :, idx:idx + self.kernel_size]

            Z[:, :, idx] = np.tensordot(window, self.W, axes=([1, 2], [1, 2])) + self.b

        # TODO: Add operation
        self.autograd_engine.add_operation(
            [A, self.W, self.b], 
            Z, 
            [None, self.dW, self.db], 
            conv1d_stride1_backward
        )

        # TODO: Return output
        return Z


class Conv1d():
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 downsampling_factor,
                 autograd_engine,
                 weight_init_fn=None,
                 bias_init_fn=None):
        # Do not modify the variable names
        self.downsampling_factor = downsampling_factor
        self.autograd_engine = autograd_engine

        # TODO: Initialize Conv1D() instance (remember to pass the autograd_engine)
        self.conv1d_stride1 = Conv1D_stride1(in_channel, out_channel, kernel_size, autograd_engine, weight_init_fn, bias_init_fn)

        # TODO: Initialize Downsample1d() instance (remember to pass the autograd_engine)
        self.downsample1d = Downsample1d(downsampling_factor, autograd_engine)

    def __call__(self, A):
        return self.forward(A)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channel, input_size)
        Return:
            Z (np.array): (batch_size, out_channel, output_size)
        """

        # TODO: Call Conv1D_stride1
        self.Z = self.conv1d_stride1.forward(A)

        # TODO: Downsample
        # NOTE: The add operation for this occurs in resampling.py
        #       so no need to do it here.
        self.Z = self.downsample1d.forward(self.Z) #TODO

        # TODO: Return output
        return self.Z

class Conv2D_stride1():
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 autograd_engine,
                 weight_init_fn=None,
                 bias_init_fn=None):

        # Do not modify this method

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size

        self.autograd_engine = autograd_engine

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channel, in_channel, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size,
                                    kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, A):
        return self.forward(A)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channel, output_width, output_height)
        """
        self.A = A

        # TODO: Perform convolution (similar to HW2P1)
        
        batch_size, input_channels, input_height, input_width = A.shape

        self.input_height, self.input_width = input_height, input_width

        output_height, output_width = input_height - self.kernel_size + 1, input_width - self.kernel_size + 1

        Z = np.zeros((batch_size, self.out_channel, output_height, output_width))

        for _i in range(output_height):

            for _j in range(output_width):

                convolve = A[:, :, _i:_i + self.kernel_size, _j:_j + self.kernel_size]

                Z[:, :, _i, _j] = np.tensordot(convolve, self.W, axes = ([1, 2, 3], [1, 2, 3])) + self.b
        # TODO: Add operation
        self.autograd_engine.add_operation(
            [A, self.W, self.b], 
            Z, 
            [None, self.dW, self.db], 
            conv2d_stride1_backward
        )        

        # TODO: Return output
        return Z


class Conv2d():
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 downsampling_factor,
                 autograd_engine,
                 weight_init_fn=None,
                 bias_init_fn=None):
        # Do not modify the variable names
        self.downsampling_factor = downsampling_factor

        self.autograd_engine = autograd_engine

        # TODO: Initialize Conv2D() instance (remember to pass the autograd_engine)
        self.conv2d_stride1 = Conv2D_stride1(in_channel, out_channel, kernel_size, autograd_engine, weight_init_fn, bias_init_fn)

        # TODO: Initialize Downsample2d() instance (remember to pass the autograd_engine)
        self.downsample2d = Downsample2d(downsampling_factor, autograd_engine)

    def __call__(self, A):
        return self.forward(A)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channel, output_width, output_height)
        """

        # TODO: Call Conv2D_stride1
        self.Z = self.conv2d_stride1.forward(A)

        # TODO: Downsample
        # NOTE: The add operation for this occurs in resampling.py
        #       so no need to do it here.
        self.Z = self.downsample2d.forward(self.Z) #TODO

        # TODO: Return output

        return self.Z


class Flatten():
    def __init__(self, autograd_engine):
        self.autograd_engine = autograd_engine

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, in_width)
        Return:
            out (np.array): (batch_size, in_channel * in width)
        """
        # Reshape x into a 2D tensor with shape (batch_size, -1)
        return x.reshape(x.shape[0], -1)
