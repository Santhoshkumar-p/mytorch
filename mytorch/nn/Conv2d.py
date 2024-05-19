import numpy as np
from resampling import *


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels,
                 kernel_size, weight_init_fn=None, bias_init_fn=None):

        # Do not modify this method

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        self.A = A

        Z = None  # TODO

        batch_size, in_channels, self.input_height, self.input_width = self.A.shape
        output_height = (self.input_height - self.kernel_size) + 1
        output_width = (self.input_width - self.kernel_size) + 1
        
        Z = np.zeros((batch_size, self.out_channels, output_height, output_width))
        
        for h_idx in range(output_height):
            for w_idx in range(output_width):
                convolve = self.A[:, :, h_idx:(h_idx+self.kernel_size), w_idx:(w_idx+self.kernel_size)]
                Z[:, : , h_idx, w_idx] =  np.tensordot(convolve, self.W, axes=([1, 2, 3], [1, 2, 3])) + self.b

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        batch_size, out_channels, output_height, output_width = dLdZ.shape
        
        for h_idx in range(self.kernel_size): # TODO
            for w_idx in range(self.kernel_size):
                convolve = self.A[:, :, h_idx:h_idx+output_height, w_idx:w_idx+output_width]
                self.dLdW[:, :, h_idx, w_idx] =  np.tensordot(dLdZ , convolve, axes=([0, 2, 3], [0, 2, 3]))

        self.dLdb = np.sum(dLdZ, axis=(0, 2, 3))  # TODO

        padd = (self.kernel_size - 1, self.kernel_size - 1)
        dLdZ_padded = np.pad(dLdZ, ((0, 0), (0, 0), padd, padd), mode='constant', constant_values=0)
        dLdA = np.zeros((batch_size, self.in_channels, self.input_height, self.input_width))  # TODO
        for h_idx in range(self.input_height):
            for w_idx in range(self.input_width):
                convolve = dLdZ_padded[:, :, h_idx:(h_idx+self.kernel_size), w_idx:(w_idx+self.kernel_size)]
                dLdA[:, :, h_idx, w_idx] = np.tensordot(convolve, np.flip(self.W, (2,3)), axes=([1, 2, 3], [0, 2, 3]))

        return dLdA



class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride
        self.pad = padding

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels=in_channels, 
                                            out_channels=out_channels,
                                            kernel_size=kernel_size, 
                                            weight_init_fn=weight_init_fn, 
                                            bias_init_fn=bias_init_fn
        )  # TODO
        self.downsample2d = Downsample2d(downsampling_factor=self.stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """

        # Pad the input appropriately using np.pad() function
        # TODO
        #2D Padding
        padded_A = np.pad(A, ((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)), 'constant', constant_values=0)

        # Call Conv2d_stride1
        # TODO
        convolved = self.conv2d_stride1.forward(padded_A)

        # downsample
        Z = self.downsample2d.forward(convolved)  # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        # Call downsample1d backward
        # TODO
        downsampled_backward = self.downsample2d.backward(dLdZ=dLdZ)

        # Call Conv2d_stride1 backward
        convolve_backward = self.conv2d_stride1.backward(downsampled_backward)  # TODO

        # Unpad the gradient
        # TODO
        dLdA = convolve_backward
        if self.pad is not 0:
            dLdA = convolve_backward[:, :, self.pad:-self.pad, self.pad:-self.pad]
        return dLdA
