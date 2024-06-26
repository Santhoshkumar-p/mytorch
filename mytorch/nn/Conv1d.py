import numpy as np
from resampling import *


class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A
        batch_size, in_channels, self.input_size = self.A.shape
        output_size = (self.input_size - self.kernel_size) + 1
        Z = np.zeros((batch_size, self.out_channels, output_size))
        for idx in range(output_size):
            convolve = self.A[:, :, idx:(idx+self.kernel_size)]
            Z[:, : , idx] = np.sum(convolve[:, None, :, :] * self.W, axis=(2, 3)) + self.b 
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        batch_size, out_channels, output_size = dLdZ.shape
        
        for idx in range(self.kernel_size): 
            convolve = self.A[:, :, idx:idx+output_size]
            self.dLdW[:, :, idx] =  np.tensordot(dLdZ , convolve, axes=([0, 2], [0, 2]))

        self.dLdb = np.sum(dLdZ, axis=(0,2))  

        padd = (self.kernel_size - 1, self.kernel_size - 1)
        dLdZ_padded = np.pad(dLdZ, ((0, 0), (0, 0), padd), mode='constant', constant_values=0)
        dLdA = np.zeros((batch_size, self.in_channels, self.input_size))  
        for idx in range(self.input_size):
            convolve = dLdZ_padded[:, :, idx:(idx+self.kernel_size)]
            dLdA[:, :, idx] = np.tensordot(convolve, np.flip(self.W, 2), axes=([1, 2], [0, 2]))

        return dLdA


class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,padding = 0,
                 weight_init_fn=None, bias_init_fn=None):

        self.stride = stride
        self.pad = padding

        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(in_channels=in_channels, 
                                            out_channels=out_channels, 
                                            kernel_size=kernel_size,
                                            weight_init_fn=weight_init_fn,
                                            bias_init_fn=bias_init_fn)  
        self.downsample1d = Downsample1d(downsampling_factor=self.stride)  

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        # Pad the input appropriately using np.pad() function
        
        padded_A = np.pad(A, ((0, 0), (0, 0), (self.pad, self.pad)), 'constant', constant_values=0)


        # Call Conv1d_stride1
        
        convolved = self.conv1d_stride1.forward(padded_A)

        # downsample
        Z = self.downsample1d.forward(convolved)  

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call downsample1d backward
        
        downsampled_backward = self.downsample1d.backward(dLdZ=dLdZ)

        # Call Conv1d_stride1 backward
        convolved_backward = self.conv1d_stride1.backward(dLdZ=downsampled_backward)  

        # Unpad the gradient
        
        end_idx = convolved_backward.shape[-1] - self.pad
        dLdA = convolved_backward[:, :, self.pad:end_idx]

        return dLdA
