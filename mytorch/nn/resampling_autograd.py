import numpy as np
from mytorch.functional_hw1 import *
from mytorch.functional_hw2 import *

class Upsample2d():
    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        # TODO: Implement forward (you can rely on HW2P1 code)
        batch_size, input_channels, input_height, input_width = A.shape
        scale = self.upsampling_factor

        output_height = scale * (input_height - 1) + 1

        output_width = scale * (input_width - 1) + 1

        Z = np.zeros((batch_size, input_channels, output_height, output_width))

        Z[:, :, ::scale, ::scale] = A

        return Z

    def backward(self, dLdZ):
        # TODO: Implement backward (you can rely on HW2P1 code)
        batch_size, in_channels, output_height, output_width = dLdZ.shape
        scale = self.upsampling_factor
        input_height = (output_height + 1) // scale

        input_width = (output_width + 1) // scale

        dLdA = dLdZ[:, :, ::scale, ::scale]

        return dLdA


class Downsample2d():
    def __init__(self, downsampling_factor, autograd_engine):
        self.downsampling_factor = downsampling_factor
        self.autograd_engine = autograd_engine

    def forward(self, A):
        # TODO: Implement forward (you can rely on HW2P1 code)
        # TODO: Add operation to autograd_engine
        batch_size, in_channels, input_height, input_width = A.shape
        scale = self.downsampling_factor
        self.input_height, self.input_width = input_height, input_width

        Z = A[:, :, ::scale, ::scale]

        self.autograd_engine.add_operation(
            [A, np.array([scale])], 
            Z, 
            [None, None], 
            downsampling2d_backward
        )
        
        return Z


class Upsample1d():
    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        # TODO: Implement forward (you can rely on HW2P1 code)
        batch_size, in_channels, input_width = A.shape
        scale = self.upsampling_factor
        output_width = scale * (input_width - 1) + 1
        
        Z = np.zeros((batch_size, in_channels, output_width))
        
        for i in range(input_width):

            Z[:, :, i * scale] = A[:, :, i]
        
        return Z

    def backward(self, dLdZ):
        # TODO: Implement backward (you can rely on HW2P1 code)
        batch_size, input_channels, output_width = dLdZ.shape
        scale = self.upsampling_factor
        input_width = ((output_width - 1) // scale) + 1
        
        dLdA = np.zeros((batch_size, input_channels, input_width))
        
        for i in range(input_width):

            dLdA[:, :, i] = dLdZ[:, :, i*scale]

        return dLdA


class Downsample1d():
    def __init__(self, downsampling_factor, autograd_engine):
        self.downsampling_factor = downsampling_factor
        self.autograd_engine = autograd_engine

    def forward(self, A):
        # TODO: Implement forward (you can rely on HW2P1 code)
        # TODO: Add operation to autograd_engine
        batch_size, input_channels, input_width = A.shape
        scale = self.downsampling_factor 
        self.input_width = input_width

        output_width = (input_width + scale - 1) // scale
        
        Z = A[:, :, ::scale]

        self.autograd_engine.add_operation(
            [A, np.array([scale])], 
            Z, 
            [None, None], 
            downsampling1d_backward
        )
        
        return Z
