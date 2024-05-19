import numpy as np
from resampling import *


class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        batch_size, self.in_channels, self.input_width, self.input_height = A.shape

        output_width  = self.input_width - self.kernel + 1 
        output_height = self.input_height - self.kernel + 1

        Z = np.zeros((batch_size, self.in_channels, output_width, output_height))

        self.maxIndices = np.zeros(Z.shape, dtype = object)

        for b in range(batch_size):

            for c in range(self.in_channels):

                for w in range(output_width):

                    for h in range(output_height):

                        patch = A[b, c, w:w + self.kernel, h:h + self.kernel]

                        ind = np.unravel_index(np.argmax(patch), patch.shape)

                        self.maxIndices[b, c, w, h] = (w + ind[0], h + ind[1])

                        Z[b, c, w, h] = patch[ind[0], ind[1]]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        batch_size, out_channels, output_width, output_height = dLdZ.shape
        dLdA = np.zeros((batch_size, self.in_channels, self.input_width, self.input_height))
        
        for b in range(batch_size):

            for c in range(self.in_channels):

                for w in range(output_width):

                    for h in range(output_height):

                        ind = self.maxIndices[b, c, w, h]

                        dLdA[b, c, ind[0], ind[1]] += dLdZ[b,c,w,h]
        return dLdA 

class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        batch_size, self.in_channels, self.input_width, self.input_height = A.shape

        output_width  = self.input_width - self.kernel + 1 
        output_height = self.input_height - self.kernel + 1

        Z = np.zeros((batch_size, self.in_channels, output_width, output_height))

        for b in range(batch_size):

            for c in range(self.in_channels):

                for w in range(output_width):

                    for h in range(output_height):

                        patch = A[b, c, w:w + self.kernel, h:h + self.kernel]

                        Z[b, c, w, h] = patch.mean()
        return Z


    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        batch_size, out_channels, output_width, output_height = dLdZ.shape

        dLdA = np.zeros((batch_size, self.in_channels, self.input_width, self.input_height))

        for b in range(batch_size):

            for c in range(self.in_channels):

                for w in range(output_width):

                    for h in range(output_height):

                        for l in range(self.kernel):

                            for m in range(self.kernel):

                                dLdA[b, c, w + l, h + m] += dLdZ[b, c, w, h] / (self.kernel ** 2)

        return dLdA


class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(self.kernel)  # TODO
        self.downsample2d = Downsample2d(self.stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        maxpooled = self.maxpool2d_stride1.forward(A)

        Z = self.downsample2d.forward(maxpooled)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        downsampled = self.downsample2d.backward(dLdZ)

        dLdA = self.maxpool2d_stride1.backward(downsampled)

        return dLdA


class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(self.kernel)  # TODO
        self.downsample2d = Downsample2d(self.stride) # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        meanpooled = self.meanpool2d_stride1.forward(A)

        Z = self.downsample2d.forward(meanpooled)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        downsampled = self.downsample2d.backward(dLdZ)
        dLdA = self.meanpool2d_stride1.backward(downsampled)
        return dLdA
