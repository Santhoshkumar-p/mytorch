import numpy as np


class Upsample1d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        batch_size, in_channels, input_width = A.shape
        output_width = (self.upsampling_factor * (input_width - 1)) + 1
        k = (self.upsampling_factor - 1) #kernel
        Z = np.zeros((batch_size, in_channels, output_width))  # TODO
        for idx in range(input_width):
            Z[:, :, (idx * self.upsampling_factor)] = A[:, :, idx]
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        batch_size, in_channels, output_width = dLdZ.shape
        input_width = int(((output_width - 1) // self.upsampling_factor) + 1)
        dLdA = np.zeros((batch_size, in_channels, input_width))  # TODO
        k = (self.upsampling_factor - 1) #kernel
        for idx in range(input_width):
            dLdA[:, : , idx] = dLdZ[:, :, (idx * self.upsampling_factor)]
        return dLdA


class Downsample1d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        batch_size, in_channels, self.input_width = A.shape
        output_width = int(((self.input_width - 1) // self.downsampling_factor) + 1)
        Z = np.zeros((batch_size, in_channels, output_width))  # TODO
        for idx in range(output_width):
            Z[:, : , idx] = A[:, :, (idx * self.downsampling_factor)]
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        batch_size, in_channels, output_width = dLdZ.shape
        dLdA = np.zeros((batch_size, in_channels, self.input_width))  # TODO
        for idx in range(output_width):
            dLdA[:, : , idx * self.downsampling_factor] = dLdZ[:, :, idx]
        return dLdA


class Upsample2d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        batch_size, in_channels, input_height, input_width = A.shape
        output_height = (self.upsampling_factor * (input_width - 1)) + 1
        output_width = (self.upsampling_factor * (input_height - 1)) + 1
        Z = np.zeros((batch_size, in_channels, output_height, output_width))  # TODO
        for w_idx in range(input_width):
            for h_idx in range(input_height):
                Z[:, :, (h_idx * self.upsampling_factor), (w_idx * self.upsampling_factor)] = A[:, :, h_idx, w_idx]
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        batch_size, in_channels, output_height, output_width = dLdZ.shape
        input_height = int(((output_height - 1) // self.upsampling_factor) + 1)
        input_width = int(((output_width - 1) // self.upsampling_factor) + 1)
        dLdA = np.zeros((batch_size, in_channels, input_height, input_width))  # TODO
        for w_idx in range(input_width):
            for h_idx in range(input_height):
                dLdA[:, : , h_idx, w_idx] = dLdZ[:, :, (h_idx * self.upsampling_factor), (w_idx * self.upsampling_factor)]
        return dLdA


class Downsample2d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        batch_size, in_channels, self.input_height, self.input_width = A.shape
        output_width = int(((self.input_width - 1) // self.downsampling_factor) + 1)
        output_height = int(((self.input_height - 1) // self.downsampling_factor) + 1)
        Z = np.zeros((batch_size, in_channels, output_height, output_width))  # TODO
        for w_idx in range(output_width):
            for h_idx in range(output_height):
                Z[:, : , h_idx, w_idx] = A[:, :, (h_idx * self.downsampling_factor), (w_idx * self.downsampling_factor)]
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        batch_size, in_channels, output_height, output_width = dLdZ.shape
        dLdA = np.zeros((batch_size, in_channels, self.input_height, self.input_width))  # TODO
        for w_idx in range(output_width):
            for h_idx in range(output_height):
                dLdA[:, : , (h_idx * self.downsampling_factor), (w_idx * self.downsampling_factor)] = dLdZ[:, :, h_idx, w_idx]
        return dLdA
