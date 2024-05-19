import numpy as np
from mytorch.autograd_engine import Autograd


def conv1d_stride1_backward(dLdZ, A, weight, bias):
    """
    Inputs
    ------
    dLdz:   Gradient from next layer
    A:      Input
    weight: Model param
    bias:   Model param

    Returns
    -------
    dLdA, dLdW, dLdb
    """
    # NOTE: You can use code from HW2P1!
    batch_size, out_channels, output_size = dLdZ.shape
        

    _, input_channels, input_size = A.shape
    
    kernel_size = weight.shape[2]

    padd = (kernel_size - 1, kernel_size - 1)

    dLdZ_padded = np.pad(dLdZ, ((0, 0), (0, 0), padd), mode='constant', constant_values=0)
    
    dLdA = np.zeros(A.shape)

    dLdW = np.zeros(weight.shape)

    dLdb = np.zeros(bias.shape)  

    for idx in range(input_size):
        convolve = dLdZ_padded[:, :, idx:(idx+kernel_size)]
        dLdA[:, :, idx] = np.tensordot(convolve, np.flip(weight, 2), axes=([1, 2], [0, 2]))

    for idx in range(kernel_size): 
        convolve = A[:, :, idx:idx+output_size]
        dLdW[:, :, idx] =  np.tensordot(dLdZ , convolve, axes=([0, 2], [0, 2]))
    dLdb = np.sum(dLdZ, axis = (0, 2))
    return dLdA, dLdW, dLdb

def conv2d_stride1_backward(dLdZ, A, weight, bias):
    """
    Inputs
    ------
    dLdz:   Gradient from next layer
    A:      Input
    weight: Model param
    bias:   Model param

    Returns
    -------
    dLdA, dLdW, dLdb
    """
    # NOTE: You can use code from HW2P1!
    batch_size, output_channels, output_height, output_width = dLdZ.shape
    
    _, input_channels, input_height, input_width = A.shape
    kernel_size = weight.shape[2]
    
    dLdA = np.zeros(A.shape)  # TODO
    
    dLdW = np.zeros(weight.shape)

    dLdb = np.sum(dLdZ, axis=(0, 2, 3)) 

    padd = (kernel_size - 1, kernel_size - 1)
    dLdZ_padded = np.pad(dLdZ, ((0, 0), (0, 0), padd, padd), mode='constant', constant_values=0)
    
    for h_idx in range(kernel_size): # TODO
        for w_idx in range(kernel_size):
            convolve = A[:, :, h_idx:h_idx+output_height, w_idx:w_idx+output_width]
            dLdW[:, :, h_idx, w_idx] =  np.tensordot(dLdZ , convolve, axes=([0, 2, 3], [0, 2, 3]))

      # TODO
    for h_idx in range(input_height):
        for w_idx in range(input_width):
            convolve = dLdZ_padded[:, :, h_idx:(h_idx+kernel_size), w_idx:(w_idx+kernel_size)]
            dLdA[:, :, h_idx, w_idx] = np.tensordot(convolve, np.flip(weight, (2,3)), axes=([1, 2, 3], [0, 2, 3]))

    return dLdA, dLdW, dLdb


def downsampling1d_backward(dLdZ, A, downsampling_factor):
    """
    Inputs
    ------
    dLdz:                   Gradient from next layer
    A:                      Input
    downsampling_factor:    NOTE: for the gradient buffer to work, 
                            this has to be a np.array. 

    Returns
    -------
    dLdA, dLdW, dLdb
    """
    batch_size, input_channels, input_width = A.shape
    output_width = dLdZ.shape[2]
    dLdA = np.zeros(A.shape)  # TODO
    scale = downsampling_factor
    dLdA[:, :, ::int(scale[0])] = dLdZ
    return dLdA, None, None


def downsampling2d_backward(dLdZ, A, downsampling_factor):
    """
    Inputs
    ------
    dLdz:                   Gradient from next layer
    A:                      Input
    downsampling_factor:    NOTE: for the gradient buffer to work, 
                            this has to be a np.array. 

    Returns
    -------
    dLdA, dLdW, dLdb
    """
    # NOTE: You can use code from HW2P1!
    batch_size, in_channels, output_height, output_width = dLdZ.shape
    dLdA = np.zeros(A.shape)  # TODO
    scale = downsampling_factor
    dLdA[:, :, ::int(scale[0]), ::int(scale[0])] = dLdZ
    return dLdA, None, None


def flatten_backward(dLdZ, A):
    """
    Inputs
    ------
    dLdz:   Gradient from next layer
    A:      Input

    Returns
    -------
    dLdA
    """
    # NOTE: You can use code from HW2P1!
    batch_size, input_channels, output_height, output_width = dLdZ.shape
    _, input_channels, input_height, input_width = A.shape
    dLdA = dLdZ.reshape(batch_size, input_channels, input_width)  # TODO
    return dLdA
