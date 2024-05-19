import numpy as np
from mytorch.autograd_engine import Autograd

"""
Mathematical Functionalities
    These are some IMPORTANT things to keep in mind:
    - Make sure grad of inputs are exact same shape as inputs.
    - Make sure the input and output order of each function is consistent with
        your other code.
    Optional:
    - You can account for broadcasting, but it is not required 
        in the first bonus.
"""

def identity_backward(grad_output, a):
    """Backward for identity. Already implemented."""

    return np.array(grad_output)

def add_backward(grad_output, a, b):
    """Backward for addition. Already implemented."""
    
    a_grad = grad_output * np.ones(a.shape)
    b_grad = grad_output * np.ones(b.shape)

    return a_grad, b_grad


def sub_backward(grad_output, a, b):
    """Backward for subtraction"""
    a_grad = grad_output * np.ones(a.shape)
    b_grad = -grad_output * np.ones(b.shape)

    return a_grad, b_grad


def matmul_backward(grad_output, a, b):
    """Backward for matrix multiplication"""
    a_grad = np.dot(grad_output, b.T)
    b_grad = np.dot(a.T, grad_output)
    return a_grad, b_grad


def mul_backward(grad_output, a, b):
    """Backward for multiplication"""
    a_grad = grad_output * b
    b_grad = grad_output * a
    return a_grad, b_grad


def div_backward(grad_output, a, b):
    """Backward for division"""
    a_grad = grad_output * 1/b
    b_grad = -grad_output * (a / (b**2))
    return a_grad, b_grad


def log_backward(grad_output, a):
    """Backward for log"""
    a_grad = grad_output / a
    return a_grad


def exp_backward(grad_output, a):
    """Backward of exponential"""
    a_grad = grad_output * np.exp(a)
    return a_grad


def max_backward(grad_output, a):
    """Backward of max"""
    
    max_indices = np.argmax(a, axis=-1)
    
    binary_matrix = np.zeros_like(a)
    binary_matrix[np.arange(len(max_indices)), max_indices] = 1

    grad_a = grad_output * binary_matrix

    return grad_a


def sum_backward(grad_output, a):
    """Backward of sum"""
    a_grad = grad_output * np.ones_like(a)
    return a_grad


def SoftmaxCrossEntropy_backward(grad_output, a):
    """
    NOTE: Since the gradient of the Softmax CrossEntropy Loss is
          is straightforward to compute, you may choose to implement
          this directly rather than rely on the backward functions of
          more primitive operations.
    """
    batch_size = a.shape[0]
    softmax_output = np.exp(a - np.max(a, axis=1, keepdims=True)) / np.sum(np.exp(a - np.max(a, axis=1, keepdims=True)), axis=1, keepdims=True)  # TODO
        
    a_grad = (softmax_output - a) / batch_size
    return np.array(a_grad)
