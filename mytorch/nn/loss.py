import numpy as np
from mytorch.functional_hw1 import (
    matmul_backward,
    add_backward,
    sub_backward,
    mul_backward,
    div_backward,
    SoftmaxCrossEntropy_backward,
    sum_backward,
)


class LossFN(object):
    """
    Interface for loss functions.

    The class serves as an abstract base class for different loss functions.
    The forward() method should be completed by the derived classes.

    This class is similar to the wrapper functions for the activations
    that you wrote in functional.py with a couple of key differences:
        1. Notice that instead of passing the autograd object to the forward
            method, we are instead saving it as a class attribute whenever
            an LossFN() object is defined. This is so that we can directly
            call the backward() operation on the loss as follows:
                >>> loss_fn = LossFN(autograd_object)
                >>> loss_val = loss_fn(y, y_hat)
                >>> loss_fn.backward()

        2. Notice that the class has an attribute called self.loss_val.
            You must save the calculated loss value in this variable.
            This is so that we do not explicitly pass the divergence to
            the autograd engine's backward method. Rather, calling backward()
            on the LossFN object will take care of that for you.

    WARNING: DO NOT MODIFY THIS CLASS!
    """

    def __init__(self, autograd_engine):
        self.autograd_engine = autograd_engine
        self.loss_val = None

    def __call__(self, y, y_hat):
        return self.forward(y, y_hat)

    def forward(self, y, y_hat):
        """
        Args:
            - y (np.ndarray) : the ground truth,
            - y_hat (np.ndarray) : the output computed by the network,

        Returns:
            - self.loss_val : the calculated loss value
        """
        raise NotImplementedError

    def backward(self):
        # Call autograd's backward here.
        self.autograd_engine.backward(self.loss_val)


class MSELoss(LossFN):
    def __init__(self, autograd_engine):
        super(MSELoss, self).__init__(autograd_engine)

    def forward(self, y, y_hat):
        # TODO: Use the primitive operations to calculate the MSE Loss
        #error = (y - y_hat) ** 2 / N
        N = len(y)
        diff = np.sub(y, y_hat)
        squared_diff = np.matmul(diff, diff)
        sum_squared_diff = np.sum(squared_diff)
        mean_squared_error = np.divide(sum_squared_diff, N)
        # TODO: Remember to use add_operation to record these operations in
        #       the autograd engine after each operation
        # self.loss_val = ...
        self.loss_val = np.array(mean_squared_error)
        self.autograd_engine.add_operation([np.array(y), np.array(y_hat)], np.array(diff), [None, None], sub_backward)
        self.autograd_engine.add_operation([np.array(diff), np.array(diff)], np.array(squared_diff), [None, None], matmul_backward)
        self.autograd_engine.add_operation([np.array(squared_diff)], np.array(sum_squared_diff), [None], sum_backward)
        self.autograd_engine.add_operation([np.array(sum_squared_diff), np.array(N)], np.array(mean_squared_error), [None, None], div_backward)

        return self.loss_val.reshape(1,)
        


# Hint: To simplify things you can just make a backward for this loss and not
# try to do it for every operation.
class SoftmaxCrossEntropy(LossFN):
    """
    :param A: Output of the model of shape (N, C)
    :param Y: Ground-truth values of shape (N, C)

    self.A = A
    self.Y = Y
    self.N = A.shape[0]
    self.C = A.shape[-1]

    Ones_C = np.ones((self.C, 1))
    Ones_N = np.ones((self.N, 1))

    self.softmax = np.exp(self.A) / np.sum(np.exp(self.A), axis=1, keepdims=True)
    crossentropy = (-1 * self.Y * np.log(self.softmax)) @ Ones_C
    sum_crossentropy = Ones_N.T @ crossentropy
    L = sum_crossentropy / self.N
    """
    def __init__(self, autograd_engine):
        super(SoftmaxCrossEntropy, self).__init__(autograd_engine)

    def forward(self, y, y_hat):
        # # TODO: calculate loss value and set self.loss_val
        # # To simplify things, add a single operation corresponding to the
        # # backward function created for this loss
        # # Compute forward with primitive operations
        N, C = y_hat.shape
        ones_C = np.ones((C, 1))
        ones_N = np.ones((N, 1))

        # Compute softmax manually
        exp_y_hat = np.exp(y_hat)
        sum_exp_y_hat = np.sum(exp_y_hat, axis=1, keepdims=True)
        softmax = exp_y_hat / sum_exp_y_hat

        crossentropy = (-1 * y * np.log(softmax)) @ ones_C
        sum_crossentropy = ones_N.T @ crossentropy
        self.loss_val = sum_crossentropy / N

        self.autograd_engine.add_operation(
            inputs=[np.array(y_hat)],
            output=np.array(self.loss_val),
            gradients_to_update=[None],
            backward_operation=SoftmaxCrossEntropy_backward
        )

        return self.loss_val.reshape(1,)
