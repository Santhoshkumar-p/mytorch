import numpy as np
from mytorch.functional_1 import *


class Activation(object):
    """
    Interface for activation functions (non-linearities).

    In all implementations, the state attribute must contain the result,
    i.e. the output of forward.
    """

    # Note that these activation functions are scalar operations. I.e, they
    # shouldn't change the shape of the input.

    def __init__(self, autograd_engine):
        self.state = None
        self.autograd_engine = autograd_engine

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplementedError


class Identity(Activation):
    def __init__(self, autograd_engine):
        super(Identity, self).__init__(autograd_engine)

    def forward(self, x):
        self.state = x
        self.autograd_engine.add_operation([x], x, [None], identity_backward)
        return self.state


class Sigmoid(Activation):
    """
    Sigmoid activation.
    """
    def __init__(self, autograd_engine):
        super(Sigmoid, self).__init__(autograd_engine)

    def forward(self, x):

        # NOTE: Compute forward with primitive operations
        self.state = 1 / (1 + np.exp(-x))
        exp = np.exp(-x)
        add = np.add(1, exp)
        div = 1 / (add)
        # NOTE: Add operations to the autograd engine as you go
        self.autograd_engine.add_operation([-x], exp, [None], exp_backward)
        self.autograd_engine.add_operation([np.array(1), exp], add, [None, None], add_backward)
        self.autograd_engine.add_operation([np.array(1), add], div, [None, None], div_backward)

        return self.state


class Tanh(Activation):
    """
    Tanh activation.
    """
    def __init__(self, autograd_engine):
        super(Tanh, self).__init__(autograd_engine)

    def forward(self, x):

        # NOTE: Compute forward with primitive operations
        two_x = 2 * x
        exp_2x = np.exp(two_x)
        exp_2x_minus_1 = exp_2x - 1
        exp_2x_plus_1 = exp_2x + 1
        tanh_result = exp_2x_minus_1 / exp_2x_plus_1

        # NOTE: Add operations to the autograd engine as you go
        self.autograd_engine.add_operation([two_x], exp_2x, [None], exp_backward)
        self.autograd_engine.add_operation([exp_2x, np.array(1)], exp_2x_minus_1, [None, None], sub_backward)
        self.autograd_engine.add_operation([exp_2x, 1], exp_2x_plus_1, [None, None], add_backward)
        self.autograd_engine.add_operation([exp_2x_minus_1, exp_2x_plus_1], tanh_result, [None, None], div_backward)
        return np.tanh(x)


class ReLU(Activation):
    """
    ReLU activation.
    """
    def __init__(self, autograd_engine):
        super(ReLU, self).__init__(autograd_engine)

    def forward(self, x):
        # NOTE: Compute forward with primitive operations
        relu_result = np.maximum(0, x)
        # NOTE: Add operations to the autograd engine as you go
        self.autograd_engine.add_operation([x], relu_result, [None], max_backward)
        return relu_result
        
class GELU:
    """
    GELU activation.
    """
    def forward(self, Z):
        self.A = (0.5 * Z) * (1 + special.erf(Z / math.sqrt(2)))
        self.Z = Z
        return self.A
    
    def backward(self, dLdA):
        #a = (0.5 * (1 + special.erf(self.A / math.sqrt(2))))
        #b = (((self.A / math.sqrt(2 * math.pi)) * (np.exp(((-self.A * self.A) / 2)))))
        a = (0.5 * (1 + special.erf(self.Z / math.sqrt(2))))
        b = (((self.Z / math.sqrt(2 * math.pi)) * (np.exp(((-self.Z * self.Z) / 2)))))
        dLdZ = dLdA * (a + b)
        return dLdZ

class Softmax:
    """
    Softmax.
    """

    def forward(self, Z):
        """
        Remember that Softmax does not act element-wise.
        It will use an entire row of Z to compute an output element.
        """
        self.A = np.exp(Z) / np.exp(Z).sum(axis=1, keepdims=True)
        return self.A
    
    def backward(self, dLdA):

        # Calculate the batch size and number of features
        N,C = dLdA.shape

        # Initialize the final output dLdZ with all zeros. Refer to the writeup and think about the shape.
        dLdZ = np.zeros((N, C))

        # Fill dLdZ one data point (row) at a time
        for i in range(N):

            # Initialize the Jacobian with all zeros.
            J = np.zeros((C, C)) 
            # Fill the Jacobian matrix according to the conditions described in the writeup
            for m in range(C):
                for n in range(C):
                    if m == n:
                        J[m,n] = self.A[i, m] * (1 - self.A[i, m])
                    else:
                        J[m,n] = -1 * self.A[i, m] * self.A[i, n]

            # Calculate the derivative of the loss with respect to the i-th input
            dLdZ[i,:] = np.dot(dLdA[i, :] , J)

        return dLdZ
