import numpy as np
from scipy import special
import math

class Identity:

    def forward(self, Z):

        self.A = Z

        return self.A

    def backward(self, dLdA):

        dAdZ = np.ones(self.A.shape, dtype="f")
        dLdZ = dLdA * dAdZ

        return dLdZ


class Sigmoid:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Sigmoid.
    """
    def forward(self, Z):
        self.A = 1 / (1 + np.exp(-Z)) #np.array([1 / (1 + np.exp(-1*val)) for val in Z])
        return self.A
    
    def backward(self, dLdA):
        dLdZ = dLdA * (self.A - self.A * self.A)
        return dLdZ



class Tanh:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Tanh.
    """
    def forward(self, Z):
        self.A = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
        return self.A
    
    def backward(self, dLdA):
        dLdZ = dLdA * (1 - (self.A * self.A)) #1- tanh^2X
        return dLdZ


class ReLU:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on ReLU.
    """
    def forward(self, Z):
        self.A = np.maximum(0, Z)
        return self.A
    
    def backward(self, dLdA):
        dLdZ = np.where(self.A <= 0, 0, dLdA)
        return dLdZ

class GELU:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on GELU.
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
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Softmax.
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