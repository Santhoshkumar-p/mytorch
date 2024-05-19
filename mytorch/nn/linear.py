import numpy as np
from mytorch.functional_hw1 import matmul_backward, add_backward


class Linear():
    def __init__(self, in_features, out_features, autograd_engine):
        """
        Do not modify
        """
        self.W = np.random.uniform(
            -np.sqrt(1 / in_features), 
            np.sqrt(1 / in_features),
            size=(out_features, in_features)
        )  # flip this to out x in to mimic pytorch
        self.b = np.random.uniform(
            -np.sqrt(1 / in_features), 
            np.sqrt(1 / in_features),
            size=(out_features, 1)
        )  # just change this to 1-d after implementing broadcasting
        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)
        self.momentum_W = np.zeros(self.W.shape)
        self.momentum_b = np.zeros(self.b.shape)
        self.autograd_engine = autograd_engine

    def __call__(self, x):

        return self.forward(x)

    
    def forward(self, x):
        """
            Computes the affine transformation forward pass of the Linear Layer

            Args:
                - x (np.ndarray): the input array,

            Returns:
                - (np.ndarray), the output of this forward computation.
        """
        
        # TODO: Use the primitive operations to calculate the affine transformation
        #      of the linear layer
        self.x = x
        weight_mul = np.matmul(self.x, self.W.T)
        affine = np.add(weight_mul, self.b.T)
        # TODO: Remember to use add_operation to record these operations in
        #      the autograd engine after each operation
        self.autograd_engine.add_operation([self.x, self.W.T], weight_mul, [None, self.dW.T], matmul_backward)
        self.autograd_engine.add_operation([weight_mul, self.b.T], affine, [None, self.db.T], add_backward)
        # TODO: remember to return the computed value
        return affine
