import numpy as np


class MSELoss:

    def forward(self, A, Y):
        """
        Calculate the Mean Squared error
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: MSE Loss(scalar)

        """

        self.A = A
        self.Y = Y
        self.N, self.C = A.shape 
        se = (A - Y) * (A - Y)
        Ones_C = np.ones(self.C)
        Ones_N = np.ones(self.N)
        sse = np.dot(np.dot((Ones_N.reshape(1, -1)),  (se)), (Ones_C))
        mse = sse / (self.N * self.C)

        return mse

    def backward(self):

        dLdA = (2 * (self.A - self.Y)) / (self.N * self.C)

        return dLdA


class CrossEntropyLoss:

    def forward(self, A, Y):
        """
        Calculate the Cross Entropy Loss
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: CrossEntropyLoss(scalar)

        Refer the the writeup to determine the shapes of all the variables.
        Use dtype ='f' whenever initializing with np.zeros()
        """
        self.A = A
        self.Y = Y
        self.N, self.C = A.shape

        Ones_C = np.ones(self.C)
        Ones_N = np.ones(self.N)

        self.softmax = np.exp(A) / np.exp(A).sum(axis=1, keepdims=True)
        crossentropy = np.dot((-Y * np.log(self.softmax)) , (Ones_C))
        sum_crossentropy = np.dot(Ones_N, crossentropy)
        L = sum_crossentropy / self.N

        return L

    def backward(self):

        dLdA = (self.softmax - self.Y) / self.N

        return dLdA
