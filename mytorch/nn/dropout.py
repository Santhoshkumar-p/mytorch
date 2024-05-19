import numpy as np

class Dropout(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x, train=True):

        if train:
            #  Generate mask and apply to x
            self.mask = np.random.binomial(1, 1 - self.p, size=x.shape)
            return (x * self.mask) / (1 - self.p)    
        else:
            #  Return x as is
            return x
		
    def backward(self, delta):
        #  Multiply mask with delta and return
        return self.mask * delta