# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np


class Dropout2d(object):
    def __init__(self, p=0.5):
        # Dropout probability
        self.p = p

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x, eval=False):
        """
        Arguments:
          x (np.array): (batch_size, in_channel, input_width, input_height)
          eval (boolean): whether the model is in evaluation mode
        Return:
          np.array of same shape as input x
        """
        # 1) Get and apply a per-channel mask generated from np.random.binomial
        # 2) Scale your output accordingly
        # 3) During test time, you should not apply any mask or scaling.   
        N, C, W, H = x.shape
        if eval == False:
          mask_bit = []
          for n in range(N):
            batch_temp = []
            # self.channels = np.random.binomial(1, 1-self.p, size=(1, x.shape[1], 1, 1))
            prob = np.random.binomial(1, self.p, C)
            for i in prob:
              temp_channel = np.ones((W, H)) if i == 0 else np.zeros((W, H))
              batch_temp.append(temp_channel)
            batch_temp = np.reshape(batch_temp, (C, W, H))
            mask_bit.append(batch_temp)
            # self.mask = np.tile(self.channels, (x.shape[0], 1, x.shape[2], x.shape[3]))
          self.mask = np.reshape(mask_bit, (x.shape))
          x = (x * mask_bit) / (1 - self.p)
        return x  


    def backward(self, delta):
        """
        Arguments:
          delta (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
          np.array of same shape as input delta
        """
        # 1) This method is only called during training.
        # 2) You should scale the result by chain rule
        #TODO
        return (delta * self.mask) / (1 - self.p)
    
    def _apply_mask(self, array: np.array, mask: np.array) -> np.array:
        array *= mask
        array /= self.p
        return array
