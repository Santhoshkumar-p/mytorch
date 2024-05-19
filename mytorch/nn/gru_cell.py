import numpy as np
from nn.activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, input_size, hidden_size):
        self.d = input_size
        self.h = hidden_size
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.brx = np.random.randn(h)
        self.bzx = np.random.randn(h)
        self.bnx = np.random.randn(h)

        self.brh = np.random.randn(h)
        self.bzh = np.random.randn(h)
        self.bnh = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbrx = np.zeros((h))
        self.dbzx = np.zeros((h))
        self.dbnx = np.zeros((h))

        self.dbrh = np.zeros((h))
        self.dbzh = np.zeros((h))
        self.dbnh = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx
        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h_prev_t: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h_prev_t
        
        self.r = self.r_act.forward(np.dot(self.Wrx, self.x) + self.brx + np.dot(self.Wrh, h_prev_t) + self.brh)
        self.z = self.z_act.forward(np.dot(self.Wzx, self.x) + self.bzx + np.dot(self.Wzh, h_prev_t) + self.bzh)
        self.n = self.h_act.forward(np.dot(self.Wnx, self.x) + self.bnx + self.r * (np.dot(self.Wnh, h_prev_t) + self.bnh))
        self.h_t = (1 - self.z) * (self.n) + (self.z * h_prev_t)
        
        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert self.h_t.shape == (self.h,) # h_t is the final output of you GRU cell.

        return self.h_t

    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (input_dim)
            derivative of the loss wrt the input x.

        dh_prev_t: (hidden_dim)
            derivative of the loss wrt the input hidden h.

        """
        da_dr = self.r_act.backward(1)       #(hidden,)
        da_dz = self.z_act.backward(1)       #(hidden,)
        da_dn = self.h_act.backward(1, state=self.n)       #(hidden,)
        # NOTES:
        # 1) Make sure the shapes of the calculated dWs and dbs match the initalized shapes of the respective Ws and bs
        # 2) When in doubt about shapes, please refer to the table in the writeup.
        # 3) Know that the autograder grades the gradients in a certain order, and the local autograder will tell you which gradient you are currently failing.
        
        #1. Forward Eqn: ht = (1 − zt) ⊙ nt + zt ⊙ ht−1
        dLdZ = delta * (-self.n + self.hidden) #dLdZ = dLdh * dhdZ    #(batch, hidden)
        dLdn = delta * (1 - self.z)            #dLdn = dLdh * dhdn    #(batch, hidden)
        dh_prev_t = delta * self.z             #dLdht-1 = dL * dhdht-1#(batch, hidden) 

        #2. Forward Eqn: nt = tanh(Wnx · xt + bnx + rt ⊙ (Wnh · ht−1 + bnh))
        self.dWnx += (dLdn * da_dn).reshape(-1,1) @ self.x.reshape(1,-1)    #dLdWnx = dLdn * dndWnx               #(hidden, input)
        self.dbnx += (dLdn * da_dn).reshape(-1)                             #dLdbnx = dLdn * dndbnx               #(hidden, )  
        self.dWnh += ((dLdn * da_dn) * self.r).reshape(-1,1) * self.hidden.reshape(1,-1)#dLdWnh = dLdn * dndWnh               #(hidden, hidden)  
        self.dbnh += ((dLdn * da_dn) * self.r).reshape(-1)                  #dLdbnh = dLdnt * dndbnh              #(hidden, )  
        dLdrt = ((dLdn * da_dn) * (self.Wnh @ self.hidden + self.bnh) )     #dLdrt = dLdn * dndrt         #(batch, hidden)
        dh_prev_t += ((dLdn * da_dn) * self.r) @ self.Wnh                   #dLdh_prev_t = ... (dLdrt * drtdht-1) #(batch, hidden)
        dx = (dLdn * da_dn) @ self.Wnx                                      #dLdxt = ... (dLdn * dndxt)           #(batch, input)

        #3. Forward Eqn: zt = σ(Wzx · xt + bzx + Wzh · ht−1 + bzh)
        self.dWzx += (dLdZ * da_dz).reshape(-1,1) @ self.x.reshape(1,-1)        #dLdWzx = dLdZ * dZdWzx
        self.dbzx += (dLdZ * da_dz).reshape(-1)                                 #dLdbzx = dLdZ * dZdbzx
        self.dWzh += (dLdZ * da_dz).reshape(-1,1) @ self.hidden.reshape(1,-1)   #dLdWzh = dLdZ * dZdWzh
        self.dbzh += (dLdZ * da_dz).reshape(-1)                                 #dLdbzh = dLdZ * dZdbzh 
        dh_prev_t += (dLdZ * da_dz) @ self.Wzh                                  #dLdh_prev_t = ... (dLdZ * dZdht-1)
        dx += (dLdZ * da_dz) @ self.Wzx                                         #dLdxt = ... (dLdz * dzdxt)
        

        # 4. Forward Eqn: rt = σ(Wrx · xt + brx + Wrh · ht−1 + brh)
        self.dWrx += (dLdrt * da_dr).reshape(-1,1) @ self.x.reshape(1,-1)     #dLdWrx = dLdrt * drdWrx
        self.dbrx += (dLdrt * da_dr).reshape(-1)                              #dLdbrx = dLdrt * drdbrx
        self.dWrh += (dLdrt * da_dr).reshape(-1,1) @ self.hidden.reshape(1,-1)#dLdWrh = dLdrt * drdWrh
        self.dbrh += (dLdrt * da_dr).reshape(-1)                              #dLdbrh = dLdrt * drdbrh
        dh_prev_t += (dLdrt * da_dr) @ self.Wrh                               #dLdh_prev_t = ... (dLdr * dZdrt)
        dx += (dLdrt * da_dr) @ self.Wrx                                      #dLdxt = ... (dLdr * drdxt)

        assert dx.shape == (self.d,)
        assert dh_prev_t.shape == (self.h,)

        return dx, dh_prev_t
        

