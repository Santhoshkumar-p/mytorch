import numpy as np


class CTC(object):

    def __init__(self, BLANK=0):
        """

        Initialize instance variables

        Argument(s)
        -----------

        BLANK (int, optional): blank label index. Default 0.

        """

        # No need to modify
        self.BLANK = BLANK


    def extend_target_with_blank(self, target):
        """Extend target sequence with blank.

        Input
        -----
        target: (np.array, dim = (target_len,))
                target output containing indexes of target phonemes
        ex: [1,4,4,7]

        Return
        ------
        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended target sequence with blanks
        ex: [0,1,0,4,0,4,0,7,0]

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections
        ex: [0,0,0,1,0,0,0,1,0]
        """

        extended_symbols = [self.BLANK]
        for symbol in target:
            extended_symbols.append(symbol)
            extended_symbols.append(self.BLANK)

        N = len(extended_symbols)
        skip_connect = [0] * len(extended_symbols)
        # -------------------------------------------->
        for i, symbol in enumerate(extended_symbols):
            skip_idx = i-2
            if skip_idx >= 0 and symbol != extended_symbols[skip_idx]:
                skip_connect[i] = 1
        # <---------------------------------------------

        extended_symbols = np.array(extended_symbols).reshape((N,))
        skip_connect = np.array(skip_connect).reshape((N,))

        return extended_symbols, skip_connect


    def get_forward_probs(self, logits, extended_symbols, skip_connect):
        """Compute forward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(Symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t, qextSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probabilities

        """

        S, T = len(extended_symbols), len(logits)
        alpha = np.zeros(shape=(T, S))

        # -------------------------------------------->
        # Intialize alpha[0][0]
        alpha[0][0] = logits[0][extended_symbols[0]]
        # Intialize alpha[0][1]
        alpha[0][1] = logits[0][extended_symbols[1]]
        # Compute all values for alpha[t][sym] where 1 <= t < T and 1 <= sym < S (assuming zero-indexing)
        for t in range(1, T):
            alpha[t][0] = alpha[t-1][0] * logits[t][extended_symbols[0]]
            for sym in range(1, S):
                alpha[t][sym] = alpha[t-1][sym-1] + alpha[t-1][sym]
                if skip_connect[sym] == 1:
                    alpha[t][sym] = alpha[t][sym] + alpha[t-1][sym-2]
                alpha[t][sym] = alpha[t][sym] * logits[t][extended_symbols[sym]]
        # IMP: Remember to check for skipConnect when calculating alpha
        # <---------------------------------------------
        return alpha


    def get_backward_probs(self, logits, extended_symbols, skip_connect):
        """Compute backward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t,extSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probabilities
    
        """
        
        S, T = len(extended_symbols), len(logits)
        beta = np.zeros(shape=(T, S))

        # -------------------------------------------->
        beta[T - 1][S - 1] = 1
        beta[T - 1][S - 2] = 1

        for t in reversed(range(T-1)):
            beta[t, S-1] = beta[t+1, S-1] * logits[t+1, extended_symbols[S-1]]
            for sym in reversed(range(S-1)):
                beta[t, sym] = beta[t+1, sym] * logits[t+1, extended_symbols[sym]] \
                                + beta[t+1, sym+1] * logits[t+1, extended_symbols[sym+1]]
                if (sym < S-3) and (skip_connect[sym + 2]):
                    beta[t, sym] = beta[t, sym] + beta[t + 1, sym + 2] * logits[t + 1, extended_symbols[sym + 2]]
        # <--------------------------------------------

        return beta


    def get_posterior_probs(self, alpha, beta):
        """Compute posterior probabilities.

        Input
        -----
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probability

        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probability

        Return
        ------
        gamma: (np.array, dim = (input_len, 2 * target_len + 1))
                posterior probability

        """

        [T, S] = alpha.shape
        gamma = np.zeros(shape=(T, S))
        sumgamma = np.zeros((T,))
        
        # -------------------------------------------->
        # Compute gamma
        for t in range(T):
            for sym in range(S):
                gamma[t][sym] = alpha[t][sym] * beta[t][sym]
                sumgamma[t] += gamma[t][sym]

        # Normalize gamma
        for t in range(T):
            if sumgamma[t] != 0:
                gamma[t] /= sumgamma[t]

        # <---------------------------------------------

        return gamma


class CTCLoss(object):

    def __init__(self, BLANK=0):
        """

        Initialize instance variables

        Argument(s)
        -----------
        BLANK (int, optional): blank label index. Default 0.
        
        """
        super(CTCLoss, self).__init__()

        self.BLANK = BLANK
        self.gammas = []
        self.ctc = CTC()

    def __call__(self, logits, target, input_lengths, target_lengths):

        return self.forward(logits, target, input_lengths, target_lengths)

    def forward(self, logits, target, input_lengths, target_lengths):
        """CTC loss forward

        Computes the CTC Loss by calculating forward, backward, and
        posterior proabilites, and then calculating the avg. loss between
        targets and predicted log probabilities

        Input
        -----
        logits [np.array, dim=(seq_length, batch_size, len(symbols)]:
            log probabilities (output sequence) from the RNN/GRU

        target [np.array, dim=(batch_size, padded_target_len)]:
            target sequences

        input_lengths [np.array, dim=(batch_size,)]:
            lengths of the inputs

        target_lengths [np.array, dim=(batch_size,)]:
            lengths of the target

        Returns
        -------
        loss [float]:
            avg. divergence between the posterior probability and the target

        """

        # No need to modify
        self.logits = logits
        self.target = target
        self.input_lengths = input_lengths
        self.target_lengths = target_lengths

        #####  IMP:
        #####  Output losses should be the mean loss over the batch

        # No need to modify
        B, _ = target.shape
        total_loss = np.zeros(B)
        self.extended_symbols = []

        for batch_itr in range(B):
            # -------------------------------------------->
            # Computing CTC Loss for single batch
            # Process:
            #     Truncate the target to target length
            #     Truncate the logits to input length
            #     Extend target sequence with blank
            #     Compute forward probabilities
            #     Compute backward probabilities
            #     Compute posteriors using total probability function
            #     Compute expected divergence for each batch and store it in totalLoss
            #     Take an average over all batches and return final result
            # <---------------------------------------------

            # -------------------------------------------->
            target_trunc = target[batch_itr, :target_lengths[batch_itr]]
            logits_trunc = logits[:input_lengths[batch_itr], batch_itr, :]
            extended_symbols, skip_connect = self.ctc.extend_target_with_blank(target_trunc)
            fwd_probs = self.ctc.get_forward_probs(logits=logits_trunc, skip_connect=skip_connect, extended_symbols=extended_symbols)
            bwd_probs = self.ctc.get_backward_probs(logits=logits_trunc, skip_connect=skip_connect, extended_symbols=extended_symbols)
            posterior_probs = self.ctc.get_posterior_probs(alpha=fwd_probs, beta=bwd_probs)
            self.extended_symbols.append(extended_symbols)
            self.gammas.append(posterior_probs)
            # Compute the loss for each symbol
            for sym in range(len(extended_symbols)):
                 # Compute the loss for each symbol - np.log(logits_trunc[:, extended_symbols[sym]])
                total_loss[batch_itr] = total_loss[batch_itr] - np.sum(posterior_probs[:, sym] * np.log(logits_trunc[:, extended_symbols[sym]])) 

            # <---------------------------------------------
            

        total_loss = np.sum(total_loss) / B

        return total_loss


    def backward(self):
        """

        CTC loss backard

        Calculate the gradients w.r.t the parameters and return the derivative 
        w.r.t the inputs, xt and ht, to the cell.

        Input
        -----
        logits [np.array, dim=(seqlength, batch_size, len(Symbols)]:
            log probabilities (output sequence) from the RNN/GRU

        target [np.array, dim=(batch_size, padded_target_len)]:
            target sequences

        input_lengths [np.array, dim=(batch_size,)]:
            lengths of the inputs

        target_lengths [np.array, dim=(batch_size,)]:
            lengths of the target

        Returns
        -------
        dY [np.array, dim=(seq_length, batch_size, len(extended_symbols))]:
            derivative of divergence w.r.t the input symbols at each time

        """

        # No need to modify
        T, B, C = self.logits.shape
        dY = np.full_like(self.logits, 0)

        for batch_itr in range(B):
            # -------------------------------------------->
            # Computing CTC Derivative for single batch
            # Process:
            #     Truncate the target to target length
            #     Truncate the logits to input length
            #     Extend target sequence with blank
            #     Compute derivative of divergence and store them in dY
            # <---------------------------------------------

            # -------------------------------------------->
            target_trunc = self.target[batch_itr, :self.target_lengths[batch_itr]]
            logits_trunc = self.logits[:self.input_lengths[batch_itr], batch_itr, :]
            extended_symbols, skip_connect = self.ctc.extend_target_with_blank(target_trunc)
            for sym in range(len(extended_symbols)):
                dY[:self.input_lengths[batch_itr], batch_itr, extended_symbols[sym]] -= self.gammas[batch_itr][:, sym] / logits_trunc[:, extended_symbols[sym]]
            # <---------------------------------------------

        return dY
