import numpy as np

class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """
        
        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """

        self.symbol_set = symbol_set


    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """

        decoded_path = []
        blank = 0
        path_prob = 1
        sym_length, seq_length, batch_size = y_probs.shape
        # STEPS:
        # 1. Iterate over sequence length - len(y_probs[0])
        # 2. Iterate over symbol probabilities
        # 3. update path probability, by multiplying with the current max probability
        # 4. Select most probable symbol and append to decoded_path
        # 5. Compress sequence (Inside or outside the loop)

        for batch_itr in range(batch_size):
            for seq in range(seq_length):
                max_prob_idx = np.argmax(y_probs[:, seq, batch_itr])
                path_prob = path_prob * y_probs[max_prob_idx, seq, batch_itr]
                if max_prob_idx != blank and (not decoded_path or max_prob_idx != decoded_path[-1]):
                    decoded_path.append(max_prob_idx)
        
        # # Convert indices to symbols
        decoded_path = [self.symbol_set[i - 1] for i in decoded_path]
        decoded_path = "".join(decoded_path)
        return decoded_path, path_prob


class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_width):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion

        """

        self.symbol_set = symbol_set
        self.beam_width = beam_width

    def init(self, y_probs):

        empty_path = [""]
        empty_score = {"":y_probs[0,0,0]}
        sym_path = []
        sym_score = {}

        for i, sym in enumerate(self.symbol_set):
            sym_path.append(sym)
            sym_score[sym] = y_probs[i + 1, 0, 0]
        
        return empty_path, empty_score, sym_path, sym_score

    def prune(self, blank_path, blank_score, sym_path, sym_score):
        scores_list = []
        for value in blank_score.values():
            scores_list.append(value)

        for value in sym_score.values():
            scores_list.append(value)

        scores_list.sort()

        if len(scores_list) < self.beam_width:
            cutoff = scores_list[-1]
        else:
            cutoff = scores_list[-self.beam_width]

        pruned_blank_path = []
        pruned_blank_score = {}
        pruned_sym_path = []
        pruned_sym_score = {}

        for p in blank_path:
            if blank_score[p] >= cutoff:
                pruned_blank_path.append(p)
                pruned_blank_score[p] = blank_score[p]

        for p in sym_path:
            if sym_score[p] >= cutoff:
                pruned_sym_path.append(p)
                pruned_sym_score[p] = sym_score[p]
        
        return pruned_blank_path, pruned_blank_score, pruned_sym_path, pruned_sym_score

    def extend_with_symbols(self, y_probs, blank_path, blank_score, sym_path, sym_score, seq):
        
        extended_symbols_path = []
        extended_symbols_score = {}
        for path in blank_path:
            for idx, char in enumerate(self.symbol_set):
                new_path = path + char
                extended_symbols_path.append(new_path)
                extended_symbols_score[new_path] = blank_score[path] * y_probs[idx+1, seq, 0]
                
        for path in sym_path:
            for idx, char in enumerate(self.symbol_set):
                new_path = path if char == path[-1] else path + char
                if new_path in extended_symbols_path:
                    extended_symbols_score[new_path] += sym_score[path] * y_probs[idx+1, seq, 0]
                else:
                    extended_symbols_path.append(new_path)
                    extended_symbols_score[new_path] = sym_score[path] * y_probs[idx+1, seq, 0]
        
        return extended_symbols_path, extended_symbols_score

    def extend_with_blanks(self, y_probs, blank_path, blank_score, sym_path, sym_score, seq):
        
        extended_blanks_path = []
        extended_blanks_score = {}

        for path in blank_path:
            extended_blanks_path.append(path)
            extended_blanks_score[path] = blank_score[path] * y_probs[0, seq, 0]
        
        for path in sym_path:
            if path in extended_blanks_path:
                extended_blanks_score[path] += sym_score[path] * y_probs[0, seq, 0]
            else:
                extended_blanks_path.append(path)
                extended_blanks_score[path] = sym_score[path] * y_probs[0, seq, 0]
        
        return extended_blanks_path, extended_blanks_score 
    
    def merge_path(self, blank_path, blank_score, sym_path, sym_score):

        paths = blank_path
        scores = blank_score
        
        for path in sym_path:
            if path in paths:
                scores[path] += sym_score[path]
            else:
                paths.append(path)
                scores[path] = sym_score[path]

        best_scores = dict(sorted(scores.items(), key=lambda x: x[1]))
        best_paths = list(best_scores.keys())[-1]

        return best_paths, best_scores
    
    def decode(self, y_probs):
        """
        
        Perform beam search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
			batch size for part 1 will remain 1, but if you plan to use your
			implementation for part 2 you need to incorporate batch_size

        Returns
        -------
        
        forward_path [str]:
            the symbol sequence with the best path score (forward probability)

        merged_path_scores [dict]:
            all the final merged paths with their scores

        """

        best_path, best_path_score = None, None
        seq_length = y_probs.shape[1]

        blank_path, blank_score, symbol_path, symbol_score = self.init(y_probs)

        for seq in range(1, seq_length):

            blank_path, blank_score, symbol_path, symbol_score = self.prune(blank_path, blank_score, symbol_path, symbol_score)
        
            ext_blank_path, ext_blank_score = self.extend_with_blanks(y_probs, blank_path, blank_score, symbol_path, symbol_score, seq)

            extended_symbol_path, extended_symbol_score = self.extend_with_symbols(y_probs, blank_path, blank_score, symbol_path, symbol_score, seq)

            blank_path, blank_score, symbol_path, symbol_score = ext_blank_path, ext_blank_score, extended_symbol_path, extended_symbol_score

        best_path, best_path_score = self.merge_path(blank_path, blank_score, symbol_path, symbol_score)

        return best_path, best_path_score

