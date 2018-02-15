import numpy as np
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
torch.set_printoptions(precision=4)


class LinearInterpTrigram(nn.Module):
    def __init__(self, V, n=3, alpha=[0.3, 0.5, 0.2]):
        super(LinearInterpTrigram, self).__init__()
        self.V = V # vocabulary size
        self.n = n # max n-gram size

        assert(n == len(alpha))
        self.is_normalized = False
        self.counts = [dict() for _ in range(n)] # counts keyed by tuples of word indices
        self.alpha = alpha                       # weights for n-gram probabilities

    def init_counts(self, n, normalize=False):
        init = 1 / self.V if normalize else (0 if n > 0 else 1)
        counts = init * torch.ones(self.V)
        return counts

    def normalize_counts(self):
        for n in range(self.n):
            for context in self.counts[n]:
                self.counts[n][context] /= self.counts[n][context].sum()
        self.is_normalized = True

    def batch_to_ngrams(self, batch, n, trim=True):
        '''
        takes in vector (batch)
        when trim=False, returns 1 extra ngram than target --
        the last ngram has no corresponding target in this batch.
        '''
        assert(0 <= n < self.n)
        if n == 0:
            return [()] * len(batch), batch.tolist()
        ngrams, targets = batch.unfold(0, n, 1), batch[n:]
        if trim:
            ngrams = ngrams[:-1]
        return ngrams.long().squeeze().tolist(), targets.long().tolist()

    def get_counts(self, batch, TEXT):
        batch_size = len(batch)

        for i in range(batch_size):
            fragment = batch[i].data
            # if i == 0:
            #     print(' '.join([TEXT.vocab.itos[j] for j in fragment]))
            for n in range(self.n):
                contexts, targets = self.batch_to_ngrams(fragment, n, trim=True)
                # if i==0 and n==1:
                #     print(' '.join([TEXT.vocab.itos[j] for j in contexts]))
                #     print(' '.join([TEXT.vocab.itos[j] for j in targets]))
                for (context, target) in zip(contexts, targets):
                    key = tuple(context) if n > 1 else context
                    c = self.counts[n].get(key, self.init_counts(n, normalize=False))
                    c[target] += 1
                    self.counts[n][key] = c

    def forward(self, batch, TEXT):
        if not self.is_normalized:
            self.normalize_counts()

        batch_size, bptt_len = batch.size()
        n_preds = bptt_len - (self.n - 1) + 1
        outputs = torch.zeros(batch_size, n_preds, self.V).float()
        targets = torch.zeros(batch_size, n_preds-1).long()

        for i in range(batch_size):
            fragment = batch[i].data
            # if i == 0:
            #     print(' '.join([TEXT.vocab.itos[j] for j in fragment]))
            for n in range(self.n):
                ngrams = self.batch_to_ngrams(fragment, n, trim=False)
                # if i==0 and n==1:
                #     print(' '.join([TEXT.vocab.itos[j] for j in ngrams[0]]))
                #     print(' '.join([TEXT.vocab.itos[j] for j in ngrams[1]]))
                for j, (context, target) in enumerate(zip(*ngrams)):
                    key = tuple(context) if n > 1 else context
                    c = self.counts[n].get(key, self.init_counts(n, normalize=True))
                    # print(key, self.alpha[n], c)
                    if j < n_preds:
                        outputs[i, j] += self.alpha[n] * c
                    if j < n_preds - 1:
                        targets[i, j] = target

        return Variable(outputs), Variable(targets)
