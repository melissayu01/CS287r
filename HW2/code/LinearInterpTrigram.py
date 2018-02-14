import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
torch.set_printoptions(precision=4)


class LinearInterpTrigram(nn.Module):
    def __init__(self, V):
        super(LinearInterpTrigram, self).__init__()
        self.V = V

        self.unigrams = self.init_counts(V, 1)
        self.bigrams  = self.init_counts(V, V)
        self.trigrams = dict()

        self.w = nn.Linear(4, 1, bias=False)

    def init_counts(self, *size):
        counts = nn.Embedding(*size)
        counts.weight.data = torch.zeros(counts.weight.data.size())
        return counts

    def batch_to_ngrams(self, batch, n, trim=True):
        '''
        NOTE: returns 1 extra ngram than target -- the last ngram
        has no corresponding target in this batch.
        '''
        ngrams, targets = batch.unfold(0, n-1, 1), batch[n-1:]
        if trim:
            ngrams = ngrams[:-1]
        return ngrams.long().squeeze(), targets.long().unsqueeze_(-1)

    def forward(self, batch, estimate_weights=False):
        batch_size = len(batch)
        words = batch.data

        if self.training and not estimate_weights:
            # update counts
            ones = torch.ones(batch_size, 1)
            self.unigrams.weight.data.index_add_(0, words, ones)

            bigrams, bigram_targets = self.batch_to_ngrams(words, 2, trim=True)
            ones = torch.zeros(batch_size-1, self.V).scatter_(1, bigram_targets, 1)
            self.bigrams.weight.data.index_add_(0, bigrams, ones)

            trigrams, trigram_targets = self.batch_to_ngrams(words, 3, trim=True)
            ones = torch.zeros(batch_size-2, self.V).scatter_(1, trigram_targets, 1)
            for (pair, target) in zip(trigrams, trigram_targets):
                tmp = self.trigrams.get(pair, self.init_counts(self.V, 1))
                tmp.weight.data[target] += 1
                self.trigrams[pair] = tmp

        elif self.eval or estimate_weights:
            # compute predictions
            trigrams, trigram_targets = self.batch_to_ngrams(words, 3, trim=estimate_weights)
            n_predictions = len(trigrams)

            bigram_indices = Variable(trigrams[:, 1], requires_grad=False)
            zero_back = self.unigrams.weight.data
            one_back = self.bigrams(bigram_indices).data
            two_back = torch.stack([
                self.trigrams.get(pair, self.init_counts(self.V, 1)).weight.data
                for pair in torch.unbind(trigrams, dim=0)
            ], dim=0).squeeze()

            p = Variable(torch.zeros(n_predictions, self.V, 4), requires_grad=False)
            p[:, :, 0] = 1 / self.V
            for i in range(n_predictions):
                p[i, :, 1] = zero_back / zero_back.sum()
            p[:, :, 2] = one_back / one_back.sum()
            p[:, :, 3] = two_back / two_back.sum()

            targets = Variable(trigram_targets, requires_grad=False)

            return self.w(p).squeeze(), targets.squeeze()
