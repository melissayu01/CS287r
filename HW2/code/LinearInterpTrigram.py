import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
torch.set_printoptions(precision=4)


class LinearInterpTrigram(nn.Module):
    def __init__(self, V):
        super(LinearInterpTrigram, self).__init__()

        self.unigrams = nn.Parameter(torch.zeros(V).int())
        self.bigrams  = nn.Parameter(torch.zeros(V, V).int())
        self.trigrams = nn.Parameter(torch.zeros(V, V, V).int())

        self.w = nn.Linear(4, 1, bias=False)

    def discrete_context_freq(self, context):
        '''
        bins context frequency; used to estimate weights.
        [currently unused]
        '''
        x = self.trigram[context[0], context[1]].sum().data
        T = self.unigram.sum().data
        return int( math.ceil(-math.log((1 + x[0]) / T[0])) )

    def batch_to_ngrams(self, batch, n):
        '''
        NOTE: returns 1 extra ngram than target -- the last ngram
        has no corresponding target in this batch.
        '''
        batch = batch.data
        ngrams, targets = batch.unfold(0, n-1, 1), batch[n-1:]
        for _ in range(n-1):
            targets.unsqueeze_(-1)
        return ngrams.long(), targets.long()

    def forward(self, batch, estimate_weights=False):
        batch_size = len(batch)
        if self.training and not estimate_weights:
            # update counts
            ones = torch.ones(batch_size)
            self.unigrams.index_add_(0, batch, ones)

            bigrams, bigram_targets = self.batch_to_ngrams(batch, 2)
            bigrams = bigrams[:-1]
            ones = torch.zeros(len(bigrams), self.V).scatter_(1, bigram_targets, 1)
            self.bigrams.index_add_(0, bigrams, ones)

            trigrams, trigram_targets = self.batch_to_ngrams(batch, 3)
            trigrams = trigrams[:-1]
            ones = torch.zeros(len(trigrams), self.V).scatter_(1, trigram_targets, 1)
            for i, context in enumerate(trigrams.tolist()):
                self.trigrams[context[0], context[1]].add_(ones[i])

        else:
            # compute predictions
            trigrams, trigram_targets = self.batch_to_ngrams(batch, 3)

            q = torch.zeros(len(trigrams)).int()
            for i, context in enumerate(trigrams.tolist()):
                q[i] = self.discrete_context_freq(context)

            zero_back = self.unigrams
            one_back = self.bigrams.index_select(0, trigrams[:, 1])
            two_back = self.trigrams.index_select(0, trigrams[:, 0]).index_select(0, trigrams[:, 1])

            if estimate_weights:
                p = torch.zeros(len(trigram_targets), 4)
                p[:, 0] = 1 / self.V
                p[:, 1] = zero_back.index_select(0, trigram_targets) / zero_back.sum()
                p[:, 2] = one_back.index_select(0, trigram_targets) / one_back.sum()
                p[:, 3] = two_back.index_select(0, trigram_targets) / two_back.sum()
            else:
                p = torch.zeros(len(trigram_targets)+1, self.V, 4)
                p[:, :, 0] = 1 / self.V
                p[:, :, 1] = zero_back / zero_back.sum()
                p[:, :, 2] = one_back / one_back.sum()
                p[:, :, 3] = two_back / two_back.sum()

            return self.w(p).squeeze()
