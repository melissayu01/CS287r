import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
torch.set_printoptions(precision=4)


class NNLM(nn.Module):
    def __init__(self, n, V, m, h, max_norm, embeddings):
        super(NNLM, self).__init__()
        self.n = n
        self.V = V
        self.m = m
        self.h = h

        self.embedding = nn.Embedding(V, m, max_norm=max_norm)
        if embeddings:
            self.embedding.weight = nn.Parameter(embeddings)
        self.skip_connection = nn.Linear((n-1) * m, V, bias=False)
        self.pass_through = nn.Sequential(
            nn.Linear((n-1) * m, h),
            nn.Tanh(),
            nn.Linear(h, V),
        )

    def forward(self, batch):
        batch_size = len(batch)
        x = self.embedding(batch).view(batch_size, -1) # [n_ngrams_in_minibatch, (n - 1) * m]
        out = self.skip_connection(x) + self.pass_through(x)
        return out
