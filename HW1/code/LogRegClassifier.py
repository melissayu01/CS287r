import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext
from torchtext.vocab import Vectors, GloVe

TEXT = torchtext.data.Field()
LABEL = torchtext.data.Field(sequential=False)
train, val, test = torchtext.datasets.SST.splits(
    TEXT, LABEL,
    filter_pred=lambda ex: ex.label != 'neutral')

TEXT.build_vocab(train)
LABEL.build_vocab(train)
n_vocab = len(TEXT.vocab)
vecs = torch.eye(n_vocab)
vecs[:, 1] = 0 # ignore <pad>
TEXT.vocab.set_vectors(TEXT.vocab.stoi, vecs, n_vocab)

BATCH_SIZE = 100
train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits(
    (train, val, test), batch_size=BATCH_SIZE, device=-1, repeat=False)

class LogRegClassifier(nn.Module):
    
    def __init__(self, vocab_size):
        super(LogRegClassifier, self).__init__()
        self.linear = nn.Linear(vocab_size, 1)
    
    def forward(self, text):
        # Create design matrix
        vecs = []
        for b in range(text.size(1)):
            v = TEXT.vocab.vectors[text.data[:, b]].max(0)[0]
            vecs.append(v.view(1, -1))
        X = Variable(torch.cat(vecs))
        p = F.sigmoid(self.linear(X))
        return torch.cat([p, 1-p], 1)

def evaluate(model, data_iter):
    data_iter.init_epoch()
    N = len(data_iter.data())
    n_correct = 0
    data_iter.init_epoch()
    for batch in data_iter:
        probs = model(batch.text)
        _, y_predicted = probs.max(1)
        y_true = batch.label - 1
        n_correct += (y_true == y_predicted).sum().float()
    return (n_correct / N).data.numpy()[0]

acc = {}
regs = [10**-2, 10**-3, 10**-4, 10**-5, 10**-6]
for reg in regs:
    lr = LogRegClassifier(n_vocab)
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(lr.parameters(), lr=0.01, weight_decay=reg)

    for _ in range(10):
        train_iter.init_epoch()
        for batch in train_iter:
            lr.zero_grad()
            probs = lr(batch.text)
            log_probs = torch.log(probs)
            y = batch.label - 1
            loss = loss_function(log_probs, y)
            loss.backward()
            optimizer.step()
    print(reg)
    train_acc = evaluate(lr, train_iter)
    val_acc = evaluate(lr, val_iter)
    acc[reg] = {'train': train_acc, 'val': val_acc}