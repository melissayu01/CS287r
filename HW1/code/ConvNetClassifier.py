import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
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

url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url))
n_comps = TEXT.vocab.vectors.size(1)

BATCH_SIZE = 50
train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits(
    (train, val, test), batch_size=BATCH_SIZE, device=-1, repeat=False)


class ConvNetClassifier(nn.Module):
    
    def __init__(self, vecs, dropout_rate=0.5):
        super(ConvNetClassifier, self).__init__()
        self.vecs = vecs
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(3, n_comps))
        self.conv4 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(4, n_comps))
        self.conv5 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(5, n_comps))
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.linear = nn.Linear(300, 1)
    
    def forward(self, text, training=False):
        while text.size(0) < 5:
            text = torch.cat([text, torch.ones((1, text.size(1))).long()], 0)
        sent_length, batch_size = text.size()
        X = self.vecs[text.data.view(-1,)].view(sent_length, batch_size, n_comps)
        X = X.permute(1, 0, 2)
        X = X.data.unsqueeze_(1)
        X = Variable(X)
        
        # Extract and pool convolutional features
        X3 = F.relu(self.conv3(X))
        X3 = F.max_pool2d(X3, (X3.size(2), 1))
        X4 = F.relu(self.conv4(X))
        X4 = F.max_pool2d(X4, (X4.size(2), 1))
        X5 = F.relu(self.conv5(X))
        X5 = F.max_pool2d(X5, (X5.size(2), 1))
        
        # Dropout for regularization
        if training:
            X3 = self.dropout(X3)
            X4 = self.dropout(X4)
            X5 = self.dropout(X5) 
        
        # Final layer
        X = torch.cat([X3, X4, X5], 1).squeeze()
        probs = F.sigmoid(self.linear(X))
        return torch.cat([probs, 1-probs], 1)

vecs = Variable(TEXT.vocab.vectors, requires_grad=True)
cn = ConvNetClassifier(vecs)
loss_function = nn.NLLLoss()
optimizer = optim.Adam(cn.parameters(), lr=0.001)
optimizer2 = optim.Adam([cn.vecs], lr=0.0001)
#optimizer = optim.SGD(cn.parameters(), lr=0.03, weight_decay=0.01)
#optimizer = optim.Adadelta(cn.parameters(), lr=0.1)
#max_vec_size = 5

for i in range(100):
    train_iter.init_epoch()
    for batch in train_iter:
        cn.zero_grad()
        probs = cn(batch.text, training=True)
        log_probs = torch.log(probs)
        y = batch.label - 1
        loss = loss_function(log_probs, y)
        loss.backward()
        optimizer.step()
#         optimizer2.step()
        
        # Regularization
#         for w in cn.parameters():
#             w_2norm = w.data.norm(2)
#             if w_2norm > max_vec_size:
#                 w.data = max_vec_size / w_2norm * w.data
    print('Iteration #{}: {}'.format(i, loss.data.numpy()[0]))
#cn.linear.weight.data *= 0.5