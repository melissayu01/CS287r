import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext
from torchtext.vocab import Vectors
from collections import deque
from torch.nn.utils.clip_grad import clip_grad_norm
import collections
import time

class LSTM_Lang_Model(nn.Module):
    
    def __init__(self, vocab_size, embedding_size, dropout_rate, hid_size):
        super(LSTM_Lang_Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.dropout_rate = dropout_rate
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hid_size, num_layers=2, dropout=dropout_rate)
        for w in self.lstm.parameters():
            w.data = torch.Tensor(w.size()).uniform_(-0.04, 0.04)
        self.dropout_o = nn.Dropout(p=dropout_rate)
        self.linear = nn.Linear(hid_size, vocab_size)
    
    def forward(self, text, hidden):
        input = self.embedding(text)
        output, hidden = self.lstm(input, hidden)
        output = self.dropout_o(output)
        output = self.linear(output)
        probs = F.softmax(output, dim=2)
        return probs, hidden


# Calculates perplexity
def eval(model, data):
    data.init_epoch()
    model.eval()
    total_n = torch.FloatTensor([0]).cuda()
    total_loss = torch.FloatTensor([0]).cuda()
    hidden = None
    for batch in data:
        batch.text = batch.text.cuda()
        probs, hidden = rnn(batch.text, hidden)
        n, m = batch.text.size()
        total_n += n * m
        log_probs = torch.log(probs)
        loss = -log_probs.gather(2, batch.target.unsqueeze(2).cuda()).sum()
        total_loss += loss.data
    ppl = np.exp(total_loss / total_n)
    model.train()
    return ppl

# Decrease learning rate by factor over time
def decrease_learning_rate(optimizer, factor):
    for param_group in optimizer.param_groups:
        param_group['lr'] /= factor