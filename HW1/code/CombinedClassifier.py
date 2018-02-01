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

class NaiveBayesClassifier:
    def __init__(self, alpha, beta, n_features):
        # 1 x C vector; dirichlet prior for class distr.
        # C = 2 for binary classification.
        self.alpha = alpha
        self.alpha0 = sum(alpha)

        # 1 x K vector; dirichlet prior for class conditional distr.
        # K = 2 for binary features, otherwise K = max(occurences_of_word_in_text)
        self.beta = beta
        self.beta0 = sum(beta)

        # dimensions of data
        self.C = len(self.alpha) # num classes
        self.K = len(self.beta)  # num possible values for each feature (count)
        self.D = n_features      # num features (size of vocabulary)

        # counts
        self.N = 0
        self.N_c = np.zeros(self.C, dtype=int)
        self.N_cj = np.zeros((self.C, self.D), dtype=int)
        self.N_ckj = np.zeros((self.C, self.K, self.D), dtype=int)

        self.flushed = False

    def fit(self, X, y):
        X = X.astype(int)
        N, _D = X.shape
        self.N += N

        # print("Fitting model")
        for c in range(self.C):
            msk = y == c
            self.N_c[c] += np.sum(msk)
            self.N_cj[c] += np.sum(X[msk], dtype=int, axis=0)
            self.N_ckj[c] += np.apply_along_axis(np.bincount, 0, X[msk], minlength=self.K)

        self.flushed = False

    def predict(self, X):
        X = X.astype(int)

        if not self.flushed:
            # print("Flushing")
            self.pi = np.array([ # class distribution
                np.log(self.N_c[c] + self.alpha[c]) - np.log(self.N + self.alpha0)
                for c in range(self.C)])
            self.mu = np.fromfunction( # log prob of each (class, count, word) tuple
                lambda c, j, k: np.log(self.N_ckj[c, k, j] + self.beta[c]) - np.log(self.N_c[c] + self.beta0),
                (self.C, self.D, self.K), dtype=int)
            self.flushed = True

        # print("Predicting labels")
        p_for_x = lambda x: [ # calculate log probability for x of class c
            self.pi[c] + np.sum([self.mu[c, j, x[j]] for j in range(len(x))])
            for c in range(self.C)]
        ps = np.apply_along_axis(p_for_x, 1, X)
        return ps # get predictions

def bag_of_words(batch, TEXT):
    """
    returns bag of words representation (Variable) of a batch.
    each bag of words has dimension [batch_size, vocab_size].
    """
    V = len(TEXT.vocab)
    X = torch.zeros(batch.text.size(0), V)
    ones = torch.ones(batch.text.size(1))
    for b in range(batch.text.size(0)):
        X[b].index_add_(0, batch.text.data[b], ones)
        X[b][TEXT.vocab.stoi['<pad>']] = 0
    X = Variable(X, requires_grad=False)
    return X

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# Train multivariate Naive Bayes
alpha = a * np.ones(C)
beta = b * np.ones(K)
n_features = len(TEXT.vocab)
nb = NaiveBayesClassifier(alpha, beta, n_features)

for i, batch in enumerate(train_iter):
    batch.text = batch.text.transpose(1, 0)
    X = bag_of_words(batch, TEXT).data.numpy()
    if binary:
        X = X > 0
    y = batch.label.data.numpy() - 1
    nb.fit(X, y)
    
# Train convolutional neural net
vecs = Variable(TEXT.vocab.vectors, requires_grad=True)
cn = ConvNetClassifier(vecs)
loss_function = nn.NLLLoss()
optimizer = optim.Adam(cn.parameters(), lr=0.0003)
optimizer2 = optim.Adam([cn.vecs], lr=0.0001)
#optimizer = optim.SGD(cn.parameters(), lr=0.03, weight_decay=0.01)
#optimizer = optim.Adadelta(cn.parameters(), lr=0.1)
max_vec_size = 3

for i in range(20):
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
cn.linear.weight.data *= 0.5

# Evaluate model
n, n_corr = 0, 0
for i, batch in enumerate(test_iter):
    probs = cn(batch.text).data.numpy()
    y_pred = probs.argmax(1)
    batch.text = batch.text.transpose(1, 0)
    X = bag_of_words(batch, TEXT).data.numpy()
    if binary:
        X = X > 0
    probs2 = softmax(nb.predict(X))
    y_pred = (probs + probs2).argmax(1)
    y = batch.label.data.numpy() - 1

    n += len(y)
    n_corr += sum(y_pred == y)
    
print(n_corr / n)