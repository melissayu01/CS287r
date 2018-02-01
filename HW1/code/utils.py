import torch
from torch.autograd import Variable
import torchtext
from torchtext.vocab import Vectors, GloVe

def load_SST(use_embeddings=False, batch_size=10, repeat=False, shuffle=False):
    """
    returns SST data batch iterators, text field, and label field.
    each batch has dimension [batch_size, max(sentence_len_in_batch)].
    NOTE: positive = 1, negative = 2
    """
    print('Loading SST data...')

    TEXT = torchtext.data.Field(batch_first=True)
    LABEL = torchtext.data.Field(sequential=False)

    train, val, test = torchtext.datasets.SST.splits(
        TEXT, LABEL,
        filter_pred=lambda ex: ex.label != 'neutral')

    train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits(
        (train, val, test),
        batch_size=batch_size, shuffle=shuffle, repeat=repeat, device=-1)
    test_iter = torchtext.data.BucketIterator(test, train=False, batch_size=10, device=-1)

    TEXT.build_vocab(train)
    LABEL.build_vocab(train)

    if use_embeddings:
        url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
        TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url))

    print('len(train) = {}, len(val) = {}, len(test) = {}'.format(
        len(train), len(val), len(test)))
    print('len(TEXT.vocab) = {}, len(LABEL.vocab) = {}'.format(
        len(TEXT.vocab), len(LABEL.vocab)))

    return train_iter, val_iter, test_iter, TEXT, LABEL

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

def skip_gram_embeddings(batch, TEXT):
    """
    returns skip-gram representation (Variable) of a batch.
    each bag of words has dimension [batch_size, skip_gram_size].
    """
    N = TEXT.vocab.vectors.size(1)
    X = torch.zeros(batch.text.size(0), N)
    for b in range(batch.text.size(0)):
        for word_idx in batch.text.data[b]:
            X[b] += TEXT.vocab.vectors[word_idx]
    X = Variable(X, requires_grad=False)
    return X