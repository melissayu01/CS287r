import torch
from torch.autograd import Variable
import torchtext
from torchtext.vocab import Vectors, GloVe

DATA_DIR = '../data/'

def load_PTB(dev, use_pretrained_embeddings, batch_size, bptt_len,
             repeat=False, shuffle=False):
    """
    returns PTB data batch iterators and text field.
    each batch has dimensions [batch_size, bptt_len]
    """
    print('Loading PTB data...')

    TEXT = torchtext.data.Field()

    train, val, test = torchtext.datasets.LanguageModelingDataset.splits(
        path=DATA_DIR, text_field=TEXT,
        train=("train.5k.txt" if dev else "train.txt"),
        validation="valid.txt", test="valid.txt")
    train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
        (train, val, test), bptt_len=bptt_len, batch_size=batch_size,
        shuffle=shuffle, repeat=repeat, device=-1)

    TEXT.build_vocab(train, max_size=1000 if dev else None)
    if use_pretrained_embeddings:
        TEXT.vocab.load_vectors(vectors=GloVe(name='6B'))
        # url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
        # TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url))

    print('len(TEXT.vocab) = {}'.format(len(TEXT.vocab)))
    print('Size of text batch [max bptt length, batch size] = {}, {}'.format(bptt_len, batch_size))

    return train_iter, val_iter, test_iter, TEXT

def load_kaggle(TEXT):
    '''
    returns list of Variables representing sentence fragments
    '''
    print('Loading Kaggle data...')

    out = []
    for line in open(DATA_DIR + "input.txt"):
        words = line.strip(' _\t\n\r').split()
        word_indexes = [TEXT.vocab.stoi[word] for word in words]
        out.append(Variable(torch.LongTensor(word_indexes), requires_grad=False))

    print('len(kaggle) = {}'.format(len(out)))
    return out
