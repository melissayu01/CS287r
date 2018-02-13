import torchtext
from torchtext.vocab import Vectors

def load_PTB(dev, use_pretrained_embeddings, batch_size, bptt_len,
             repeat=False, shuffle=False):
    """
    returns PTB data batch iterators and text field.
    each batch has dimensions [batch_size, bptt_len]
    """
    print('Loading PTB data...')

    TEXT = torchtext.data.Field()

    train, val, test = torchtext.datasets.LanguageModelingDataset.splits(
        path="../data/", text_field=TEXT,
        train=("train.5k.txt" if dev else "train.txt"),
        validation="valid.txt", test="valid.txt")
    train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
        (train, val, test), bptt_len=bptt_len, batch_size=batch_size,
        shuffle=shuffle, repeat=repeat, device=-1)

    TEXT.build_vocab(train, max_size=1000 if dev else None)
    if use_pretrained_embeddings:
        url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
        TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url))

    print('len(TEXT.vocab) = {}'.format(len(TEXT.vocab)))
    print('Size of text batch [max bptt length, batch size] = {}, {}'.format(bptt_len, batch_size))

    return train_iter, val_iter, test_iter, TEXT
