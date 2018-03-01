import torch
from torch.autograd import Variable
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe
import spacy
import joblib
import matplotlib.pyplot as plt
from matplotlib import ticker


BOS_WORD = '<s>'
EOS_WORD = '</s>'
MAX_LEN = 20
MIN_FREQ = 5


def tokenize(text, spacy):
    return [tok.text for tok in spacy.tokenizer(text)]

def load_dataset_from_fields(DE, EN):
    # Loads German-to-English dataset
    train, val, test = datasets.IWSLT.splits(
        exts=('.de', '.en'), fields=(DE, EN),
        filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and
        len(vars(x)['trg']) <= MAX_LEN
    )
    return train, val, test

def load_dataset(batch_size, use_pretrained_emb=False, save_dir='.save'):
    '''
    Each batch has dimensions (seq_len, batch_size)
    Returns tuple of (train iterator, val iterator, test iterator, TRG, SRC).
    '''
    print('Loading German-English data...')

    fname = './{}/vocabs.jl'.format(save_dir)
    try:
        data_dict = joblib.load(fname)
        if ((MAX_LEN != data_dict['max_len']) or (MIN_FREQ != data_dict['min_freq'])):
            raise ValueError()

        print('Using cached vocabs...')
        DE = data_dict['DE']
        EN = data_dict['EN']
        train, val, test = load_dataset_from_fields(DE, EN)
    except:
        # Load tokenizers
        spacy_de = spacy.load('de')
        spacy_en = spacy.load('en')
        DE = data.Field(tokenize=lambda text: tokenize(text, spacy_de))
        EN = data.Field(tokenize=lambda text: tokenize(text, spacy_en),
                        init_token = BOS_WORD, eos_token = EOS_WORD)

        train, val, test = load_dataset_from_fields(DE, EN)

        # Build vocabulary
        DE.build_vocab(train.src, min_freq=MIN_FREQ)
        EN.build_vocab(train.trg, min_freq=MIN_FREQ)

        # Load pretrained embeddings
        if use_pretrained_emb == 'GloVe':
            print('Loading GloVe EN embeddings...')
            EN.vocab.load_vectors(vectors=GloVe(name='6B'))
        elif use_pretrained_emb == 'fastText':
            print('Loading fastText EN / DE embeddings...')
            en_url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
            de_url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.de.vec'
            EN.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=en_url))
            DE.vocab.load_vectors(vectors=Vectors('wiki.de.vec', url=de_url))

        # Save vocab fields
        data_dict = {'DE': DE, 'EN': EN, 'max_len': MAX_LEN, 'min_freq': MIN_FREQ}
        joblib.dump(data_dict, fname)

    # Train-validation split
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train, val, test), batch_size=batch_size, device=-1,
        repeat=False, sort_key=lambda x: len(x.src)
    )

    print("[TRAIN]:{} (dataset:{})\t[VAL]:{} (dataset:{})\t[TEST]:{} (dataset:{})".format(
        len(train_iter), len(train_iter.dataset),
        len(val_iter), len(val_iter.dataset),
        len(test_iter), len(test_iter.dataset)))
    print("[SRC_vocab]:{} (DE)\t[TRG_vocab]:{} (EN)".format(len(DE.vocab), len(EN.vocab)))

    return train_iter, val_iter, test_iter, DE, EN

def load_kaggle(TEXT):
    print('Loading Kaggle data...')

    out = []
    for line in open('source_test.txt'):
        words = line.split()[:-1]
        out.append([TEXT.vocab.stoi[word] for word in words])

    print("[PRED]:{} (dataset:{})".format(len(out), len(out)))
    return out

def get_src_and_trgs(batch, use_cuda, is_eval):
    '''
    Returns tuple of Variables representing
    (src, trg_input, trg_targets) for batch.
    Each batch has shape (batch_size, seq_len)
    '''
    src_and_trgs = (batch.src, batch.trg[:-1], batch.trg[1:])
    out = tuple(o.data.t().contiguous() for o in src_and_trgs)
    if use_cuda:
        return tuple(Variable(o.cuda(), volatile=is_eval) for o in out)
    else:
        return tuple(Variable(o, volatile=is_eval) for o in out)

def seq_to_text(seq, TEXT):
    '''
    seq: torch.Tensor
    '''
    return [TEXT.vocab.itos[idx] for idx in seq]

def sample(num_samples, src, trg, pred, SRC, TRG):
    '''
    Sample src, trg, and pred sentences.
    '''
    src = src.data
    trg = trg.data
    pred = pred.data
    for i in range(num_samples):
        print('>>>>> SAMPLE {}'.format(i))
        print('[SRC] {}'.format(' '.join(seq_to_text(src[i], SRC))))
        print('[TRG] {}'.format(' '.join(seq_to_text(trg[i], TRG))))
        print('[PRED] {}'.format(' '.join(seq_to_text(pred[i], TRG))))

def visualize_attn(attn, src, trg, fname):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attn, cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + src, rotation=90)
    ax.set_yticklabels([''] + trg)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # Save
    plt.savefig(fname)
    plt.close()
