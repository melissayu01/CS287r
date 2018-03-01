# Text text processing library and methods for pretrained word embeddings
from torchtext import data
from torchtext import datasets
import spacy
import joblib

# Globals
BOS_WORD = '<s>'
EOS_WORD = '</s>'

# Tokenize German words
def tokenize_de(text, spacy_de):
    return [tok.text for tok in spacy_de.tokenizer(text)]

# Tokenize English words
def tokenize_en(text, spacy_en):
    return [tok.text for tok in spacy_en.tokenizer(text)]

# Create training and validation set iterators, load German and English vocabs
def load_data(max_len, min_freq, batch_size):
    try: 
        data_dict = joblib.load('vocabs.jl')
        if ((max_len != data_dict['max_len']) | (min_freq != data_dict['min_freq'])):
            raise ValueError()
        else:
            DE = data_dict['DE']
            EN = data_dict['EN']
            
        # Get German-to-English dataset
        train, val, test = datasets.IWSLT.splits(exts=('.de', '.en'), fields=(DE, EN), 
                                                 filter_pred=lambda x: len(vars(x)['src']) <= max_len and 
                                                 len(vars(x)['trg']) <= max_len)
    except: 
        # Load tokenizers
        spacy_de = spacy.load('de')
        spacy_en = spacy.load('en')
        DE = data.Field(tokenize=lambda text: tokenize_de(text, spacy_de))
        EN = data.Field(tokenize=lambda text: tokenize_en(text, spacy_en), init_token = BOS_WORD, eos_token = EOS_WORD)
        
        # Get German-to-English dataset
        train, val, test = datasets.IWSLT.splits(exts=('.de', '.en'), fields=(DE, EN), 
                                                 filter_pred=lambda x: len(vars(x)['src']) <= max_len and 
                                                 len(vars(x)['trg']) <= max_len)

        # Build vocabulary 
        DE.build_vocab(train.src, min_freq=min_freq)
        EN.build_vocab(train.trg, min_freq=min_freq)
        
        data_dict = {'DE': DE, 'EN': EN, 'max_len': max_len, 'min_freq': min_freq}
        joblib.dump(data_dict, 'vocabs.jl')

    # Create iterators; train-validation split
    train_iter, val_iter = data.BucketIterator.splits((train, val), batch_size=batch_size, device=-1,
                                                      repeat=False, sort_key=lambda x: len(x.src))
    return train_iter, val_iter, DE, EN