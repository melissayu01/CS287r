import numpy as np
import torch
from torch.autograd import Variable
from torchtext import data
import spacy
import time
import joblib
from utils import tokenize_de
import argparse

# Globals
USE_CUDA = True

def main():
    args = parse_args()
    
    # Load model
    print('Loading model...')
    s2s = torch.load(args.model)
    s2s.eval()
    
    # Load dictionaries
    print('Loading dictionaries...')
    data_dict = joblib.load('vocabs.jl')
    DE = data_dict['DE']
    EN = data_dict['EN']
    BOS_WORD = '<s>'
    BOS_INT = EN.vocab.stoi[BOS_WORD]
    EOS_WORD = '</s>'
    EOS_INT = EN.vocab.stoi[EOS_WORD]
    
    # Read in sentences to translate
    print('Loading input file...')
    all_sents = []
    spacy_de = spacy.load('de')
    with open("source_test.txt") as file:
        for line in file:
            words = tokenize_de(line, spacy_de)
            words = [DE.vocab.stoi[w] for w in words]
            all_sents.append(Variable(torch.LongTensor(words)))
    
    # Perform beam search and write to file
    print('Translating...')
    with open('output/' + args.output, 'w') as file:
        file.write('id,word\n')
        for i in range(len(all_sents)):
            beams = beam_search(s2s, all_sents[i], EN, beam_size=args.beamsize, reverse=args.revinput, 
                                BOS_INT=BOS_INT, EOS_INT=EOS_INT)
            top100 = ' '.join([ints_to_strings(b, EN) for b in beams[:100]])
            file.write('{},{}\n'.format(i+1, top100))
            if i % 50 == 0:
                print('Finished {} translations.'.format(i))
        
# Flatten a list of lists into a single list
def flatten(lst):
    return [item for sublist in lst for item in sublist]

# Return top N max values from a list/array
def idx_sort(a,N):
    return np.argsort(a)[::-1][:N]

# Replace quotes and commas with delimiters
def escape(l):
    return l.replace("\"", "<quote>").replace(",", "<comma>")

# Convert a list of ints to strings based on given vocabulary
def ints_to_strings(lst, V):
    strings = [V.vocab.itos[i] for i in lst[1:]]
    return escape('|'.join(strings))

# Perform beam search on an input sentence in German
def beam_search(model, sent, EN, beam_size=100, reverse=False, BOS_INT=None, EOS_INT=None):
    if reverse:
        sent = sent[range(sent.shape[0]-1, -1, -1)].unsqueeze(1)
    else:
        sent = sent.unsqueeze(1)
    if USE_CUDA:
        sent = sent.cuda()
    
    beams = [[BOS_INT]]
    beams_log_probs = [0]
    for _ in range(3):
        prev_w = Variable(torch.LongTensor(np.array(beams).T))
        if USE_CUDA:
            prev_w = prev_w.cuda()

        # Calculate probabilities of new word
        new_w_log_probs = model(sent.expand(-1, len(beams)), prev_w)
        new_w_log_probs = new_w_log_probs.data.cpu().numpy()[-1]
        assert len(beams) == new_w_log_probs.shape[0]

        # Incorporate probabilities of new beam trajectories
        new_beams = []
        new_beams_log_probs = []
        for b in range(len(beams)):
            beam = beams[b]
            beam_log_prob = beams_log_probs[b]
            new_w_log_prob = new_w_log_probs[b]
            potential_new_beams = np.array(range(len(EN.vocab)))

            # Pre-filter best beams
            best_idx = idx_sort(new_w_log_prob, beam_size)
            potential_new_beams = potential_new_beams[best_idx]
            new_w_log_prob = new_w_log_prob[best_idx]
            new_beams.append([beam + [i] for i in potential_new_beams])
            new_beams_log_probs.append((beam_log_prob + new_w_log_prob).tolist())
        beams = flatten(new_beams)
        beams_log_probs = flatten(new_beams_log_probs)

        # Cut list to length = beam_size
        top_idx = idx_sort(beams_log_probs, beam_size).tolist()
        beams = np.array(beams)[top_idx].tolist()
        beams_log_probs = np.array(beams_log_probs)[top_idx].tolist()
    return beams

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str) # Model .pt file
    parser.add_argument("--output", type=str) # Output file name
    parser.add_argument("--revinput", type=bool, default=False) # Whether or not to reverse input
    parser.add_argument("--beamsize", type=int, default=100) # Desired beam search size
    return parser.parse_args()

if __name__ == '__main__':
    main()