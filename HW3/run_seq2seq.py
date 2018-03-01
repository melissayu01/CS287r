import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from Seq2Seq import Seq2Seq
from torch.nn.utils import clip_grad_norm
from utils import *
import time

USE_CUDA = True

# Set seed
torch.manual_seed(2222)
if USE_CUDA:
    torch.cuda.manual_seed(2222)

def main():
    # Load data
    print('Loading data...')
    train_iter, val_iter, DE, EN = load_data(max_len=20, min_freq=5, batch_size=32)
    PAD_IDX = EN.vocab.stoi[EN.pad_token]

    # Build model and optimizer
    print('Building model and optimizer...')
    s2s = Seq2Seq(enc_vocab_size=len(DE.vocab), dec_vocab_size=len(EN.vocab), 
                  enc_embed_dim=1500, dec_embed_dim=1500, hidden_size=1500, 
                  enc_num_layers=2, dec_num_layers=2, padding_idx=PAD_IDX, 
                  dropout_rate=0.2)
    if USE_CUDA:
        s2s = s2s.cuda()
    optimizer = optim.SGD(params=s2s.parameters(), lr=1)
    # optimizer = optim.Adam(params=s2s.parameters(), lr=10**-2)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, threshold=10**-3, patience=0)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[7, 14], gamma=0.5, last_epoch=-1)
    loss_func = nn.NLLLoss(size_average=False, ignore_index=PAD_IDX)

    # Train model
    print('Training...')
    print('-' * 102)
    N_EPOCHS = 20
    for i in range(N_EPOCHS):
        # Train a single epoch
        lr = optimizer.param_groups[0]['lr']
        start = time.time()
        train_loss = train(s2s, train_iter, loss_func, optimizer, pad_idx=PAD_IDX, max_grad_norm=5)
        end = time.time()
        train_loss = train_loss[0]
        train_time = (end - start)

        # Calculate loss and ppl on validation set, adjust learning rate if needed
        val_loss = eval(s2s, val_iter, loss_func, PAD_IDX)
        val_loss = val_loss[0]
        ppl = np.exp(val_loss)
        scheduler.step()

        # Save model
        torch.save(s2s, 's2s_model.pt')

        # Print epoch update
        print('| Epoch #{:2d} | train loss {:5.4f} | train time {:5.2f} | val loss {:5.4f} | val ppl {:7.4f} | lr {:4f}'.format(i, train_loss, train_time, val_loss, ppl, lr))
        print('-' * 102)

# Train model
def train(model, data, loss_func, optimizer, pad_idx, max_grad_norm):
    model.train()
    data.init_epoch()
    total_n = torch.FloatTensor([0])
    total_loss = torch.FloatTensor([0])
    if USE_CUDA:
        total_n = total_n.cuda()
        total_loss = total_loss.cuda()
    for batch in data:
        model.zero_grad()
        # Reverse order of input
        src = batch.src[range(batch.src.size(0)-1, -1, -1)]
        # src = batch.src
        
        trg_input = batch.trg[:-1]
        trg_output = batch.trg[1:]
        n_not_pad = (trg_output.data != pad_idx).int().sum()
        total_n += n_not_pad
        if USE_CUDA:
            src = src.cuda()
            trg_input = trg_input.cuda()
            trg_output = trg_output.cuda()
        log_probs = model(src, trg_input)
        if USE_CUDA:
            log_probs = log_probs.cuda()
        batch_loss = loss_func(log_probs.view(-1, model.dec_vocab_size), trg_output.view(-1))
        batch_avg_loss = batch_loss / n_not_pad
        if USE_CUDA:
            batch_avg_loss = batch_avg_loss.cuda()
        batch_avg_loss.backward()
        clip_grad_norm(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += batch_loss.data
    return total_loss / total_n

# Evaluate model; calculate avg validation loss
def eval(model, data, loss_func, pad_idx):
    model.eval()
    data.init_epoch()
    total_n = torch.FloatTensor([0])
    total_loss = torch.FloatTensor([0])
    if USE_CUDA:
        total_n = total_n.cuda()
        total_loss = total_loss.cuda()
    for batch in data:
        # Reverse order of input
        src = batch.src[range(batch.src.size(0)-1, -1, -1)]
        # src = batch.src
        
        trg_input = batch.trg[:-1]
        trg_output = batch.trg[1:]
        n_not_pad = (trg_output.data != pad_idx).int().sum()
        total_n += n_not_pad
        if USE_CUDA:
            src = src.cuda()
            trg_input = trg_input.cuda()
            trg_output = trg_output.cuda()
        
        log_probs = model(src, trg_input)
        if USE_CUDA:
            log_probs = log_probs.cuda()
        loss = loss_func(log_probs.view(-1, model.dec_vocab_size), trg_output.view(-1))
        total_loss += loss.data   
    avg_loss = total_loss / total_n
    model.train()
    return avg_loss

if __name__ == '__main__':
    main()