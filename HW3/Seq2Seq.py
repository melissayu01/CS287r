import torch
import torch.nn as nn
import torch.nn.functional as F

class Seq2Seq(nn.Module):
    
    def __init__(self, enc_vocab_size, dec_vocab_size, enc_embed_dim, dec_embed_dim, hidden_size,
                 enc_num_layers, dec_num_layers, padding_idx, dropout_rate=0):
        # Save hyperparameters
        super(Seq2Seq, self).__init__()
        self.enc_vocab_size = enc_vocab_size
        self.dec_vocab_size = dec_vocab_size
        self.enc_embed_dim = enc_embed_dim
        self.dec_embed_dim = dec_embed_dim
        self.hidden_size = hidden_size
        self.enc_num_layers = enc_num_layers
        self.dec_num_layers = dec_num_layers
        self.padding_idx = padding_idx
        self.dropout_rate = dropout_rate
        
        # Layers
        self.enc_embedding = nn.Embedding(enc_vocab_size, enc_embed_dim, padding_idx=padding_idx)
        self.enc_lstm = nn.LSTM(input_size=enc_embed_dim, hidden_size=hidden_size, 
                                num_layers=enc_num_layers, dropout=dropout_rate)
        self.dec_embedding = nn.Embedding(dec_vocab_size, dec_embed_dim, padding_idx=padding_idx)
        self.dec_lstm = nn.LSTM(input_size=dec_embed_dim, hidden_size=hidden_size, 
                                num_layers=dec_num_layers, dropout=dropout_rate)
        self.linear = nn.Linear(hidden_size, dec_vocab_size)
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Weight initialization
        for p in self.enc_lstm.parameters():
            p.data.uniform_(-0.08, 0.08)
        for p in self.dec_lstm.parameters():
            p.data.uniform_(-0.08, 0.08)
    
    def forward(self, src, trg):        
        # Encoder
        enc_input = self.enc_embedding(src)
        _, hidden = self.enc_lstm(enc_input)
        
        # Decoder
        dec_input = self.dec_embedding(trg)
        output, _ = self.dec_lstm(dec_input, hidden)
        output = self.dropout(output)
        output = self.linear(output)
        log_probs = F.log_softmax(output, dim=2)
        return log_probs