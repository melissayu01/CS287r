import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import MAX_LEN

class EncoderRNN(nn.Module):
    def __init__(self,
                 input_size, emb_size, embeddings, max_norm, padding_idx,
                 hidden_size, num_layers, dropout, bidirectional):
        super(EncoderRNN, self).__init__()

        self.V = input_size
        self.M = emb_size
        self.H = hidden_size
        self.N = num_layers
        self.padding_idx = padding_idx
        self.num_directions = 2 if bidirectional else 1

        # embedded: (batch_size, seq_len, emb_size)
        self.embedding = nn.Embedding(
            input_size, emb_size, padding_idx=padding_idx,
            max_norm=max_norm
        )
        if embeddings:
            self.embedding.weight = nn.Parameter(embeddings)

        '''
        output: (batch_size, seq_len, hidden_size * num_directions)
                all hidden states for last layer in RNN
        hidden[0]: (num_layers * num_directions, batch_size, hidden_size)
                   last hidden state for each layer and direction in RNN
        hidden[1]: (num_layers * num_directions, batch_size, hidden_size)
                   last cell state for each layer and direction in RNN
        '''
        self.lstm = nn.LSTM(
            batch_first=True, input_size=emb_size, hidden_size=hidden_size,
            num_layers=num_layers, dropout=dropout,
            bidirectional=bidirectional
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        embedded = self.embedding(input)
        self.lstm.flatten_parameters()
        output, hidden = self.lstm(embedded)
        return output, hidden


class DecoderRNN(EncoderRNN):
    def __init__(self, enc_num_directions, enc_hidden_size,
                 use_context, **kwargs):
        super(DecoderRNN, self).__init__(**kwargs)
        self.enc_num_directions = enc_num_directions
        self.enc_hidden_size = enc_hidden_size
        self.use_context = use_context

        # out: (batch_size, seq_len, vocab_size)
        context_size = (
            use_context * self.N * enc_hidden_size * enc_num_directions)
        output_size = context_size + self.H * self.num_directions
        self.out = nn.Sequential(
            self.dropout,
            nn.Linear(output_size, self.V),
        )

    def forward(self, input, hidden, _):
        embedded = self.embedding(input)
        embedded = F.relu(embedded)
        self.lstm.flatten_parameters()
        output, hidden = self.lstm(embedded, hidden)

        if self.use_context > 0:
            seq_len = trg.size(1)
            context = torch.cat(hidden[:use_context], dim=0)
            batch_size = context.size(1)

            '''
            (batch_size, seq_len,
             use_context * num_layers * enc_hidden_size * enc_num_directions)
            '''
            context = context.permute(1, 0, 2).contiguous().view(
                batch_size, 1, -1).expand(-1, seq_len, -1)
            output = torch.cat((output, context), dim=2)

        output = self.out(output)
        return output, hidden, context


class AttnDecoderRNN(DecoderRNN):
    def __init__(self, **kwargs):
        super(AttnDecoderRNN, self).__init__(**kwargs)

        context_size = self.enc_hidden_size * self.enc_num_directions
        output_size = context_size + self.H * self.num_directions
        self.out = nn.Sequential(
            self.dropout,
            nn.Linear(output_size, self.V),
        )

    def forward(self, input, hidden, enc_output):
        embedded = self.embedding(input)
        embedded = F.relu(embedded)
        self.lstm.flatten_parameters()
        output, hidden = self.lstm(embedded, hidden)

        # (batch_size, trg_seq_len, src_seq_len)
        attn = F.softmax(
            torch.bmm(output, enc_output.permute(0, 2, 1)),
            dim=2
        )

        # (batch_size, trg_seq_len, enc_hidden_size * enc_num_directions)
        context = torch.bmm(attn, enc_output)
        output = torch.cat((output, context), dim=2)

        output = self.out(output)
        return output, hidden, attn


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, use_cuda):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.use_cuda = use_cuda

    def forward(self, src, trg):
        enc_output, enc_hidden = self.encoder(src)

        if True or src.size(0) == trg.size(0):
            dec_output, dec_hidden, context_or_attn = self.decoder(
                trg, enc_hidden, enc_output)
        else:
            batch_size = src.size(0)
            vocab_size = self.decoder.V

            dec_output = Variable(
                torch.zeros(batch_size, MAX_LEN, vocab_size))
            if self.use_cuda:
                dec_output.cuda()

            output = trg
            hidden = enc_hidden
            dec_output[:, 0, :] = output
            for t in range(1, MAX_LEN):
                output, hidden, attn_weights = self.decoder(
                        output, hidden, enc_output)
                dec_output[:, t, :] = output
                output = Variable(output.data.max(dim=2)[1])
                if self.use_cuda:
                    output.cuda()

        return dec_output, context_or_attn
