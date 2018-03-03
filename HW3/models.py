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

        self.dropout = nn.Dropout(dropout)

        emb = nn.Embedding(
            input_size, emb_size, padding_idx=padding_idx,
            max_norm=max_norm
        )
        if embeddings:
            emb.weight = nn.Parameter(embeddings)
        self.embedding = nn.Sequential(emb, self.dropout)

        self.lstm = nn.LSTM(
            batch_first=True, input_size=emb_size, hidden_size=hidden_size,
            num_layers=num_layers, dropout=dropout,
            bidirectional=bidirectional
        )


    def forward(self, input):
        # (batch_size, seq_len, emb_size)
        embedded = self.embedding(input)

        self.lstm.flatten_parameters()
        '''
        output: (batch_size, seq_len, hidden_size * num_directions)
                all hidden states for last layer in RNN
        hidden[0]: (num_layers * num_directions, batch_size, hidden_size)
                   last hidden state for each layer and direction in RNN
        hidden[1]: (num_layers * num_directions, batch_size, hidden_size)
                   last cell state for each layer and direction in RNN
        '''
        output, hidden = self.lstm(embedded)

        return output, hidden


class DecoderRNN(EncoderRNN):
    def __init__(self, enc_num_directions, enc_hidden_size,
                 use_context, **kwargs):
        super(DecoderRNN, self).__init__(**kwargs)
        self.enc_num_directions = enc_num_directions
        self.enc_hidden_size = enc_hidden_size
        self.use_context = use_context

        context_size = (
            use_context * self.N * enc_hidden_size * enc_num_directions)
        output_size = context_size + self.H * self.num_directions

        self.out = nn.Sequential(
            self.dropout,
            nn.Linear(output_size, self.V),
        )

    def forward(self, input, hidden, _enc_outputs, _mask):
        embedded = self.embedding(input)

        self.lstm.flatten_parameters()
        output, hidden = self.lstm(embedded, hidden)

        if self.use_context > 0:
            context = torch.cat(hidden[:use_context], dim=0)

            seq_len = trg.size(1)
            batch_size = context.size(1)

            # (batch_size, seq_len, context_size)
            context = context.transpose(0, 1).view(
                batch_size, 1, -1).expand(-1, seq_len, -1)

            output = torch.cat((output, context), dim=2)

        # (batch_size, out_len, out_vocab_size)
        output = self.out(output)
        return output, hidden, context


class AttnDecoderRNN(DecoderRNN):
    def __init__(self, **kwargs):
        super(AttnDecoderRNN, self).__init__(**kwargs)

        context_size = self.enc_hidden_size * self.enc_num_directions
        decoder_size = self.H * self.num_directions

        self.context_to_decoder = nn.Linear(
            context_size,
            decoder_size
        )
        self.out_decoder = nn.Sequential(
            self.dropout,
            nn.Linear(decoder_size, self.V),
        )
        self.out_context =  nn.Sequential(
            self.dropout,
            nn.Linear(context_size, self.V),
        )

    def forward(self, input, hidden, enc_output, mask=None):
        embedded = self.embedding(input)

        self.lstm.flatten_parameters()
        output, hidden = self.lstm(embedded, hidden)

        # (batch_size, in_len, decoder_size)
        enc_to_dec = self.context_to_decoder(enc_output)

        # (batch_size, out_len, in_len)
        attn = torch.bmm(output, enc_to_dec.transpose(1, 2))

        batch_size, in_len = attn.size(0), attn.size(2)
        if mask is not None:
            attn.data.masked_fill_(mask, -float('inf'))

        # (batch_size, out_len, in_len)
        attn = F.softmax(attn.view(-1, in_len), dim=1).view(
            batch_size, -1, in_len)

        # (batch_size, out_len, context_size)
        context = torch.bmm(attn, enc_output)

        # (batch_size, out_len, out_vocab_size)
        output = self.out_decoder(output) + self.out_context(context)
        return output, hidden, attn


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, use_cuda):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.use_cuda = use_cuda

    def _get_attn_mask(self, src):
        mask = torch.eq(src, self.encoder.padding_idx)
        if self.use_cuda:
            mask = mask.cuda()
        return mask

    def forward(self, src, trg, use_attn_mask=False, use_teacher_forcing=True):
        enc_output, enc_hidden = self.encoder(src)

        if self.encoder.num_directions == 2:
            enc_hidden = tuple(h[self.encoder.N:] for h in enc_hidden)

        mask = self._get_attn_mask(src.data) if use_attn_mask else None

        if use_teacher_forcing:
            dec_output, dec_hidden, context_or_attn = self.decoder(
                trg, enc_hidden, enc_output, mask)
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
