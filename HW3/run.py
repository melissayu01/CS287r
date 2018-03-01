import os
import time
import math
import argparse

import torch
from torch import optim, nn
from torch.nn import functional as F

import utils
import models

torch.set_printoptions(precision=4)

SAVE_DIR = '.save'
VIS_DIR = 'vis'
PREDS_DIR = 'preds'

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch DE-EN NMT')

    # embeddings
    parser.add_argument('--pretrained-emb', type=str,
                        choices=['GloVe', 'fastText'],
                        help='use pretrained word embeddings for EN')
    parser.add_argument('--emb-size', type=int, default=200,
                        help='size of word embeddings')
    parser.add_argument('--emb-maxnorm', type=float,
                        help='max norm for learned embeddings')

    # LSTM
    parser.add_argument('--hidden-size', type=int, default=200,
                        help='number of features in hidden state \
                        for encoder and decoder')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='number of recurrent layers for \
                        encoder and decoder')
    parser.add_argument('--dropout', type=float, default=0,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--bidirectional', action='store_true',
                        help='use bidirectional RNN for encoder \
                        and decoder (doubles hidden-size if on)')
    parser.add_argument('--use-context', type=int, default=0,
                        help='use encoder hidden & cell states as \
                        context for decoder; only used when attention=False')
    parser.add_argument('--attention', action='store_true',
                        help='use attention in decoder')

    # training
    parser.add_argument('--optimizer', type=str, default='SGD',
                        choices=['Adam', 'Adadelta', 'Adagrad', 'SGD'],
                        help='initial learning rate')
    parser.add_argument('--lr', type=float, default=1,
                        help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=10,
                        help='upper epoch limit')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--clip', type=float,
                        help='gradient clipping')

    # pytorch
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')

    # logging and saving
    parser.add_argument('--log-interval', type=int, default=400,
                        help='report interval')
    parser.add_argument('--sample', action='store_true',
                        help='sample predictions and attention every log')
    parser.add_argument('--save', type=str,  default='seq2seq',
                        help='filename for saving')

    return parser.parse_args()


def make_predictions(model, criterion, TRG, SRC, args):
    model.eval()

    test = utils.load_kaggle(SRC)
    vocab_size = model.decoder.V
    total_loss = 0
    fname = './{}/{}.txt'.format(PREDS_DIR, args.save)

    with open(fname, "w") as fout:
        print("id,word", file=fout)
        for i, batch in enumerate(test, start=1):
            src, trg_input, trg_targets = utils.get_src_and_trgs(
                batch, args.cuda, is_eval=True
            )

            output, context_or_attn = model(src, trg_input)

            loss = criterion(output.view(-1, vocab_size),
                             trg_targets.contiguous().view(-1))
            total_loss += loss.data[0]

            # _, indices = torch.topk(output[-1], k=20)
            # predictions = [TRG.vocab.itos[i] for i in indices.data.tolist()]
            # print("%d,%s"%(i, " ".join(predictions)), file=fout)

            # criterion.step(dec_output, trg_targets)
            pass

    pred_loss = total_loss / len(test)
    print('=' * 89)
    print('| End of predicting | kaggle loss {:5.2f} | kaggle ppl {:8.2f}'
          .format(pred_loss, math.exp(pred_loss)))
    print('=' * 89)


def evaluate(model, data_loader, criterion, use_cuda):
    model.eval()

    vocab_size = model.decoder.V
    total_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            src, trg_input, trg_targets = utils.get_src_and_trgs(
                batch, use_cuda, is_eval=True
            )

            output, context_or_attn = model(src, trg_input)

            loss = criterion(output.view(-1, vocab_size),
                             trg_targets.contiguous().view(-1))
            total_loss += loss.data[0]

    return total_loss / len(data_loader)


def train(epoch, model, data_loader, criterion, optimizer, use_cuda, args, SRC, TRG):
    model.train()

    vocab_size = model.decoder.V
    total_loss = 0
    attns = []
    start_time = time.time()

    for i, batch in enumerate(data_loader):
        optimizer.zero_grad()

        src, trg_input, trg_targets = utils.get_src_and_trgs(
            batch, use_cuda, is_eval=False
        )

        output, context_or_attn = model(src, trg_input)

        loss = criterion(output.view(-1, vocab_size),
                         trg_targets.contiguous().view(-1))
        loss.backward()

        if args.clip:
            nn.utils.clip_grad_norm(model.parameters(), max_norm=args.clip)
        optimizer.step()

        # record loss and attention for batch
        total_loss += loss.data[0]
        # if args.attention and args.sample:
        #     attns.append(context_or_attn)

        if i % args.log_interval == 0 and i > 0:
            pred = torch.topk(output.data, k=1, dim=2)[1]
            if args.sample:
                utils.sample(1, src, trg_targets, pred, SRC, TRG)
                if args.attention:
                    fname = './{}/{}.png'.format(VIS_DIR, args.save)
                    j = 0
                    sample_attn = attns[-1][j].squeeze().data.numpy()
                    sample_src = utils.seq_to_text(src[j].squeeze().data, SRC)
                    sample_pred = utils.seq_to_text(pred[j].squeeze(), TRG)
                    utils.visualize_attn(sample_attn, sample_src, sample_pred, fname)
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.4f} '
                  '| ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}'.format(
                      epoch, i, len(data_loader), args.lr,
                      elapsed * 1000 / args.log_interval,
                      cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

        # free some memory
        del output, context_or_attn, loss

    return attns

def main():
    # Get arguments
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    # Cuda
    use_cuda = False
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you \
            should probably run with --cuda")
        else:
            use_cuda = True
            torch.cuda.manual_seed(args.seed)

    # Load data + text fields
    print('=' * 89)
    train_iter, val_iter, test_iter, SRC, TRG = utils.load_dataset(
        batch_size=args.batch_size,
        use_pretrained_emb=args.pretrained_emb,
        save_dir=SAVE_DIR
    )
    print('=' * 89)

    # Intialize model
    enc = models.EncoderRNN(
        input_size=len(SRC.vocab),
        emb_size=(SRC.vocab.vectors.size(1)
                  if args.pretrained_emb == 'fastText'
                  else args.emb_size),
        embeddings=(SRC.vocab.vectors
                    if args.pretrained_emb == 'fastText'
                    else None),
        max_norm=args.emb_maxnorm,
        padding_idx=SRC.vocab.stoi['<pad>'],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional
    )
    decoder = models.AttnDecoderRNN if args.attention else models.DecoderRNN
    dec = decoder(
        enc_num_directions=enc.num_directions,
        enc_hidden_size=args.hidden_size,
        use_context=args.use_context,
        input_size=len(TRG.vocab),
        emb_size=(TRG.vocab.vectors.size(1)
                  if args.pretrained_emb
                  else args.emb_size),
        embeddings=(TRG.vocab.vectors
                    if args.pretrained_emb
                    else None),
        max_norm=args.emb_maxnorm,
        padding_idx=TRG.vocab.stoi['<pad>'],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional
    )
    model = models.Seq2Seq(enc, dec, use_cuda=use_cuda)
    if use_cuda:
        model.cuda()
    print(model)

    # Intialize loss
    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=TRG.vocab.stoi["<pad>"])

    # Create optimizer
    if args.optimizer == 'Adam':
        optim = torch.optim.Adam
    elif args.optimizer == 'Adadelta':
        optim = torch.optim.Adadelta
    elif args.optimizer == 'Adagrad':
        optim = torch.optim.Adagrad
    else:
        optim = torch.optim.SGD
    optimizer = optim(model.parameters(), lr=args.lr)

    # Create scheduler
    lambda_lr = lambda epoch: 0.5 if epoch > 8 else 1
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_lr)

    # Train
    best_val_loss = None
    fname = './{}/{}.pt'.format(SAVE_DIR, args.save)

    print('=' * 89)
    try:
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()

            attns = train(epoch, model, train_iter, criterion, optimizer,
                  use_cuda, args, SRC, TRG)
            val_loss = evaluate(model, val_iter, criterion, use_cuda)

            # Log results
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s '
                  '| valid loss {:5.2f} | valid ppl {:8.2f}'.format(
                      epoch, (time.time() - epoch_start_time),
                      val_loss, math.exp(val_loss)))
            print('-' * 89)

            # Save the model if validation loss is best we've seen so far
            if not best_val_loss or val_loss < best_val_loss:
                if not os.path.isdir(SAVE_DIR):
                    os.makedirs(SAVE_DIR)
                torch.save(model, fname)
                best_val_loss = val_loss

            # Anneal learning rate
            scheduler.step()
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model
    with open(fname, 'rb') as f:
        model = torch.load(f)

    # Run on test data
    test_loss = evaluate(model, test_iter, criterion, use_cuda)

    # Log results
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)

    # Make Kaggle predictions.
    make_predictions(model, criterion, TRG, SRC, args)

if __name__ == '__main__':
    main()
