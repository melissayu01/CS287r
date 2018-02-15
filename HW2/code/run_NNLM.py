import time
import math
import argparse

import torch
from torch import optim, nn
from torch.nn import functional as F

import utils
from NNLM import NNLM

torch.set_printoptions(precision=4)
PREDS_DIR = '../predictions/'
MODELS_DIR = '../models/'


def make_predictions(model, TEXT, ngram, criterion, pred_file):
    model.eval()
    test = utils.load_kaggle(TEXT)

    n = ngram
    padding_idx = TEXT.vocab.stoi["<pad>"]
    ntokens = 0
    total_loss = 0

    with open(PREDS_DIR+pred_file, "w") as fout:
        print("id,word", file=fout)
        for i, vec in enumerate(test, start=1):
            data, targets = vec.unfold(0, n-1, 1), vec[n-1:]
            output = model(data)

            _, indices = torch.topk(output[-1], k=20)
            predictions = [TEXT.vocab.itos[i] for i in indices.data.tolist()]
            print("%d,%s"%(i, " ".join(predictions)), file=fout)

            total_loss += criterion(output[:-1], targets).data
            ntokens += targets.ne(padding_idx).int().sum().data

    pred_loss = total_loss[0] / ntokens[0]
    print('=' * 89)
    print('| End of predicting | kaggle loss {:5.2f} | kaggle ppl {:8.2f}'.format(
        pred_loss, math.exp(pred_loss)))
    print('=' * 89)

def evaluate(model, data_loader, TEXT, criterion, args):
    model.eval()

    n = args.ngram
    padding_idx = TEXT.vocab.stoi["<pad>"]
    ntokens = 0
    total_loss = 0

    for i, batch in enumerate(data_loader):
        vec = batch.text.transpose(0, 1).contiguous().view(-1)
        data, targets = vec.unfold(0, n-1, 1)[:-1], vec[n-1:]

        output = model(data)
        total_loss += criterion(output, targets).data
        ntokens += targets.ne(padding_idx).int().sum().data

    return total_loss[0] / ntokens[0]

def train(epoch, model, data_loader, TEXT, criterion, optimizer, args):
    model.train()

    n = args.ngram
    padding_idx = TEXT.vocab.stoi["<pad>"]
    ntokens = 0
    total_loss = 0
    start_time = time.time()

    for i, batch in enumerate(data_loader):
        vec = batch.text.transpose(0, 1).contiguous().view(-1)
        data, targets = vec.unfold(0, n-1, 1)[:-1], vec[n-1:]

        model.zero_grad()
        output = model(data)
        loss = criterion(output, targets)
        loss.backward()

        nn.utils.clip_grad_norm(model.parameters(), max_norm=args.clip)
        optimizer.step()

        ntokens += targets.ne(padding_idx).int().sum().data
        total_loss += loss.data

        if i % args.log_interval == 0 and i > 0:
            cur_loss = total_loss[0] / ntokens[0]
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.4f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, i, len(data_loader), args.lr, elapsed * 1000 / args.log_interval,
                        cur_loss, math.exp(cur_loss)))
            ntokens = 0
            total_loss = 0
            start_time = time.time()

def main():
    parser = argparse.ArgumentParser(description='PyTorch PTB NN Language Model')
    parser.add_argument('--ngram', type=int, default=3,
                        help='n-gram size')
    parser.add_argument('--use_pretrained_em', action='store_true',
                        help='use pretrained word embeddings')
    parser.add_argument('--emsize', type=int, default=200,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=200,
                        help='number of hidden units per layer')
    parser.add_argument('--em_maxnorm', type=float, default=5,
                        help='max norm for learned embeddings')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    # parser.add_argument('--nlayers', type=int, default=2,
    #                     help='number of layers')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=10,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt_len', type=int, default=32,
                        help='sequence length')
    # parser.add_argument('--dropout', type=float, default=0.2,
    #                     help='dropout applied to layers (0 = no dropout)')
    # parser.add_argument('--tied', action='store_true',
    #                     help='tie the word embedding and softmax weights')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str,  default='model.pt',
                        help='path to save the final model')
    parser.add_argument('--kaggle', type=str,  default='',
                        help='path to save kaggle predictions')
    parser.add_argument('--dev', action='store_true',
                        help='use dev mode')
    args = parser.parse_args()

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)

    # Load data
    train_iter, val_iter, test_iter, TEXT = utils.load_PTB(
        dev=args.dev, use_pretrained_embeddings=args.use_pretrained_em,
        batch_size=args.batch_size, bptt_len=args.bptt_len)

    # Intialize model, loss, and optimizers.
    model = NNLM(n=args.ngram, V=len(TEXT.vocab),
                 m=(TEXT.vocab.vectors.size(1) if args.use_pretrained_em else args.emsize),
                 h=args.nhid, max_norm=args.em_maxnorm,
                 embeddings=(TEXT.vocab.vectors if args.use_pretrained_em else None))
    if args.cuda:
        model.cuda()
    criterion = torch.nn.CrossEntropyLoss(
        size_average=False, ignore_index=TEXT.vocab.stoi["<pad>"])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=0, threshold=5e-3)

    # Train / Test.
    lr = args.lr
    best_val_loss = None

    print('=' * 89)
    try:
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            train(epoch, model, train_iter, TEXT, criterion, optimizer, args)
            val_loss = evaluate(model, val_iter, TEXT, criterion, args)

            # Log results
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(
                        epoch, (time.time() - epoch_start_time),
                        val_loss, math.exp(val_loss)))
            print('-' * 89)

            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                torch.save(model, MODELS_DIR+args.save)
                best_val_loss = val_loss

            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            scheduler.step(val_loss)
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open(MODELS_DIR+args.save, 'rb') as f:
        model = torch.load(f)

    # Run on test data.
    test_loss = evaluate(model, test_iter, TEXT, criterion, args)

    # Log results
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)

    # Make Kaggle predictions.
    if args.kaggle:
        make_predictions(model, TEXT, args.ngram, criterion, args.kaggle)

if __name__ == '__main__':
    main()
