import time
import math
import argparse

import torch
from torch import optim, nn
from torch.nn import functional as F

import utils
from LinearInterpTrigram import LinearInterpTrigram

torch.set_printoptions(precision=4)
PREDS_DIR = '../predictions/'
MODELS_DIR = '../models/'


def make_predictions(model, TEXT, ngram, criterion, pred_file):
    model.eval()
    test = utils.load_kaggle(TEXT)

    padding_idx = TEXT.vocab.stoi["<pad>"]
    ntokens = 0
    total_loss = 0

    with open(PREDS_DIR+pred_file, "w") as fout:
        print("id,word", file=fout)
        for i, data in enumerate(test, start=1):
            output = model(data)

            _, indices = torch.topk(output[-1], k=20)
            predictions = [TEXT.vocab.itos[i] for i in indices.data.tolist()]
            print("%d,%s"%(i, " ".join(predictions)), file=fout)

            total_loss += criterion(output[:-1], targets).data
            ntokens += targets.ne(padding_idx).int().sum().data

    return total_loss[0] / ntokens[0]

def evaluate(model, data_loader, TEXT, criterion, args):
    model.eval()

    padding_idx = TEXT.vocab.stoi["<pad>"]
    ntokens = 0
    total_loss = 0

    for i, batch in enumerate(data_loader):
        data = batch.text.transpose(0, 1).contiguous().view(-1)

        output = model(data)
        total_loss += criterion(output, targets).data
        ntokens += targets.ne(padding_idx).int().sum().data

    return total_loss[0] / ntokens[0]

def train(model, train_loader, val_loader, TEXT, criterion, args):
    model.train()

    padding_idx = TEXT.vocab.stoi["<pad>"]
    ntokens = 0
    total_loss = 0
    start_time = time.time()

    for i, batch in enumerate(train_loader):
        data = batch.text.transpose(0, 1).contiguous().view(-1)
        model(data)

    for i, batch in enumerate(val_loader):
        data = batch.text.transpose(0, 1).contiguous().view(-1)
        model(data)

        loss = criterion(output, targets)
        loss.backward()

        ntokens += targets.ne(padding_idx).int().sum().data
        total_loss += loss.data

        if i % args.log_interval == 0 and i > 0:
            cur_loss = total_loss[0] / ntokens[0]
            elapsed = time.time() - start_time
            print('| {:5d}/{:5d} batches | lr {:02.4f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                        i, len(data_loader), args.lr, elapsed * 1000 / args.log_interval,
                        cur_loss, math.exp(cur_loss)))
            ntokens = 0
            total_loss = 0
            start_time = time.time()

def main():
    parser = argparse.ArgumentParser(description='PyTorch PTB NN Language Model')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
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

    # Intialize model, loss.
    model = LinearInterpTrigram(V=len(TEXT.vocab))
    if args.cuda:
        model.cuda()
    criterion = torch.nn.CrossEntropyLoss(
        size_average=False, ignore_index=TEXT.vocab.stoi["<pad>"])

    # Train
    print('=' * 89)
    train(model, train_iter, val_iter, TEXT, criterion, args)
    val_loss = evaluate(model, val_iter, TEXT, criterion, args)

    # TODO: learn weights...

    # Save model
    torch.save(model, MODELS_DIR+args.save)

    # Run on test data.
    test_loss = evaluate(model, test_iter, TEXT, criterion, args)

    # Log results
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)

    # Make Kaggle predictions.
    if args.kaggle:
        pred_loss = make_predictions(model, TEXT, args.ngram, criterion, args.kaggle)
        print('=' * 89)
        print('| End of predicting | kaggle loss {:5.2f} | kaggle ppl {:8.2f}'.format(
            pred_loss, math.exp(pred_loss)))
        print('=' * 89)


if __name__ == '__main__':
    main()
