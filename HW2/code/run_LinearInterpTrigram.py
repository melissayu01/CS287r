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


def make_predictions(model, TEXT, criterion, pred_file):
    model.eval()
    test = utils.load_kaggle(TEXT)

    padding_idx = TEXT.vocab.stoi["<pad>"]
    ntokens = 0
    total_loss = 0

    with open(PREDS_DIR+pred_file, "w") as fout:
        print("id,word", file=fout)
        for i, data in enumerate(test, start=1):
            output, targets = model(data.view(1, -1), TEXT)

            _, indices = torch.topk(output.squeeze()[-1], k=20)
            predictions = [TEXT.vocab.itos[i] for i in indices.data.tolist()]
            print("%d,%s"%(i, " ".join(predictions)), file=fout)

            output = output[:, :-1, :].contiguous().view(-1, len(TEXT.vocab))
            targets = targets.view(-1)
            total_loss += criterion(output, targets).data
            ntokens += targets.ne(padding_idx).int().sum().data

    return total_loss[0] / ntokens[0]

def evaluate(model, data_loader, TEXT, criterion, args):
    model.eval()

    padding_idx = TEXT.vocab.stoi["<pad>"]
    ntokens = 0
    total_loss = 0
    start_time = time.time()

    for i, batch in enumerate(data_loader):
        print(i)
        if i > 0: break
        data = batch.text.transpose(0, 1).contiguous()

        output, targets = model(data, TEXT)
        output = output[:, :-1, :].contiguous().view(-1, len(TEXT.vocab))
        targets = targets.view(-1)
        total_loss += criterion(output, targets).data
        ntokens += targets.ne(padding_idx).int().sum().data

    return total_loss[0] / ntokens[0]

def train(model, data_loader, TEXT, criterion, args):
    model.train()

    start_time = time.time()

    # get weights from val set
    for i, batch in enumerate(data_loader):
        data = batch.text.transpose(0, 1).contiguous()
        model.get_counts(data, TEXT)

        if i % args.log_interval == 0 and i > 0:
            elapsed = time.time() - start_time
            print('| {:5d}/{:5d} batches | ms/batch {:5.2f}'.format(
                i, len(data_loader), elapsed * 1000 / args.log_interval))
            start_time = time.time()


def main():
    parser = argparse.ArgumentParser(description='PyTorch PTB Trigram Model w/ Linear Interpolation')
    parser.add_argument('--batch_size', type=int, default=10, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt_len', type=int, default=32,
                        help='sequence length')
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
        dev=args.dev, use_pretrained_embeddings=False,
        batch_size=args.batch_size, bptt_len=args.bptt_len)

    # Intialize model, loss.
    model = LinearInterpTrigram(V=len(TEXT.vocab))
    if args.cuda:
        model.cuda()
    criterion = torch.nn.CrossEntropyLoss(
        size_average=False, ignore_index=TEXT.vocab.stoi["<pad>"])
    print('=' * 89)

    # Get counts.
    try:
        count_start_time = time.time()
        train(model, train_iter, TEXT, criterion, args)
        val_loss = evaluate(model, val_iter, TEXT, criterion, args)

        # Log results
        print('-' * 89)
        print('| End of counting | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(
                    (time.time() - count_start_time),
                    val_loss, math.exp(val_loss)))
        print('-' * 89)

        # Save the model.
        torch.save(model, MODELS_DIR+args.save)
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    model = torch.load(MODELS_DIR+args.save)

    # Run on test data.
    test_loss = evaluate(model, test_iter, TEXT, criterion, args)

    # Log results
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)

    # Make Kaggle predictions.
    if args.kaggle:
        pred_loss = make_predictions(model, TEXT, criterion, args.kaggle)
        print('=' * 89)
        print('| End of predicting | kaggle loss {:5.2f} | kaggle ppl {:8.2f}'.format(
            pred_loss, math.exp(pred_loss)))
        print('=' * 89)


if __name__ == '__main__':
    main()
