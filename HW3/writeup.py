import os
import numpy as np
import torch
from torch import optim, nn
from torch.nn import functional as F

import utils
import models

torch.set_printoptions(precision=4)


SAVE_DIR = '.save'
VIS_DIR = 'vis'


def visualize_attn(model, data_loader, use_cuda, save, SRC, TRG):
    model.eval()

    if not os.path.isdir(VIS_DIR):
        os.makedirs(VIS_DIR)

    for b, batch in enumerate(data_loader):
        print(b)
        if b > 10:
            break

        src, trg_input, trg_targets = utils.get_src_and_trgs(
            batch, use_cuda, is_eval=True
        )
        batch_size = src.size(0)
        output, context_or_attn = model(src, trg_input)

        pred = torch.topk(output, k=1, dim=2)[1].squeeze().cpu()

        fname = './{}/attn_{}{}.png'.format(VIS_DIR, save, b)

        i = np.random.randint(batch_size)
        sample_attn = context_or_attn[i].squeeze().data.numpy()
        sample_src = utils.seq_to_text(src[i].squeeze().data, SRC)
        sample_trg = utils.seq_to_text(trg_targets[i].squeeze().data, TRG)
        sample_pred = utils.seq_to_text(pred[i].squeeze(), TRG)

        print('visualizing')
        utils.visualize_attn(
            sample_attn, sample_src, sample_pred, sample_trg, fname)

        # free some memory
        del output, context_or_attn

def main():
    # Cuda
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True

    # Load data + text fields
    print('=' * 89)
    train_iter, val_iter, test_iter, SRC, TRG = utils.load_dataset(
        batch_size=16,
        use_pretrained_emb=False,
        save_dir=SAVE_DIR
    )
    print('=' * 89)

    fname='./{}/unidirectional.pt'.format(SAVE_DIR)
    with open(fname, 'rb') as f:
        model = torch.load(f)
    print(model)

    save = ''
    visualize_attn(model, train_iter, use_cuda, save, SRC, TRG)

if __name__ == '__main__':
    main()
