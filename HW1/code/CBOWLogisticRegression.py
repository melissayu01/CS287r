import torch
import torch.nn as nn
from torch.autograd import Variable
import utils


class LogisticRegression(nn.Module):
    def __init__(self, input_size, n_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, n_classes)

    def forward(self, x):
        out = self.linear(x)
        return out

def main(n_epochs, learning_rate):
    # load data
    train_iter, val_iter, test_iter, text_field, label_field = utils.load_SST(use_embeddings=True)

    # initialize classifier
    input_size = 300
    n_classes = len(label_field.vocab) - 1
    model = nn.Sequential(
        nn.Linear(input_size, n_classes),
        nn.Linear(n_classes, n_classes),
        nn.Linear(n_classes, n_classes),
    )

    # model = LogisticRegression(input_size, n_classes)

    # loss and optimizer (softmax is internally computed)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    print("Training model...")
    for epoch in range(n_epochs):
        for i, batch in enumerate(train_iter):
            X = utils.skip_gram_embeddings(batch, text_field)
            y = batch.label - 1

            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        print ('Epoch: [%d/%d], Loss: %.4f' % (epoch + 1, n_epochs, loss.data[0]))

    print("Testing model...")
    n, n_corr = 0, 0
    upload = []
    for i, batch in enumerate(test_iter):
        X = utils.skip_gram_embeddings(batch, text_field)
        outputs = model(X)
        _, y_pred = torch.max(outputs.data, 1)
        y = batch.label.data - 1

        n += y.size(0)
        n_corr += (y_pred == y).sum()
        upload += (y_pred + 1).tolist()

    # write predictions to file
    print('Writing predictions to file...')
    with open("predictions_cbow_2.txt", "w") as f:
        f.write('Id,Cat\n')
        for i, u in enumerate(upload):
            f.write('{},{}\n'.format(i, u))

    return n_corr / n

if __name__ == '__main__':
    n_epochs = 100
    learning_rate = 0.05
    print('>>> n_epochs = {}, learning_rate = {}'.format(n_epochs, learning_rate))
    print('acc = ', main(n_epochs, learning_rate))
