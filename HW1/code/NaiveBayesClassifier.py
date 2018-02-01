import numpy as np
import utils

class NaiveBayesClassifier:
    def __init__(self, alpha, beta, n_features):
        # 1 x C vector; dirichlet prior for class distr.
        # C = 2 for binary classification.
        self.alpha = alpha
        self.alpha0 = sum(alpha)

        # 1 x K vector; dirichlet prior for class conditional distr.
        # K = 2 for binary features, otherwise K = max(occurences_of_word_in_text)
        self.beta = beta
        self.beta0 = sum(beta)

        # dimensions of data
        self.C = len(self.alpha) # num classes
        self.K = len(self.beta)  # num possible values for each feature (count)
        self.D = n_features      # num features (size of vocabulary)

        # counts
        self.N = 0
        self.N_c = np.zeros(self.C, dtype=int)
        self.N_cj = np.zeros((self.C, self.D), dtype=int)
        self.N_ckj = np.zeros((self.C, self.K, self.D), dtype=int)

        self.flushed = False

    def fit(self, X, y):
        X = X.astype(int)
        N, _D = X.shape
        self.N += N

        # print("Fitting model")
        for c in range(self.C):
            msk = y == c
            self.N_c[c] += np.sum(msk)
            self.N_cj[c] += np.sum(X[msk], dtype=int, axis=0)
            self.N_ckj[c] += np.apply_along_axis(np.bincount, 0, X[msk], minlength=self.K)

        self.flushed = False

    def predict(self, X):
        X = X.astype(int)

        if not self.flushed:
            # print("Flushing")
            self.pi = np.array([ # class distribution
                np.log(self.N_c[c] + self.alpha[c]) - np.log(self.N + self.alpha0)
                for c in range(self.C)])
            self.mu = np.fromfunction( # log prob of each (class, count, word) tuple
                lambda c, j, k: np.log(self.N_ckj[c, k, j] + self.beta[c]) - np.log(self.N_c[c] + self.beta0),
                (self.C, self.D, self.K), dtype=int)
            self.flushed = True

        # print("Predicting labels")
        p_for_x = lambda x: [ # calculate log probability for x of class c
            self.pi[c] + np.sum([self.mu[c, j, x[j]] for j in range(len(x))])
            for c in range(self.C)]
        ps = np.apply_along_axis(p_for_x, 1, X)
        return np.apply_along_axis(np.argmax, 1, ps) # get predictions

def main(C, K, a, b, binary):
    # load data
    train_iter, val_iter, test_iter, text_field, label_field = utils.load_SST()

    # initialize classifier
    alpha = a * np.ones(C)
    beta = b * np.ones(K)
    n_features = len(text_field.vocab)
    nb = NaiveBayesClassifier(alpha, beta, n_features)

    print("Training model...")
    for i, batch in enumerate(train_iter):
        X = utils.bag_of_words(batch, text_field).data.numpy()
        if binary:
            X = X > 0
        y = batch.label.data.numpy() - 1
        nb.fit(X, y)

    print("Testing model...")
    n, n_corr = 0, 0
    upload = []
    for i, batch in enumerate(test_iter):
        X = utils.bag_of_words(batch, text_field).data.numpy()
        if binary:
            X = X > 0
        y_pred = nb.predict(X)
        y = batch.label.data.numpy() - 1

        n += len(y)
        n_corr += sum(y_pred == y)

        upload += list(y_pred + 1)

    # write predictions to file
    print('Writing predictions to file...')
    with open("predictions.txt", "w") as f:
        f.write('Id,Cat\n')
        for i, u in enumerate(upload):
            f.write('{},{}\n'.format(i, u))

    return n_corr / n

if __name__ == '__main__':
    C = 2
    K = 2
    a = 1
    b = 0.5
    binary = True
    print('>>> alpha = {}, beta = {}, binary features = {}'.format(a, b, binary))
    print('acc = ', main(C, K, a, b, binary))
