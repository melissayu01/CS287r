from LSTM import *

# Flag for GPU
USE_CUDA = True

# Standardize randomization across trials
torch.manual_seed(1)

# Prevent memory overflow
torch.backends.cudnn.enabled = False 

# Load in data
TEXT = torchtext.data.Field()
train, val, test = torchtext.datasets.LanguageModelingDataset.splits(
    path="", train="train.txt", validation="valid.txt", test="valid.txt", text_field=TEXT)
TEXT.build_vocab(train)
train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
    (train, val, test), batch_size=500, device=-1, bptt_len=35, repeat=False)
# Build the vocabulary with word embeddings
url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url))

it = iter(train_iter)
batch = next(it) 

# Create model
rnn = LSTM_Lang_Model(vocab_size=len(TEXT.vocab), embedding_size=1000, dropout_rate=0.65, hid_size=1500)
if USE_CUDA:
    rnn = rnn.cuda()

# Train model
rnn.train()
optimizer = optim.Adam(rnn.parameters(), lr=10**-3)
for i in range(30):
    train_iter.init_epoch()
    hidden = None
    for batch in train_iter:
        rnn.zero_grad()
        if USE_CUDA:
            batch.text = batch.text.cuda()
        probs, hidden = rnn(batch.text, hidden)
        hidden = hidden[0].detach(), hidden[1].detach()
        log_probs = torch.log(probs)
        if USE_CUDA:
            log_probs = log_probs.cuda()
        loss = -log_probs.gather(2, batch.target.unsqueeze(2).cuda()).mean()
        print(loss.data[0])
        loss.backward()
        clip_grad_norm(rnn.lstm.parameters(), 10)
        optimizer.step()
    ppl = eval(rnn, test_iter)
    decrease_learning_rate(optimizer, 1.2)
    print("Epoch #{}: {}".format(i, ppl))

# Save model and trained parameters
torch.save(rnn, 'rnn.pt')

# Load model
rnn = torch.load('rnn.pt')

# Collect Kaggle inputs
out = []
for line in open("../data/input.txt"):
    words = line.strip(' _\t\n\r').split()
    word_indexes = [TEXT.vocab.stoi[word] for word in words]
    out.append(Variable(torch.LongTensor(word_indexes)).cuda())

# Generate top 20 predictions for each Kaggle input
rnn.eval()
with open("../predictions/lstm.txt", "w") as fout:
    print("id,word", file=fout)
    for i, vec in enumerate(out, start=1):
        hidden = None
        probs, hidden = rnn(vec.view(-1, 1), hidden)
        p = probs.squeeze()[-1].data.cpu().numpy()
        top20 = sorted(range(len(p)), key=lambda i: p[i], reverse=True)[:20]
        if np.isin(top20, [1]).any():
            raise ValueError()
        preds20 = np.array(TEXT.vocab.itos)[top20]
        print("%d,%s"%(i, " ".join(preds20)), file=fout)
