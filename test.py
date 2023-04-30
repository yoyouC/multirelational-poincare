import nltk
import torch
import pickle

train_test_ratio = 0.8
window_size = 5
sents = nltk.corpus.treebank.tagged_sents()

# Data Preparation

## pad the sentence
for sent in sents:
    for _ in range(window_size):
        sent.insert(0, ('<s>', 'START'))
        sent.append(('</s>', 'END'))

## split train and test sets
test_sents = sents[int(len(sents)*train_test_ratio):]
train_sents = sents[:int(len(sents)*train_test_ratio)]

## load pretrained word embeddings and vocab
word_embeddings = torch.load('word_embeddings.pt')
vocab = pickle.load(open('vocab.pkl', 'rb'))

## generate vocab
vocab = set()
for sent in sents:
    for word, tag in sent:
        vocab.add(word)

## generate word2idx and idx2word
word2idx = {}
idx2word = {}
for i, word in enumerate(vocab):
    word2idx[word] = i
    idx2word[i] = word

## generate tag2idx and idx2tag
tag2idx = {}
idx2tag = {}
for i, tag in enumerate(set([tag for sent in sents for word, tag in sent])):
    tag2idx[tag] = i
    idx2tag[i] = tag

## generate train and test data
train_data = []
test_data = []
for sent in train_sents:
    for i in range(window_size, len(sent)-window_size):
        context = [word2idx[word] for word, tag in sent[i-window_size:i+window_size+1]]
        target = sent[i][1]
        train_data.append((context, tag2idx[target]))

for sent in test_sents:
    for i in range(window_size, len(sent)-window_size):
        context = [word2idx[word] for word, tag in sent[i-window_size:i+window_size+1]]
        target = sent[i][1]
        test_data.append((context, tag2idx[target]))

## define model
class CBOW(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, tagset_size):
        super(CBOW, self).__init__()
        self.embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = torch.nn.Linear(embedding_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, tagset_size)
        self.activation = torch.nn.ReLU()
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).sum(dim=1)
        out = self.linear1(embeds)
        out = self.activation(out)
        out = self.linear2(out)
        out = self.softmax(out)
        return out

## define training
def train(model, train_data, test_data, word_embeddings, epochs=10, batch_size=128, learning_rate=0.01):
    loss_function = torch.nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        print('Epoch: {}'.format(epoch))
        train_loss = 0
        for i, (context, target) in enumerate(train_data):
            context = torch.tensor(context)
            target = torch.tensor([target])
            optimizer.zero_grad()
            log_probs = model(context)
            loss = loss_function(log_probs, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if i % 100 == 0:
                print('Train Loss: {}'.format(train_loss/100))
                train_loss = 0
        test_loss = 0
        for i, (context, target) in enumerate(test_data):
            context = torch.tensor(context)
            target = torch.tensor([target])
            log_probs = model(context)
            loss = loss_function(log_probs, target)
            test_loss += loss.item()
        print('Test Loss: {}'.format(test_loss/len(test_data)))