import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

import math
import os

import process
import rnn


class Config(object):

    vocab_size = len(process.get_vocab()) + 1 # num of tokens
    embed_size = 50 # size of input x (token)
    hid_size = 50 # size of hidden state
    num_layers = 1 # num of lstm cells
    lstm_drop = 0.5 # dropout rate in lstm
    fc_drop = 0.5 # dropout rate in fc
    lr = 1.0 # learning rate
    T = 10 # max epoch
    batch_size = 32 # batch size
    clip = 5 # max grad norm clipping
    log_interval = 50
    save_path = "./saver/lstm_res" # path to save the model
    debug = False # debug mode?
    L2 = 0.001 # L2 penalty in optimizer


class Data(object):

    def __init__(self, path, vocab_size, batch_size, debug=False):
        self.batch_size = batch_size
        self.vocab_size = vocab_size

        data = process.load_raw_data(path)
        self.data_size = len(data)
        if debug:
            self.data_size = 1024

        self.limit = 100 # max sequence length
        self.data_x = []
        self.data_y = []
        for i in range(self.data_size):
            seq_len = len(data[i][0])
            if seq_len > 0 and seq_len < self.limit:
                self.data_x.append(data[i][0])
                self.data_y.append(data[i][1])

        self.data_size = len(self.data_x) / batch_size * batch_size
        self.data_x = self.data_x[:self.data_size]
        self.data_y = self.data_y[:self.data_size]
        print("{} data loaded.".format(self.data_size), 
                1.0 * sum(self.data_y) / len(self.data_y))
        #self.data_y = [[1,0] if i == 0 else [0,1] for i in self.data_y]
        del data

    def data_iterator(self):
        batches = int(self.data_size/self.batch_size)
        for i in range(batches):
            idx = i * self.batch_size
            x = self.data_x[idx:idx+self.batch_size]
            seq_len = [len(i) for i in x]
            x = process.padding(x, self.vocab_size-1, self.limit)
            y = self.data_y[idx:idx+self.batch_size]
            yield (x, y, seq_len)


config = Config()

# load training data and validation data
encoded_train = Data("../data/train.p", config.vocab_size, 
        config.batch_size, config.debug)
encoded_valid = Data("../data/valid.p", config.vocab_size, 
        config.batch_size, config.debug)

if os.path.exists(config.save_path) and False:
    model = torch.load(config.save_path)
else:
    #model = rnn.RNNvanilla(config.vocab_size, config.embed_size, config.hid_size)
    #model = rnn.RNNbn(config.vocab_size, config.embed_size, config.hid_size, 
    #        config.lstm_drop, config.fc_drop)
    model = rnn.RNNres(config.vocab_size, config.embed_size, config.hid_size, 
            config.lstm_drop, config.fc_drop)

#criterion = nn.CrossEntropyLoss() #nn.MSELoss()

lr = config.lr
opt = optim.Adam(model.parameters(), weight_decay=config.L2)


def clip_gradient(model, clip):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        if p.grad:
            modulenorm = p.grad.data.norm()
            totalnorm += modulenorm ** 2

    totalnorm = math.sqrt(totalnorm)

    if totalnorm > clip:
        for p in model.parameters():
            if p.grad:
                p.grad.data = p.grad.data / totalnorm * clip
    #return min(1, clip / (totalnorm + 1e-6))


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def acc(output, target):
    pred = output.data.max(1)[1] # get the index of the max log-probability
    #print("predict | true label")
    #print(pred.view(1, -1).numpy())
    #print(target.data.view(1, -1).numpy())
    correct = pred.eq(target.data).cpu().sum()
    return 1. * correct / len(output)
    """
    #print(output, target)
    m = (output * (2 * target - 1) > 0).float()
    #print(m)
    return 1. * torch.sum(m) / len(m)
    """


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    total_acc = 0.
    ntokens = config.vocab_size
    hidden = model.init_hidden(config.batch_size)
    for i, (x, y, seq_len) in enumerate(data_source.data_iterator()):
        data = Variable(torch.Tensor(x), volatile=True)
        target = Variable(torch.LongTensor(y))
        output, hidden = model(data, hidden, seq_len)
        total_loss += len(data) * F.nll_loss(output, target).data[0]
        #criterion(output, target).data
        total_acc += len(data) * acc(output, target)
        hidden = repackage_hidden(hidden)
    return (total_loss / data_source.data_size, 
            total_acc / data_source.data_size)

def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    total_acc = 0.
    ntokens = config.vocab_size
    hidden = model.init_hidden(config.batch_size)
    for batch, (x, y, seq_len) in enumerate(encoded_train.data_iterator()):
        data = Variable(torch.Tensor(x), volatile=False)
        target = Variable(torch.LongTensor(y))
        #opt.zero_grad()
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden, seq_len)
        loss = F.nll_loss(output, target) #criterion(output, target)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        #torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        #clipped_lr = lr * clip_gradient(model, config.clip)
        clip_gradient(model, config.clip)

        #for p in model.parameters():
        #    if p.grad:
        #        p.data.add_(-clipped_lr, p.grad.data)

        opt.step()

        total_loss += loss.data * len(data)
        total_acc += len(data) * acc(output, target)

        if batch % config.log_interval == 0 and batch > 0:
            data_num = config.batch_size * (batch+1)
            print("[{:5d}/{:5d}] batches\tLoss: {:5.4f}\tAccuracy: {:2.4f}"
                    .format(data_num, encoded_train.data_size, 
                        total_loss.numpy()[0] / data_num, 
                        total_acc / data_num))

    return (total_loss.numpy()[0] / encoded_train.data_size, 
            total_acc / encoded_train.data_size)


prev_val_loss = 99999
best_val_acc = 0
for epoch in range(1, config.T+1):
    train_loss, train_acc = train()
    val_loss, val_acc = evaluate(encoded_valid)
    print('-' * 89)
    print('| end of epoch {:3d} | train loss {:5.4f} | ' 'train acc {:8.4f}'
            .format(epoch, train_loss, train_acc))
    print('| end of epoch {:3d} | valid loss {:5.4f} | ' 'valid acc {:8.4f}'
            .format(epoch, val_loss, val_acc))
    print('-' * 89)

    # Anneal the learning rate.
    if prev_val_loss and val_loss > prev_val_loss:
        lr /= 4

    #if val_acc > best_val_acc:
    if val_loss < prev_val_loss:
        with open(config.save_path, 'wb') as f:
            torch.save(model, f)

    prev_val_loss = val_loss


del encoded_train
del encoded_valid

model = torch.load(config.save_path)

# Run on test data
encoded_test = Data("../data/test.p", config.vocab_size, 
        config.batch_size, config.debug)

test_loss, test_acc = evaluate(encoded_test)
print('=' * 89)
print('| End of training | test loss {:5.4f} | test acc {:8.4f}'.format(
    test_loss, test_acc))
print('=' * 89)

