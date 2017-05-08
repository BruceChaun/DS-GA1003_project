import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class RNNvanilla(nn.Module):
    """
    unidiretional, single layer lstm cell
    mean pooling
    fc w/o activation function and dropout
    """

    def __init__(self, ntoken, ninp, nhid):
        super(RNNvanilla, self).__init__()

        self.encoder = nn.Embedding(ntoken, ninp)
        self.lstm = nn.LSTMCell(ninp, nhid)
        self.nhid = nhid

        self.init_weights()

        # fc
        fc_size = [nhid, nhid // 3, 2]
        self.fc1 = nn.Linear(fc_size[0], fc_size[1])
        self.fc2 = nn.Linear(fc_size[1], fc_size[2])

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, seq_len):
        emb = self.encoder(input.transpose(0, 1).long())
        output = []
        hx, cx = hidden
        for t in range(emb.size(0)):
            hx, cx = self.lstm(emb[t], (hx, cx))
            output.append(hx)

        output = torch.stack(output)
        # mean pooling
        stack = []
        for i in range(output.size(1)):
            stack.append(torch.mean(output[:seq_len[i],i,:], 0).squeeze())

        feature = torch.stack(stack)
        fc_data = self.fc1(feature)
        fc_data = self.fc2(fc_data)

        return F.log_softmax(fc_data), (hx, cx)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(bsz, self.nhid).zero_()),
                Variable(weight.new(bsz, self.nhid).zero_()))


def lReLU(x, p=0.01):
    return torch.max(x, x * p)


class RNNbn(nn.Module):
    """
    unidiretional lstm cell + mean pooling
    add leaky ReLU function, batch norm and dropout
    """

    def __init__(self, ntoken, ninp, nhid, lstm_dropout, fc_dropout):
        super(RNNbn, self).__init__()

        self.encoder = nn.Embedding(ntoken, ninp)
        self.lstm = nn.LSTMCell(ninp, nhid)
        self.lstm_drop = nn.Dropout(lstm_dropout)
        self.fc_drop = nn.Dropout(fc_dropout)

        self.nhid = nhid
        self.init_weights()

        # fc
        fc_size = [nhid, nhid // 3, 2]
        self.fc1 = nn.Linear(fc_size[0], fc_size[1])
        self.fc2 = nn.Linear(fc_size[1], fc_size[2])

        self.bn1 = nn.BatchNorm1d(fc_size[1])

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, seq_len):
        emb = self.encoder(input.transpose(0, 1).long())
        emb = self.lstm_drop(emb)
        output = []
        hx, cx = hidden
        hx = self.lstm_drop(hx)
        for t in range(emb.size(0)):
            hx, cx = self.lstm(emb[t], (hx, cx))
            hx = self.lstm_drop(hx)
            output.append(hx)

        output = torch.stack(output)
        # mean pooling
        stack = []
        for i in range(output.size(1)):
            stack.append(torch.mean(output[:seq_len[i],i,:], 0).squeeze())

        feature = torch.stack(stack)

        noise = 0.2 if self.training else 0
        #feature = self.fc_drop(feature)
        fc_data = self.fc1(feature)
        fc_data = lReLU(self.bn1(
            fc_data + Variable(noise * torch.randn(fc_data.size()))))
        fc_data = self.fc_drop(fc_data)
        fc_data = self.fc2(fc_data)

        return F.log_softmax(fc_data), (hx, cx)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(bsz, self.nhid).zero_()),
                Variable(weight.new(bsz, self.nhid).zero_()))


class ResNet(nn.Module):

    def __init__(self, in_size, hid_size):
        super(ResNet, self).__init__()

        self.fc1 = nn.Linear(in_size, hid_size)
        self.fc2 = nn.Linear(hid_size, in_size)
        self.bn = nn.BatchNorm1d(hid_size)

    def forward(self, x, training):
        if training:
            noise = 0.2
        else:
            noise = 0

        residue = self.fc1(F.relu(x + Variable(noise * torch.randn(x.size()))))
        residue = self.bn(F.relu(residue))
        return x + self.fc2(residue)


class RNNres(nn.Module):
    """
    unidiretional lstm cell + mean pooling
    add leaky ReLU function, batch norm, dropout
    add resnet before fc
    """

    def __init__(self, ntoken, ninp, nhid, lstm_dropout, fc_dropout):
        super(RNNres, self).__init__()

        self.encoder = nn.Embedding(ntoken, ninp)
        self.lstm = nn.LSTMCell(ninp, nhid)
        self.lstm_drop = nn.Dropout(lstm_dropout)
        self.fc_drop = nn.Dropout(fc_dropout)

        self.nhid = nhid
        self.init_weights()

        resnet_size = 10
        self.resnet = ResNet(nhid, resnet_size)

        # fc
        fc_size = [nhid, nhid // 3, 2]
        self.fc1 = nn.Linear(fc_size[0], fc_size[1])
        self.fc2 = nn.Linear(fc_size[1], fc_size[2])

        self.bn1 = nn.BatchNorm1d(fc_size[1])

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, seq_len):
        emb = self.encoder(input.transpose(0, 1).long())
        emb = self.lstm_drop(emb)
        output = []
        hx, cx = hidden
        hx = self.lstm_drop(hx)
        for t in range(emb.size(0)):
            hx, cx = self.lstm(emb[t], (hx, cx))
            hx = self.lstm_drop(hx)
            output.append(hx)

        output = torch.stack(output)
        # mean pooling
        stack = []
        for i in range(output.size(1)):
            stack.append(torch.mean(output[:seq_len[i],i,:], 0).squeeze())

        feature = torch.stack(stack)

        noise = 0.2 if self.training else 0
        feature = self.resnet(feature, self.training)
        #feature = self.fc_drop(feature)
        fc_data = self.fc1(feature)
        fc_data = lReLU(self.bn1(
            fc_data + Variable(noise * torch.randn(fc_data.size()))))
        fc_data = self.fc2(self.fc_drop(fc_data))

        return F.log_softmax(fc_data), (hx, cx)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(bsz, self.nhid).zero_()),
                Variable(weight.new(bsz, self.nhid).zero_()))
