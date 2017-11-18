import torch
import torch.nn as nn
from torch.autograd import Variable

class DataParameter:
    INPUT = "input"
    OUTPUT = "output"

    def __init__(self, input, output):
        self._input = input
        self._output = output

    def __getitem__(self, key):
        if key == DataParameter.INPUT:
            return self._input
        elif key == DataParameter.OUTPUT:
            return self._output

class LRLoss(nn.Module):
    def __init__(self, size_average=True):
        super(LRLoss, self).__init__()
        self._nllloss = nn.NLLLoss(size_average=size_average)
        self._sigmoid = nn.Sigmoid()

    def forward(self, input, target):
        out = self._sigmoid(input)
        other_out = 1.0 - out
        both_out = torch.log(torch.cat((other_out, out), dim=1))
        return self._nllloss(both_out, target.squeeze().long())

class LinearModel(nn.Module):
    def __init__(self, name, input_size, init_params=None, bias=False):
        super(LinearModel, self).__init__()
        self._name = name

        self._input_size = input_size
        self._output_size = 1
        self._linear = nn.Linear(self._input_size, self._output_size, bias=bias)
        if init_params is not None:
            self._linear.weight = nn.Parameter(init_params.unsqueeze(0))

    def get_name(self):
        return self._name

    def on_gpu(self):
        return next(self.parameters()).is_cuda

    def get_weights(self):
        return list((list(self.parameters()))[0].data.view(-1))

    def get_bias(self):
        return list(self.parameters())[1].data[0]

    def forward(self, input):
        return self._linear(input)

    def forward_batch(self, batch, data_parameters):
        input = Variable(batch[data_parameters[DataParameter.INPUT]])
        if self.on_gpu():
            input = input.cuda()
        return self(input)

    def loss(self, batch, data_parameters, loss_criterion):
        utterance = None
        input = batch[data_parameters[DataParameter.INPUT]]
        output = batch[data_parameters[DataParameter.OUTPUT]]
        if self.on_gpu():
            input = input.cuda()
            output = output.cuda()

        model_out = self.forward_batch(batch, data_parameters)
        return loss_criterion(model_out, Variable(output))

class LinearRegression(LinearModel):
    def __init__(self, name, input_size, init_params=None, bias=False):
        super(LinearRegression, self).__init__(name, input_size, init_params=init_params, bias=bias)
        self._mseloss = nn.MSELoss(size_average=False)

    def predict(self, batch, data_parameters, rand=False):
        if not rand:
            return self.forward_batch(batch, data_parameters)
        else:
            mu = self.forward_batch(batch, data_parameters)
            return torch.normal(mu)

    def default_loss(self, batch, data_parameters):
        return self.loss(batch, data_parameters, self._mseloss)

    def get_loss_criterion(self):
        return self._mseloss

class LogisticRegression(LinearModel):
    def __init__(self, name, input_size, init_params=None, bias=False):
        super(LogisticRegression, self).__init__(name, input_size, init_params=init_params, bias=bias)
        self._lrloss = LRLoss(size_average=False)
        self._sigmoid = nn.Sigmoid()

    def predict(self, batch, data_parameters, rand=False):
        p = self._sigmoid(self.forward_batch(batch, data_parameters))
        if not rand:
            return p > 0.5
        else:
            return torch.bernoulli(p)

    def default_loss(self, batch, data_parameters):
        return self.loss(batch, data_parameters, self._lrloss)

    def get_loss_criterion(self):
        return self._lrloss

