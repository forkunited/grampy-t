import torch
import torch.nn as nn
from torch.autograd import Variable
from linear import DataParameter, LRLoss

class ArchitectureGrammar:
    LAYER = 0
    TREE = 1

class GrammarModel(nn.Module):
    def __init__(self, name, Ms, F0_size, Fmax_size, R, t, init_params=None, bias=False, arch=None, arch_depth=1, arch_width=1, max_expand_unary=None, max_expand_binary=None):
        super(GrammarModel, self).__init__()
        self._name = name

        self._Ms = Ms
        self._F0_size = F0_size
        self._Fmax_size = Fmax_size
        self._R = R       
        self._t = t

        self._arch = arch
        self._arch_depth = arch_depth
        self._arch_width = arch_width
        self._arch_layers = []
        self._structure_masks = [] # Imposes tree structure on the architecture
        self._grammar_masks = [] # Imposes grammar on the structure
        self._arch_nl = nn.ReLU()

        self._output_size = 1
        self._input_size = self._Fmax_size

        if arch == ArchitectureGrammar.LAYER:
            self._input_size += self._Fmax_size*arch_depth*arch_width 
            self._arch_layers.append(nn.Linear(self._Fmax_size, self._Fmax_size*arch_width))
            self._arch_layers[-1].weight = nn.Parameter(torch.zeros(self._Fmax_size*arch_width, self._Fmax_size))
            for i in range(1, arch_depth):
                self._arch_layers.append(nn.Linear(self._Fmax_size*arch_width, self._Fmax_size*arch_width))
                self._arch_layers[-1].weight = nn.Parameter(torch.zeros(self._Fmax_size*arch_width, self._Fmax_size*arch_width))
        elif arch == ArchitectureGrammar.TREE:
            self._input_size = self._Fmax_size
            for i in range(arch_depth):
                next_layer_size = arch_width*self._input_size*(self._input_size-1)/2
                self._arch_layers.append(nn.Linear(self._input_size, next_layer_size))
                self._arch_layers[-1].weight = nn.Parameter(torch.zeros(next_layer_size, self._input_size))
                self._input_size += next_layer_size

        self._linear = nn.Linear(self._input_size, self._output_size, bias=bias)
        if init_params is not None:
            self._linear.weight = nn.Parameter(init_params.unsqueeze(0))
        else:
            self._linear.weight = nn.Parameter(torch.zeros(1,self._input_size))

        self._max_expand_unary = max_expand_unary
        self._max_expand_binary = max_expand_binary
        self._input_padding = None
        self._expanded_unary = set([])
        self._expanded_binary = set([])

    def get_name(self):
        return self._name

    def on_gpu(self):
        return next(self.parameters()).is_cuda

    def get_weights(self):
        return list((list(self.parameters()))[0].data.view(-1))

    def get_bias(self):
        return list(self.parameters())[1].data[0]

    def _make_padded_input(self, input):
        if input.size(1) == self._Fmax_size:
            return input
        if self._input_padding is None or self._input_padding.size(0) != input.size(0) or self._input_padding.size(1) != self._Fmax_size - input.size(1):
            self._input_padding = Variable(torch.zeros(input.size(0), self._Fmax_size - input.size(1)))
        return torch.cat((input, self._input_padding), dim=1)

    # FIXME: This function is taken from the old grampy to save time
    # Many of its parts can be vectorized to speed things up if necessary
    # On the other hand, this is probably not the bottleneck, so it's probably fine
    def _get_expanding_feature_indices(self):
        expand_f_unary = []
        
        w_indices = torch.nonzero(torch.abs(self._linear.weight.data.squeeze()) > self._t).squeeze()
        if len(w_indices.size()) > 0:
            for i in range(w_indices.size(0)):
                expand_f_unary.append((w_indices[i], abs(self._linear.weight.data[0,w_indices[i]])))
        
        expand_f_binary = []
        for (i, w_i) in expand_f_unary:
            for (j, w_j) in expand_f_unary:
                expand_f_binary.append((i, j, w_i*w_j))
        
        expand_f_unary_filtered = []
        expand_f_binary_filtered = []
        for (i,w) in expand_f_unary:
            if i not in self._expanded_unary:
                expand_f_unary_filtered.append((i,w))

        for (i,j,w) in expand_f_binary:
            if (i,j) not in self._expanded_binary:
                expand_f_binary_filtered.append((i,j,w))

        expand_f_unary = expand_f_unary_filtered
        expand_f_binary = expand_f_binary_filtered

        expand_f_unary.sort(key=lambda x : -x[1])
        expand_f_binary.sort(key=lambda x : -x[2])

        if self._max_expand_unary is not None:
            expand_f_unary = expand_f_unary[:self._max_expand_unary] 

        if self._max_expand_binary is not None:
            expand_f_binary = expand_f_binary[:self._max_expand_binary]

        u_tensor = torch.zeros(len(expand_f_unary)).long()
        for i in range(len(expand_f_unary)):
            u_tensor[i] = expand_f_unary[i][0]

        b_tensor = torch.zeros(len(expand_f_binary), 2).long()
        for i in range(len(expand_f_binary)):
            b_tensor[i,0] = expand_f_binary[i][0]
            b_tensor[i,1] = expand_f_binary[i][1]

        return u_tensor, b_tensor

    def _get_expanding_feature_tokens(self, data_parameters, unary_indices, binary_indices):
        F = self._Ms[0][data_parameters[DataParameter.INPUT]].get_feature_set()

        unary_f = []
        if len(unary_indices.size()) > 0:
            for i in range(unary_indices.size(0)):
                index = unary_indices[i]
                if index < self._Fmax_size:
                    unary_f.append((index, F.get_feature_token(index)))
            
        binary_f = []
        if len(binary_indices.size()) > 0:
            for i in range(binary_indices.size(0)):
                index_0 = binary_indices[i,0]
                index_1 = binary_indices[i,1]
                if index_0 < self._Fmax_size and index_1 < self._Fmax_size:
                    binary_f.append((index_0, index_1, F.get_feature_token(index_0), F.get_feature_token(index_1)))
        
        return unary_f, binary_f

    def _expand_features(self, data_parameters, unary_tokens, binary_tokens):
        unary_f = []
        for (i, f) in unary_tokens:
            self._expanded_unary.add(i)
            unary_f.append(f)        

        binary_f = []
        for (i, j, f_i, f_j) in binary_tokens:
            self._expanded_binary.add((i,j))
            binary_f.append((f_i,f_j))

        new_f = self._R.apply_unary_binary(unary_f, binary_f)

        new_f_filtered = []
        new_size = 0
        for f in new_f:
            if new_size + f.get_size() + self._F0_size > self._Fmax_size:
                break
            new_size += f.get_size()

        feature_set = self._Ms[0][data_parameters[DataParameter.INPUT]].get_feature_set()
        start_num_feature_types = feature_set.get_num_feature_types()
        start_feature_size = feature_set.get_size()
        self._Ms[0][data_parameters[DataParameter.INPUT]].extend(new_f)
        if len(self._Ms) > 1:
            for i in range(1, len(self._Ms)):
                self._Ms[i][data_parameters[DataParameter.INPUT]].extend(new_f, start_num=start_num_feature_types, start_size=start_feature_size)
        return new_size

    def _extend_model(self, data_parameters):
        if not self.training:
            return

        DF = self._Ms[0][data_parameters[DataParameter.INPUT]]
        F = DF.get_feature_set()
        if F.get_size() == self._Fmax_size:
            return

        u_i, b_i = self._get_expanding_feature_indices()
        u_f, b_f = self._get_expanding_feature_tokens(data_parameters, u_i, b_i)
        added = self._expand_features(data_parameters, u_f, b_f)
        return added

    def forward(self, input):
        input = self._make_padded_input(input)
        return self._linear(input)

    def forward_batch(self, batch, data_parameters):
        self._extend_model(data_parameters)
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

class LinearGrammarRegression(GrammarModel):
    def __init__(self, name, Ms, F0_size, Fmax_size, R, t, arch=None, arch_depth=1, arch_width=1, init_params=None, bias=False, max_expand_unary=None, max_expand_binary=None):
        super(LinearGrammarRegression, self).__init__(name, Ms, F0_size, Fmax_size, R, t, arch=arch, arch_depth=arch_depth, arch_width=arch_width, init_params=init_params, bias=bias, max_expand_unary=max_expand_unary, max_expand_binary=max_expand_binary)
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

class LogisticGrammarRegression(GrammarModel):
    def __init__(self, name, Ms, F0_size, Fmax_size, R, t, arch=None, arch_depth=1, arch_width=1, init_params=None, bias=False, max_expand_unary=None, max_expand_binary=None):
        super(LogisticGrammarRegression, self).__init__(name, Ms, F0_size, Fmax_size, R, t, arch=arch, arch_depth=arch_depth, arch_width=arch_wdith, init_params=init_params, bias=bias, max_expand_unary=max_expand_unary, max_expand_binary=max_expand_binary)
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

