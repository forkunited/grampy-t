import numpy as np
import itertools
import torch
import torch.nn as nn
from torch.autograd import Variable
from linear import DataParameter, LRLoss

# This is for sparse layers
# Taken from https://discuss.pytorch.org/t/custom-connections-in-neural-network-layers/3027/5
def zero_grad(self, grad_input, grad_output):
    if self.on_gpu():
       self._mask = self._mask.cuda()
    return grad_input[0], grad_input[1], grad_input[2] * self._mask

class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, mask):
        super(MaskedLinear, self).__init__()
        self._linear = nn.Linear(in_features, out_features)
        #self._linear.weight = nn.Parameter(torch.zeros(out_features, in_features))
        #self._linear.bias = nn.Parameter(torch.zeros(out_features))
        self._mask = mask
        self._handle = self.register_backward_hook(zero_grad)  # to make sure gradients won't propagate

    def forward(self, input):
        if self.on_gpu():
            self._mask = self._mask.cuda()
        self._linear.weight.data *= self._mask.data.transpose(0,1)
        return self._linear(input)

    def get_weights(self):
        self._linear.weight.data *= self._mask.data.transpose(0,1)
        return self._linear.weight

    def get_bias(self):
        return self._linear.bias

    def get_mask(self):
        return self._mask

    def set_weights(self, weights, biases, new_weights_mask):
        self._linear.weight.data *= (1.0 - new_weights_mask.unsqueeze(1))
        self._linear.weight.data += new_weights_mask.unsqueeze(1) * weights.transpose(0,1).data
        self._linear.bias.data *= (1.0 - new_weights_mask)
        self._linear.bias.data += new_weights_mask * biases.data

    def on_gpu(self):
        return next(self.parameters()).is_cuda

class ArchitectureGrammar:
    NONE = "None"
    LAYER = "Layer"
    TREE = "Tree"

class FeatureNode:
    def __init__(self, children=[], weights=None, bias=0.0, feature_name=None):
        if len(children) > 0 and (feature_name is not None or len(children) != len(weights)):
            raise ValueError
        self._children = children
        self._weights = weights
        self._bias = bias
        self._feature_name = feature_name

    def get_max_value(self):
        if len(self._children) == 0:
            return 1.0
        child_values = self.get_maximizing_child_values()
        return np.dot(child_values, self._weights) + self._bias

    def get_min_value(self):
        if len(self._children) == 0:
            return 0.0
        child_values = self.get_minimizing_child_values()
        return np.dot(child_values, self._weights) + self._bias

    def get_maximizing_child_values(self):
        child_values = np.zeros(len(self._children))
        for i in range(len(self._children)):
            if self._weights[i] < 0:
                child_values[i] = self._children[i].get_min_value()
            else:
                child_values[i] = self._children[i].get_max_value()
        return child_values

    def get_minimizing_child_values(self):
        child_values = np.zeros(len(self._children))
        for i in range(len(self._children)):
            if self._weights[i] < 0:
                child_values[i] = self._children[i].get_max_value()
            else:
                child_values[i] = self._children[i].get_min_value()
        return child_values

    # Compute the operator based on the smallest number of children that must be "on"
    # (after negation) for this unit to be "on" on average.  If more than
    # half must be on, return '&" and the number of children necessary.  Else return
    # '|' and the number of children necessary
    def get_weighted_operator(self):
        if len(self._children) == 0:
            return None

        min_child_vals = self.get_minimizing_child_values()
        max_child_vals = self.get_maximizing_child_values()
        indices = range(len(self._children))
        for i in range(1, len(self._children)):
            subsets_i = list(itertools.combinations(indices, i)) # All subsets of child indices of size i
            indicators_i = np.zeros((len(subsets_i), len(self._children)))
            for s in range(len(subsets_i)):
                indicators_i[s,list(subsets_i[s])] = 1.0
            avg_i = np.sum(self._bias + np.dot(indicators_i*max_child_vals, self._weights) + np.dot((1.0-indicators_i)*min_child_vals, self._weights))/len(subsets_i)
            if avg_i > 0.0:
                op = '|'
                if i > len(self._children) / 2:
                    op = '&'
                return op, i

        return '&', len(self._children)

    def get_str(self, weighted=False):
        if self._feature_name is not None:
            return self._feature_name
        if weighted:
            s = "("
            for i in range(len(self._children)):
                s += self._children[i].get_str(weighted=True) + ": " + str(self._weights[i]) + ", "
            s += "bias: " + str(self._bias) + ")"
            return s
        else:
            op, n = self.get_weighted_operator()
            s = "("
            op_str = " " + op + " "

            child_strs = []
            for i in range(len(self._children)):
                child_str = self._children[i].get_str()
                if self._weights[i] == 0:
                    child_strs.append("z." + child_str)
                elif self._weights[i] < 0:
                    child_strs.append("-" + child_str)
                else:
                    child_strs.append(child_str)

            s += op_str.join(child_strs)
            s += ")"
            return s


class GrammarType:
    NONE = "None"
    SELECTING = "Selecting"
    RANDOM = "Random"

class GrammarModel(nn.Module):
    def __init__(self, name, Ms, F0_size, Fmax_size, R, t, init_params=None, bias=False, arch=None, arch_depth=1, arch_width=1, max_expand_unary=None, max_expand_binary=None, opt=None, reset_opt=False, extend_interval=None, arch_grammar_type=GrammarType.SELECTING):
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
        self._grammar_masks = [] # Imposes grammar on the structure
        self._arch_nl = nn.Tanh()

        self._output_size = 1
        self._input_size = self._Fmax_size

        arch_layers = []
        if arch == ArchitectureGrammar.LAYER:
            self._input_size += self._Fmax_size*arch_depth*arch_width
            arch_layers.append(nn.Linear(self._Fmax_size, self._Fmax_size*arch_width))
            # This seems like a maybe bad idea here
            #self._arch_layers[-1].weight = nn.Parameter(torch.zeros(self._Fmax_size*arch_width, self._Fmax_size))
            self._grammar_masks.append(Variable(torch.zeros(1, self._Fmax_size)))
            for i in range(1, arch_depth):
                arch_layers.append(nn.Linear(self._Fmax_size*arch_width, self._Fmax_size*arch_width))
                # This seems like maybe a bad idea here
                #self._arch_layers[-1].weight = nn.Parameter(torch.zeros(self._Fmax_size*arch_width, self._Fmax_size*arch_width))
                self._grammar_masks.append(torch.zeros(1, self._Fmax_size*arch_width).long())
        elif arch == ArchitectureGrammar.TREE:
            self._input_size = self._Fmax_size
            for i in range(arch_depth):
                next_layer_size = arch_width*self._input_size*(self._input_size-1)/2
                mask = self._make_tree_structure_mask(self._input_size)
                arch_layers.append(MaskedLinear(self._input_size, next_layer_size, mask))
                self._grammar_masks.append(torch.zeros(1, next_layer_size).long())
                self._input_size += next_layer_size

        self._arch_layers = nn.ModuleList(arch_layers)
        self._linear = nn.Linear(self._input_size, self._output_size, bias=bias)
        if init_params is not None:
            self._linear.weight = nn.Parameter(init_params.unsqueeze(0))
        else:
            self._linear.weight = nn.Parameter(torch.zeros(1,self._input_size))

        self._max_expand_unary = max_expand_unary
        self._max_expand_binary = max_expand_binary
        self._opt = opt
        self._reset_opt = reset_opt
        self._extend_interval = extend_interval
        self._arch_grammar_type = arch_grammar_type

        self._input_padding = None
        self._expanded_unary = set([])
        self._expanded_binary = set([])
        self._expanded_arch = torch.zeros(self._input_size)

    def _make_tree_structure_mask(self, input_size):
        output_size = input_size*(input_size-1)/2
        mask = torch.zeros(input_size, output_size)

        mask_part = torch.zeros(input_size, input_size-1)
        mask_part[0] = 1.0
        mask_part[1:,:] = torch.eye(input_size-1)

        part_left = 0
        for i in range(input_size-1):
            mask_part_i = mask_part[0:(input_size-i), 0:(input_size-1-i)]
            mask[i:input_size,part_left:(part_left+input_size-1-i)] = mask_part_i
            part_left += input_size-1-i

        return Variable(mask.repeat(1,self._arch_width))

    def set_optimizer(self, opt):
        self._opt = opt

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

        expand_f_unary_unfiltered = expand_f_unary
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

    def _expand_architecture(self, indices):
        if len(indices.size()) == 0:
            return 0

        if self._arch_grammar_type == GrammarType.RANDOM:
            indices = torch.from_numpy(np.random.randint(0, high=self._input_size, size=indices.size(0)))

        added = 0
        if self._arch == ArchitectureGrammar.LAYER:
            indices_0 = indices[torch.nonzero(indices < self._Fmax_size).squeeze()]
            self._grammar_masks[0][indices_0] = 1.0
            for i in range(1, len(self._grammar_masks)):
                # For layer 1...i...k:
                # indices in self._Fmax_size + (i-1)*(self._Fmax_size*arch_width) up to self._Fmax_size + i*(self._Fmax_size*arch_width)
                indices_i = indices[torch.nonzero(indices < self._Fmax_size+i*self._Fmax_size*self._arch_width).squeeze()]
                indices_i = indices_i[torch.nonzero(indices_i >= self._Fmax_size+(i-1)*self._Fmax_size*self._arch_width).squeeze()]
                indices_i -= self._Fmax_size+(i-1)*self._Fmax_size*self._arch_width
                self._grammar_masks[i][indices_i] = 1.0
            added = None # FIXME
        elif self._arch == ArchitectureGrammar.TREE:
            layer_max = self._Fmax_size
            for i in range(len(self._grammar_masks)):
                i_nz = torch.nonzero(indices < layer_max).squeeze()
                if len(i_nz.size()) == 0:
                    layer_max += self._grammar_masks[i].size(1)
                    continue
                indices_i = indices[i_nz]
                mask_i = torch.zeros(layer_max)
                mask_i = self._expanded_arch[0:layer_max]
                mask_i[indices_i] = 1.0
                pairs_mask_i = torch.ger(mask_i, mask_i)
                pairs_mask_i = pairs_mask_i[torch.triu(torch.ones(layer_max,layer_max),diagonal=1) == 1].long()
                if self.on_gpu():
                    pairs_mask_i = pairs_mask_i.cuda()
                new_pairs_mask_i = (pairs_mask_i-self._grammar_masks[i][0]) > 0

                #init_tree_weights = self._linear.weight.data[0,0:layer_max].clone().unsqueeze(1).repeat(1,pairs_mask_i.size(0))
                #init_tree_weights *= self._arch_layers[i].get_mask().data
                #self._arch_layers[i].set_weights(Variable(init_tree_weights), Variable(self._linear.bias.data.clone().repeat(pairs_mask_i.size(0))), new_pairs_mask_i.float())

                added += torch.sum(new_pairs_mask_i)
                self._grammar_masks[i][0] = self._grammar_masks[i][0] | pairs_mask_i
                self._expanded_arch[indices_i] = 1.0
                layer_max += self._grammar_masks[i].size(1)

        return added

    def _extend_model(self, data_parameters):
        if not self.training:
            return 0

        DF = self._Ms[0][data_parameters[DataParameter.INPUT]]
        F = DF.get_feature_set()
        if self._arch is None and F.get_size() == self._Fmax_size:
            return 0

        u_i, b_i = self._get_expanding_feature_indices()
        added = 0
        if self._arch_grammar_type != GrammarType.NONE:
            added += self._expand_architecture(u_i)

        if F.get_size() == self._Fmax_size:
            return added

        u_f, b_f = self._get_expanding_feature_tokens(data_parameters, u_i, b_i)
        added += self._expand_features(data_parameters, u_f, b_f)
        return added

    def _apply_arch_layer(self, layer_i, all_output, last_output):
        z = None
        if self._arch == ArchitectureGrammar.LAYER:
            z = self._arch_layers[layer_i](last_output)
        elif self._arch == ArchitectureGrammar.TREE:
            z = self._arch_layers[layer_i](all_output)

        if self._arch_grammar_type != GrammarType.NONE:
            z_masked = z
        else:
            if self.on_gpu():
                self._grammar_masks[layer_i] = self._grammar_masks[layer_i].cuda()
            z_masked = Variable(self._grammar_masks[layer_i].float()) * z

        return self._arch_nl(z_masked)

    def forward(self, input):
        if self._arch is None:
            input = self._make_padded_input(input)
            return self._linear(input)
        else:
            linear_input = self._make_padded_input(input)
            next_output = linear_input
            for i in range(self._arch_depth):
                next_output = self._apply_arch_layer(i, linear_input, next_output)
                linear_input = torch.cat((linear_input, next_output), dim=1)

            return self._linear(linear_input)

    def forward_batch(self, batch, data_parameters):
        if self.training and (self._extend_interval is None or self._opt is None
            or self._opt.get_step() % self._extend_interval == 0):
            added = self._extend_model(data_parameters)
            if added > 0:
                if self._opt is not None and self._reset_opt:
                    print "Added " + str(added) + " features... resetting optimizer"
                    mask = torch.zeros(self._linear.weight.data.size(1)) # Was ones
                    if self.on_gpu():
                        mask = mask.cuda()
                        #mask[indices] = 0.0
                    self._linear.weight.data *= mask
                    self._linear.bias.data *= 0.0
                    self._opt.reset()

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

    def get_nonzero_tree_features(self, feature_set):
        if self._arch != ArchitectureGrammar.TREE:
            raise ValueError

        features = []
        weights = []
        w_indices = torch.nonzero(self._linear.weight.data.squeeze()).squeeze()
        cur_composite_layer = 0
        next_layer_index = self._Fmax_size + self._grammar_masks[0].size(1)

        for index in range(self._linear.weight.data.size(1)):
            feat = None
            if index < self._Fmax_size:
                feat = FeatureNode(feature_name=str(feature_set.get_feature_token(index)))
            else:
                while index >= next_layer_index:
                    cur_composite_layer += 1
                    next_layer_index += self._grammar_masks[cur_composite_layer].size(1)
                cur_layer_size = self._grammar_masks[cur_composite_layer].size(1)
                layer = self._arch_layers[cur_composite_layer]
                index_in_layer = index - (next_layer_index - cur_layer_size)
                f_weights = layer.get_weights().data[index_in_layer].squeeze()
                layer_child_indices = torch.nonzero(f_weights).squeeze()

                if len(layer_child_indices.size()) == 0:
                    bias = layer.get_bias().data[index_in_layer]
                    feat = FeatureNode(feature_name="B(" + str(bias) + ")")
                else:
                    if layer_child_indices.size(0) > 2:
                        raise ValueError

                    children = [features[layer_child_indices[0]], features[layer_child_indices[1]]]
                    children_weights = np.array([f_weights[layer_child_indices[0]], f_weights[layer_child_indices[1]]])
                    bias = layer.get_bias().data[index_in_layer]
                    feat = FeatureNode(children=children, weights=children_weights, bias=bias)

            features.append(feat)

        nonzero_indices = torch.nonzero(self._linear.weight.data.squeeze()).squeeze()
        if len(nonzero_indices.size()) == 0:
            return [], []

        nonzero_features = [features[nonzero_indices[i]] for i in range(nonzero_indices.size(0))]
        nonzero_weights = [self._linear.weight.data[0,nonzero_indices[i]] for i in range(nonzero_indices.size(0))]

        return nonzero_features, nonzero_weights

class LinearGrammarRegression(GrammarModel):
    def __init__(self, name, Ms, F0_size, Fmax_size, R, t, arch=None, arch_depth=1, arch_width=1, init_params=None, bias=False, max_expand_unary=None, max_expand_binary=None, opt=None, reset_opt=False, extend_interval=None, arch_grammar_type=GrammarType.SELECTING):
        super(LinearGrammarRegression, self).__init__(name, Ms, F0_size, Fmax_size, R, t, arch=arch, arch_depth=arch_depth, arch_width=arch_width, init_params=init_params, bias=bias, max_expand_unary=max_expand_unary, max_expand_binary=max_expand_binary, opt=opt, reset_opt=reset_opt, extend_interval=extend_interval, arch_grammar_type=arch_grammar_type)
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
    def __init__(self, name, Ms, F0_size, Fmax_size, R, t, arch=None, arch_depth=1, arch_width=1, init_params=None, bias=False, max_expand_unary=None, max_expand_binary=None, opt=None, reset_opt=False, extend_interval=None, arch_grammar_type=GrammarType.SELECTING):
        super(LogisticGrammarRegression, self).__init__(name, Ms, F0_size, Fmax_size, R, t, arch=arch, arch_depth=arch_depth, arch_width=arch_width, init_params=init_params, bias=bias, max_expand_unary=max_expand_unary, max_expand_binary=max_expand_binary, opt=opt, reset_opt=reset_opt, extend_interval=extend_interval, arch_grammar_type=arch_grammar_type)
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
