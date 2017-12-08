import unittest
import numpy as np
import nltk
import torch

import mung.fol.data as data
import mung.feature as feat
import mung.rule as rule

from gram.model.linear import LinearRegression
from gram.model.grammar import LinearGrammarRegression, ArchitectureGrammar
from mung.torch_ext.eval import Loss
from mung.torch_ext.learn import OptimizerType, Trainer
from mung.util.log import Logger
from mung.fol.feature_top import FeatureTopType
from mung.fol.feature_form_indicator import FeatureFormIndicatorType
from gram.model.linear import LinearRegression, DataParameter
from gram.model.grammar import LinearGrammarRegression, LogisticGrammarRegression, GrammarType

TRAINING_ITERATIONS = 7000 #3000 #1500
DATA_SIZE = 5000
LOG_INTERVAL = 250
EXTEND_INTERVAL = 50
BATCH_SIZE = 50 #100
GRAMMAR_T = 0.0 
L1_C = 0.01 #1 #0.5
LEARNING_RATE = 1.0
SEED = 4
RESET_OPTIMIZER = True
TRUE_STD = 0.005
GRAMMAR_TYPE = GrammarType.NONE
ONLY_CONJUNCTIONS = True
GPU = False

if GPU:
   torch.cuda.manual_seed(SEED)

torch.manual_seed(SEED)
np.random.seed(SEED)


class TestFOLArchitecture(unittest.TestCase):

    def _test_arch(self, arch_type, lr=1.0, l1_C=0.1):
        properties_n = 15
        num_complex_forms = 5
        max_form_depth = 2
        domain = ["0"]

        data_parameters = DataParameter("input", "output")

        properties = []
        F_relevant = feat.FeatureSet()
        F_0 = feat.FeatureSet()
        w = []
        R = rule.RuleSet()

        feature = FeatureTopType("T")
        F_relevant.add_feature_type(feature)
        w.append(-1.0)

        for i in range(properties_n):
            prop = "P" + str(i)
            properties.append(prop)

            form = data.OpenFormula(domain, prop + "(x)", ["x"])
            feature = FeatureFormIndicatorType(prop, form)

            F_relevant.add_feature_type(feature)
            F_0.add_feature_type(feature)
            w.append(0.0) #.1*2.0**i)

        for i in range(num_complex_forms):
            random_exp_str = self._make_random_proposition("x", properties_n, max_form_depth)
            form = data.OpenFormula(domain, random_exp_str, ["x"])
            init_size = F_relevant.get_size()
            F_relevant.add_feature_type(FeatureFormIndicatorType("C"+str(i), form))
            if F_relevant.get_size() != init_size:
                w.append(1.0*(i+1)) #w.append(.1*4.0*i) #w.append(10.0**(i+1)) #w.append(.1*4.0**i)

        w_params = torch.FloatTensor(w)
        model_true = LinearRegression("linear_true", w_params.size(0), init_params=w_params, std=TRUE_STD)

        #if GPU:
        #    model_true = model_true.cuda()

        def label_fn(d):
            D = data.DataSet(data=[d])
            fmat = feat.DataFeatureMatrix(D, F_relevant)
            return model_true.predict(dict({ data_parameters[DataParameter.INPUT] : fmat.get_batch(0,1) }), data_parameters, rand=True)[0,0].data[0]
        def get_label_as_array(d):
            return np.array([d.get("label")])

        f_label = feat.FeatureMatrixType("label", get_label_as_array, 1)
        F_label = feat.FeatureSet()
        F_label.add_feature_type(f_label)

        D_unfeat = data.DataSet.make_random_labeled(DATA_SIZE, domain, properties, [], label_fn)
        D_unfeat_parts = D_unfeat.split([0.8, 0.2])
        D_unfeat_train = D_unfeat_parts[0]
        D_unfeat_dev = D_unfeat_parts[1]

        M_train_x = feat.DataFeatureMatrix(D_unfeat_train, F_relevant)
        M_train_x_0 = feat.DataFeatureMatrix(D_unfeat_train, F_0)
        M_train_y = feat.DataFeatureMatrix(D_unfeat_train, F_label)
        D_train_relevant = feat.MultiviewDataSet(data=D_unfeat_train, dfmats={ data_parameters[DataParameter.INPUT] : M_train_x, data_parameters[DataParameter.OUTPUT] : M_train_y })
        D_train_0 = feat.MultiviewDataSet(data=D_unfeat_train, dfmats={ data_parameters[DataParameter.INPUT] : M_train_x_0, data_parameters[DataParameter.OUTPUT] : M_train_y })

        M_dev_x = feat.DataFeatureMatrix(D_unfeat_dev, F_relevant)
        M_dev_x_0 = feat.DataFeatureMatrix(D_unfeat_dev, F_0)
        M_dev_y = feat.DataFeatureMatrix(D_unfeat_dev, F_label)
        D_dev_relevant = feat.MultiviewDataSet(data=D_unfeat_dev, dfmats={ data_parameters[DataParameter.INPUT] : M_dev_x, data_parameters[DataParameter.OUTPUT] : M_dev_y })
        D_dev_0 = feat.MultiviewDataSet(data=D_unfeat_dev, dfmats={ data_parameters[DataParameter.INPUT] : M_dev_x_0, data_parameters[DataParameter.OUTPUT] : M_dev_y })

        modell1 = None
        D_train = D_train_0
        D_dev = D_dev_0
        if arch_type == ArchitectureGrammar.LAYER:
            modell1 = LinearGrammarRegression("arch_layer_gramression", [D_train_0, D_dev_0], F_0.get_size(), F_0.get_size(), R, GRAMMAR_T, bias=True, arch=arch_type, arch_depth=max_form_depth, arch_width=1, \
                                              reset_opt=RESET_OPTIMIZER, extend_interval=EXTEND_INTERVAL, arch_grammar_type=GRAMMAR_TYPE)
            D_train = D_train_0
            D_dev = D_dev_0
        elif arch_type == ArchitectureGrammar.TREE:
            modell1 = LinearGrammarRegression("arch_tree_gramression", [D_train_0, D_dev_0], F_0.get_size(), F_0.get_size(), R, GRAMMAR_T, bias=True, arch=arch_type, arch_depth=max_form_depth, arch_width=1, \
                                              reset_opt=RESET_OPTIMIZER, extend_interval=EXTEND_INTERVAL, arch_grammar_type=GRAMMAR_TYPE)
            D_train = D_train_0
            D_dev = D_dev_0

        logger = Logger()
        loss_criterion = modell1.get_loss_criterion()

        if GPU:
            modell1 = modell1.cuda()
            loss_criterion = loss_criterion.cuda()

        dev_loss = Loss("loss", D_dev, data_parameters, loss_criterion)
        trainer = Trainer(data_parameters, loss_criterion, logger, \
            dev_loss, other_evaluations=[])
        modell1, best_place_holder, best_iteration = trainer.train(modell1, D_train, TRAINING_ITERATIONS, \
            batch_size=BATCH_SIZE, optimizer_type=OptimizerType.ADAGRAD_MUNG, lr=lr, \
            grad_clip=5.0, log_interval=LOG_INTERVAL, best_part_fn=lambda m : torch.ones(1), l1_C=l1_C)

        print "True model"
        w_true = model_true.get_weights()
        for i in range(F_relevant.get_size()):
            print str(F_relevant.get_feature_token(i)) + "\t" + str(w_true[i])
        print "\n"

        print "l1 model"
        w_model = modell1.get_weights()
        F_train = D_train[data_parameters[DataParameter.INPUT]].get_feature_set()
        print "Bias: " + str(modell1.get_bias())
        if arch_type == ArchitectureGrammar.LAYER:
            for i in range(F_train.get_size()):
                print str(F_train.get_feature_token(i)) + "\t" + str(w_model[i])
            print "\n"
        else:
            nz_feats, nz_weights = modell1.get_nonzero_tree_features(F_train)
            for i in range(len(nz_feats)):
                f = nz_feats[i]
                print f.get_str(weighted=False) + "\t" + str(nz_weights[i])

    def _make_random_proposition(self, var, num_properties, max_form_depth):
        if ONLY_CONJUNCTIONS:
            return self._make_random_conjunction(var, num_properties, 2**max_form_depth)
        else:
            return self._make_random_proposition_helper(var, num_properties, max_form_depth, 0)

    def _make_random_proposition_helper(self, var, num_properties, max_form_depth, cur_depth):
        connective = np.random.randint(0,2)
        if connective == 0 or cur_depth == max_form_depth:
            neg = np.random.randint(0,2)
            if neg == 0:
                return "-" + self._make_random_property(var, num_properties)
            else:
                return self._make_random_property(var, num_properties)
        else:
            first_prop = self._make_random_proposition_helper(var, num_properties, max_form_depth, cur_depth+1)
            second_prop = self._make_random_proposition_helper(var, num_properties, max_form_depth, cur_depth+1)
            connective_type = np.random.randint(0,2)
            if connective_type == 0:
                return "(" + first_prop + " & " + second_prop + ")" # FIXME
            else:
                return "(" + first_prop + " & " + second_prop + ")"

    def _make_random_conjunction(self, var, num_properties, max_conjuncts, min_conjuncts=2):
        conj_n = np.random.randint(min_conjuncts, max_conjuncts+1)
        conj_indices = np.random.choice(np.arange(0, num_properties), size=conj_n, replace=False)
        negs = ["-" if np.random.randint(0,2) == 1 else "" for i in range(conj_n)]
        conjs = [negs[i] + "P" + str(conj_indices[i]) + "(" + var + ")" for i in range(conj_n)]
        return " & ".join(conjs)

    def _make_random_property(self, var, max_property_n):
        return "P" + str(np.random.randint(0,high=max_property_n)) + "(" + var + ")"

    def test_tree_architecture(self):
        self._test_arch(ArchitectureGrammar.TREE, lr=LEARNING_RATE, l1_C=L1_C)

