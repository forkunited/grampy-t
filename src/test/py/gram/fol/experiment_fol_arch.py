import sys
import numpy as np
import nltk
import torch

import mung.fol.data as data
import mung.feature as feat
import mung.rule as rule

from gram.model.linear import LinearRegression
from gram.model.grammar import LinearGrammarRegression, ArchitectureGrammar
from mung.torch_ext.eval import Evaluation, Loss, ModelStatistic
from mung.torch_ext.learn import OptimizerType, Trainer
from mung.util.log import Logger
from mung.fol.feature_top import FeatureTopType
from mung.fol.feature_form_indicator import FeatureFormIndicatorType
from gram.model.linear import LinearRegression, DataParameter
from gram.model.grammar import LinearGrammarRegression, LogisticGrammarRegression, GrammarType

TRAINING_ITERATIONS = 20000 #3000 #1500
DATA_SIZE = 5000
LOG_INTERVAL = 250
EXTEND_INTERVAL = 50
BATCH_SIZE = 50 #100
GRAMMAR_T = 0.0
LEARNING_RATE = 1.0
SEED = 4
RESET_OPTIMIZER = True
TRUE_STD = 0.1
ONLY_CONJUNCTIONS = True
PROPERTIES_N = 15
DOMAIN = ["0"]

GPU = bool(int(sys.argv[1]))
RERUNS = int(sys.argv[2])
MAX_FORM_DEPTH = int(sys.argv[3]) # 2
NUM_COMPLEX_FORMS = int(sys.argv[4]) # 5
MODEL_NAME = sys.argv[5]
GRAMMAR_TYPE = sys.argv[6] #GrammarType.NONE
ARCH_TYPE = sys.argv[7] #ArchitectureGrammar.TREE
L1_C = float(sys.argv[8]) #1 #0.5
OUTPUT_RESULTS_PATH = sys.argv[9]
FINAL_OUTPUT_RESULTS_PATH = sys.argv[10]

if GPU:
   torch.cuda.manual_seed(SEED)

torch.manual_seed(SEED)
np.random.seed(SEED)

def _make_random_proposition(var, num_properties, max_form_depth):
    if ONLY_CONJUNCTIONS:
        return _make_random_conjunction(var, num_properties, 2**max_form_depth)
    else:
        return _make_random_proposition_helper(var, num_properties, max_form_depth, 0)

def _make_random_proposition_helper(var, num_properties, max_form_depth, cur_depth):
    connective = np.random.randint(0,2)
    if connective == 0 or cur_depth == max_form_depth:
        neg = np.random.randint(0,2)
        if neg == 0:
            return "-" + _make_random_property(var, num_properties)
        else:
            return _make_random_property(var, num_properties)
    else:
        first_prop = _make_random_proposition_helper(var, num_properties, max_form_depth, cur_depth+1)
        second_prop = _make_random_proposition_helper(var, num_properties, max_form_depth, cur_depth+1)
        connective_type = np.random.randint(0,2)
        if connective_type == 0:
            return "(" + first_prop + " & " + second_prop + ")" # FIXME
        else:
            return "(" + first_prop + " & " + second_prop + ")"

def _make_random_conjunction(var, num_properties, max_conjuncts, min_conjuncts=2):
    conj_n = np.random.randint(min_conjuncts, max_conjuncts+1)
    conj_indices = np.random.choice(np.arange(0, num_properties), size=conj_n, replace=False)
    negs = ["-" if np.random.randint(0,2) == 1 else "" for i in range(conj_n)]
    conjs = [negs[i] + "P" + str(conj_indices[i]) + "(" + var + ")" for i in range(conj_n)]
    return " & ".join(conjs)

def _make_random_property(var, max_property_n):
    return "P" + str(np.random.randint(0,high=max_property_n)) + "(" + var + ")"


data_parameters = DataParameter("input", "output")

properties = []
F_relevant = feat.FeatureSet()
F_0 = feat.FeatureSet()
w = []
R = rule.RuleSet()

feature = FeatureTopType("T")
F_relevant.add_feature_type(feature)
w.append(-1.0)

for i in range(PROPERTIES_N):
    prop = "P" + str(i)
    properties.append(prop)

    form = data.OpenFormula(DOMAIN, prop + "(x)", ["x"])
    feature = FeatureFormIndicatorType(prop, form)

    F_relevant.add_feature_type(feature)
    F_0.add_feature_type(feature)
    w.append(0.0) #.1*2.0**i)

for i in range(NUM_COMPLEX_FORMS):
    random_exp_str = _make_random_proposition("x", PROPERTIES_N, MAX_FORM_DEPTH)
    form = data.OpenFormula(DOMAIN, random_exp_str, ["x"])
    init_size = F_relevant.get_size()
    F_relevant.add_feature_type(FeatureFormIndicatorType("C"+str(i), form))
    if F_relevant.get_size() != init_size:
        w.append(1.0*(i+1)) #w.append(.1*4.0*i) #w.append(10.0**(i+1)) #w.append(.1*4.0**i)

w_params = torch.FloatTensor(w)
model_true = LinearRegression("linear_true", w_params.size(0), init_params=w_params, std=TRUE_STD)

if GPU:
    model_true = model_true.cuda()

def label_fn(d):
    D = data.DataSet(data=[d])
    fmat = feat.DataFeatureMatrix(D, F_relevant)
    return model_true.predict(dict({ data_parameters[DataParameter.INPUT] : fmat.get_batch(0,1) }), data_parameters, rand=True)[0,0].data[0]
def get_label_as_array(d):
    return np.array([d.get("label")])

f_label = feat.FeatureMatrixType("label", get_label_as_array, 1)
F_label = feat.FeatureSet()
F_label.add_feature_type(f_label)

D_unfeat = data.DataSet.make_random_labeled(DATA_SIZE, DOMAIN, properties, [], label_fn)
D_unfeat_parts = D_unfeat.split([0.8, 0.1, 0.1])
D_unfeat_train = D_unfeat_parts[0]
D_unfeat_dev = D_unfeat_parts[1]
D_unfeat_test = D_unfeat_parts[2]

M_train_x = feat.DataFeatureMatrix(D_unfeat_train, F_0)
M_train_y = feat.DataFeatureMatrix(D_unfeat_train, F_label)
D_train = feat.MultiviewDataSet(data=D_unfeat_train, dfmats={ data_parameters[DataParameter.INPUT] : M_train_x, data_parameters[DataParameter.OUTPUT] : M_train_y })

M_dev_x = feat.DataFeatureMatrix(D_unfeat_dev, F_0)
M_dev_y = feat.DataFeatureMatrix(D_unfeat_dev, F_label)
D_dev = feat.MultiviewDataSet(data=D_unfeat_dev, dfmats={ data_parameters[DataParameter.INPUT] : M_dev_x, data_parameters[DataParameter.OUTPUT] : M_dev_y })

M_test_x = feat.DataFeatureMatrix(D_unfeat_test, F_0)
M_test_y = feat.DataFeatureMatrix(D_unfeat_test, F_label)
D_test = feat.MultiviewDataSet(data=D_unfeat_test, dfmats={ data_parameters[DataParameter.INPUT] : M_test_x, data_parameters[DataParameter.OUTPUT] : M_test_y })

record_prefix = dict()
record_prefix["forms"] = NUM_COMPLEX_FORMS
record_prefix["depth"] = MAX_FORM_DEPTH
record_prefix["model"] = MODEL_NAME
record_prefix["arch"] = ARCH_TYPE
record_prefix["grammar"] = GRAMMAR_TYPE
record_prefix["l1"] = L1_C

for i in range(RERUNS):
    D_train.shuffle()

    modell1 = None
    other_evaluations = []
    if ARCH_TYPE == ArchitectureGrammar.LAYER:
        # FIXME Broken for now
        #modell1 = LinearGrammarRegression("arch_layer_gramression", [D_train, D_dev], F_0.get_size(), F_0.get_size(), R, GRAMMAR_T, bias=True, arch=ARCH_TYPE, arch_depth=MAX_FORM_DEPTH, arch_width=1, \
        #                                  reset_opt=RESET_OPTIMIZER, extend_interval=EXTEND_INTERVAL, arch_grammar_type=GRAMMAR_TYPE)
        raise ValueError
    elif ARCH_TYPE == ArchitectureGrammar.TREE:
        modell1 = LinearGrammarRegression("arch_tree_gramression", [D_train, D_dev], F_0.get_size(), F_0.get_size(), R, GRAMMAR_T, bias=True, arch=ARCH_TYPE, arch_depth=MAX_FORM_DEPTH, arch_width=1, \
                                          reset_opt=RESET_OPTIMIZER, extend_interval=EXTEND_INTERVAL, arch_grammar_type=GRAMMAR_TYPE)
        other_evaluations = [ModelStatistic("nzf", D_dev, data_parameters, lambda m : len(m.get_nonzero_tree_features(D_train[data_parameters[DataParameter.INPUT]].get_feature_set())[0]))]
    elif ARCH_TYPE == ArchitectureGrammar.NONE:
        modell1 = LinearRegression("linear_regression", F_0.get_size(), bias=True)
        other_evaluations = [ModelStatistic("nzf", D_dev, data_parameters, lambda m : 0.0)]

    record_prefix["run"] = i
    logger = Logger()
    final_logger = Logger()
    logger.set_record_prefix(record_prefix)
    logger.set_file_path(OUTPUT_RESULTS_PATH + "_" + str(i))
    final_logger.set_record_prefix(record_prefix)
    final_logger.set_file_path(FINAL_OUTPUT_RESULTS_PATH + "_" + str(i))

    loss_criterion = modell1.get_loss_criterion()

    if GPU:
        modell1 = modell1.cuda()
        loss_criterion = loss_criterion.cuda()

    dev_loss = Loss("dev-loss", D_dev, data_parameters, loss_criterion)
    trainer = Trainer(data_parameters, loss_criterion, logger, \
        dev_loss, other_evaluations=other_evaluations)
    modell1, best_place_holder, best_iteration = trainer.train(modell1, D_train, TRAINING_ITERATIONS, \
        batch_size=BATCH_SIZE, optimizer_type=OptimizerType.ADAGRAD_MUNG, lr=LEARNING_RATE, \
        grad_clip=5.0, log_interval=LOG_INTERVAL, best_part_fn=lambda m : torch.ones(1), l1_C=L1_C)

    logger.dump(file_path=OUTPUT_RESULTS_PATH + "_" + str(i), record_prefix=record_prefix)

    train_loss = Loss("train-loss", D_train, data_parameters, loss_criterion)
    test_loss = Loss("test-loss", D_test, data_parameters, loss_criterion)
    final_evals = [train_loss, dev_loss, test_loss]
    final_evals.append(other_evaluations[0])

    results = Evaluation.run_all(final_evals, modell1)
    final_logger.log(results)
    final_logger.dump(file_path=FINAL_OUTPUT_RESULTS_PATH + "_" + str(i), record_prefix=record_prefix)

    print "Finished training run " + str(i)
    if ARCH_TYPE != ArchitectureGrammar.NONE:
        print "True model"
        w_true = model_true.get_weights()
        for i in range(F_relevant.get_size()):
            print str(F_relevant.get_feature_token(i)) + "\t" + str(w_true[i])
        print "\n"

        print "Trained model"
        w_model = modell1.get_weights()
        F_train = D_train[data_parameters[DataParameter.INPUT]].get_feature_set()
        print "Bias: " + str(modell1.get_bias())
        if ARCH_TYPE == ArchitectureGrammar.LAYER:
            for i in range(F_train.get_size()):
                print str(F_train.get_feature_token(i)) + "\t" + str(w_model[i])
            print "\n"
        else:
            nz_feats, nz_weights = modell1.get_nonzero_tree_features(F_train)
            for i in range(len(nz_feats)):
                f = nz_feats[i]
                print f.get_str(weighted=False) + "\t" + str(nz_weights[i])

