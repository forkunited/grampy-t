import unittest
import numpy as np
import nltk
import torch

import mung.fol.data as data
import mung.feature as feat
import mung.rule as rule
from mung.torch_ext.eval import Loss 
from mung.torch_ext.learn import OptimizerType, Trainer
from mung.util.log import Logger
from mung.fol.rule_form_indicator import BinaryRule
from mung.fol.rule_form_indicator import UnaryRule
from mung.fol.feature_top import FeatureTopType
from mung.fol.feature_form_indicator import FeatureFormIndicatorType
from gram.model.linear import LinearRegression, LogisticRegression, DataParameter

TRAINING_ITERATIONS = 750 #1500
DATA_SIZE = 3000
LOG_INTERVAL = 250
BATCH_SIZE = 100

# FIXME Add back if using gpus
#if gpu:
#torch.cuda.manual_seed(seed)

torch.manual_seed(1)
np.random.seed(1)

class ModelType:
    LINEAR_REGRESSION = 0
    LOGISTIC_REGRESSION = 1

class TestFOLModel(unittest.TestCase):

    def _test_model(self, model_type, lr=1.0, l1_C=0.1):
        print "Starting test..."
        properties_n = 5
        domain = ["0", "1", "2"]

        data_parameters = DataParameter("input", "output")

        properties = []
        F_full = feat.FeatureSet()
        F_relevant = feat.FeatureSet()
        F_0 = feat.FeatureSet()
        w = []
        R = rule.RuleSet()

         
        feature = FeatureTopType("T")
        #F_full.add_feature_type(feature)
        F_relevant.add_feature_type(feature)
        #F_0.add_feature_type(feature)
        w.append(-1.0)

        for i in range(properties_n):
            prop = "P" + str(i)
            properties.append(prop)

            form = data.OpenFormula(domain, prop + "(x)", ["x"])
            feature = FeatureFormIndicatorType(prop, form)
            F_full.add_feature_type(feature)            
            
            token_relevant = feature.get_token(0)
            form_relevant = data.OpenFormula(domain, prop + "(x)", ["x"], init_g=token_relevant.get_closed_form().get_g())
            feature_relevant = FeatureFormIndicatorType(prop, form_relevant)
            F_relevant.add_feature_type(feature_relevant)

            if i == 0:
                F_0.add_feature_type(feature)
            else:
                def p_fn(cf, i=i):
                    ofs = []
                    for var in cf.get_g():
                        ofs.append(data.OpenFormula(domain, "P" + str(i) + "(" + var + ")", [var], init_g=nltk.Assignment(domain,[(var, cf.get_g()[var])])))
                    return ofs
                cf = data.ClosedFormula("P" + str(i-1) + "(x)", nltk.Assignment(domain, []))
                rP = UnaryRule("P" + str(i-1) + "P" + str(i), cf, p_fn)
                R.add_unary_rule(rP)
            w.append(.1*2.0**i)

        w_params = torch.FloatTensor(w)
        model_true = None
        modell1 = None
        if model_type == ModelType.LINEAR_REGRESSION:
            model_true = LinearRegression("linear_true", w_params.size(0), init_params=w_params)
            modell1 = LinearRegression("linear_regression", F_full.get_size(), bias=True)
        else:
            model_true = LogisticRegression("logistic_true", w_params.size(0), init_params=w_params)
            modell1 = LogisticRegression("logistic_regression", F_full.get_size(), bias=True)

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

        M_train_x = feat.DataFeatureMatrix(D_unfeat_train, F_full)
        M_train_y = feat.DataFeatureMatrix(D_unfeat_train, F_label)
        D_train = feat.MultiviewDataSet(data=D_unfeat_train, dfmats={ data_parameters[DataParameter.INPUT] : M_train_x, data_parameters[DataParameter.OUTPUT] : M_train_y })

        M_dev_x = feat.DataFeatureMatrix(D_unfeat_dev, F_full)
        M_dev_y = feat.DataFeatureMatrix(D_unfeat_dev, F_label)
        D_dev = feat.MultiviewDataSet(data=D_unfeat_dev, dfmats={ data_parameters[DataParameter.INPUT] : M_dev_x, data_parameters[DataParameter.OUTPUT] : M_dev_y })
       
        logger = Logger()
        loss_criterion = modell1.get_loss_criterion()
        dev_loss = Loss("loss", D_dev, data_parameters, loss_criterion)
        trainer = Trainer(data_parameters, loss_criterion, logger, \
            dev_loss, other_evaluations=[])
        modell1, best_meaning, best_iteration = trainer.train(modell1, D_train, TRAINING_ITERATIONS, \
            batch_size=BATCH_SIZE, optimizer_type=OptimizerType.ADAGRAD_MUNG, lr=lr, \
            grad_clip=5.0, log_interval=LOG_INTERVAL, best_part_fn=None, l1_C=l1_C)


        # FIXME
        #modell1 = model_linear.PredictionModel.make(model_type)
        #l1_hist = modell1.train_l1(D, F_full, iterations=140001, C=16.0, eta_0=eta_0, alpha=0.8, evaluation_fn=eval_dev)

        # FIXME
        #modell1_g = model_linear.PredictionModel.make(model_type)
        #l1_g_hist = modell1_g.train_l1_g(D, F_0, R, t=0.04, iterations=140001, C=8.0, eta_0=eta_0, alpha=0.8, evaluation_fn=eval_dev)

        print "True model"
        print model_true.get_weights()

        print "l1 model"
        print "Bias: " + str(modell1.get_bias())
        print modell1.get_weights()

        # FIXME
        #print "l1-g model"
        #print str(modell1_g)

        #print "l1 histories"
        #self._print_table(l1_hist, "losses")
        #self._print_table(l1_hist, "l1s")
        #self._print_table(l1_hist, "nzs")
        #self._print_table(l1_hist, "vals")

        # print "l1-g histories"
        # self._print_table(l1_g_hist, "losses")
        # self._print_table(l1_g_hist, "l1s")
        # self._print_table(l1_g_hist, "nzs")
        # self._print_table(l1_g_hist, "vals")


    def test_linear(self):
        print "Linear regression..."
        l1_C = 0.2
        lr = 1.0
        self._test_model(ModelType.LINEAR_REGRESSION, lr=lr, l1_C=l1_C)

    def test_log_linear(self):
        print "Logistic regression..."
        l1_C = 0.2 #0.001
        lr = 1.0
        self._test_model(ModelType.LOGISTIC_REGRESSION, lr=lr, l1_C=l1_C)

if __name__ == '__main__':
    unittest.main()

