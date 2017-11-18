import unittest
import numpy as np
import mung.fol.data as data

class TestFOLRep(unittest.TestCase):

    def test_ops(self):
        np.random.seed(2)
        
        domain = ["0", "1", "2", "3"]
        properties = ["P0", "P1"]
        binary_rels = []        
      
        form1 = data.OpenFormula(domain, "P0(x)", ["x"])
        form2 = data.OpenFormula(domain, "P1(y)", ["y"])
        c12 = form1.conjoin(form2)


        print form1.orthogonize(form2).get_exp()
        print form1.conjoin(form2, orthogonize=True).get_exp()
        print form1.conjoin(form2, orthogonize=False).get_exp()

        model = data.RelationalModel.make_random(domain, properties, binary_rels)
        s1 = form1.satisfiers(model)
        s2 = form2.satisfiers(model)
        s12 = c12.satisfiers(model)
       
        print "Model:"
        print model._model
        print "P0(x)"
        print s1
        print "P1(y)"
        print s2
        print "P0(x) & P1(y)"
        print s12 


if __name__ == '__main__':
    unittest.main()
