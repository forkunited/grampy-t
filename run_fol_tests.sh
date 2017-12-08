#!/bin/bash

TEST_DIR=src/test/py/gram/fol
PWD=`pwd` 
export PYTHONPATH=$PYTHONPATH:${PWD}/src/main/py/
export PROVER9=<PATH TO PROVER BINARY>

python ${TEST_DIR}/test_fol_rep.py
#python ${TEST_DIR}/test_fol_data.py
#python ${TEST_DIR}/test_fol_rules.py
#python ${TEST_DIR}/test_fol_model.py
#python ${TEST_DIR}/test_fol_arch.py
