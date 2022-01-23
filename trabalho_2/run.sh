#!/bin/bash
python3 hmm.py pt/Floresta_7.4.PennTreebank.ptb_train.pairs pt/Floresta_7.4.PennTreebank.ptb_val.pairs "@"
mv metrics_hmm.csv metrics_hmm_val.csv
mv pred_expc_hmm.csv pred_expc_hmm_val.csv
mv results_hmm.csv results_hmm_val.csv
mv sentences_precision_hmm.csv sentences_precision_hmm_val.csv

python3 hmm.py pt/Floresta_7.4.PennTreebank.ptb_train.pairs pt/Floresta_7.4.PennTreebank.ptb_test.pairs "@"
mv metrics_hmm.csv metrics_hmm_test.csv
mv pred_expc_hmm.csv pred_expc_hmm_test.csv
mv results_hmm.csv results_hmm_test.csv
mv sentences_precision_hmm.csv sentences_precision_hmm_test.csv