#!/bin/bash
python3 unigram.py data/secs0-18-training data/secs19-21-development
mv metrics_unigram.csv metrics_unigram_val.csv
mv pred_expc_unigram.csv pred_expc_unigram_val.csv
mv resultados_unigram.csv resultados_unigram_val.csv

python3 unigram.py data/secs0-18-training data/secs22-24-testing
mv metrics_unigram.csv metrics_unigram_test.csv
mv pred_expc_unigram.csv pred_expc_unigram_test.csv
mv resultados_unigram.csv resultados_unigram_test.csv

python3 bigram.py data/secs0-18-training data/secs19-21-development
mv metrics_bigram.csv metrics_bigram_val.csv
mv pred_expc_bigram.csv pred_expc_bigram_val.csv
mv resultados_bigram.csv resultados_bigram_val.csv

python3 bigram.py data/secs0-18-training data/secs22-24-testing
mv metrics_bigram.csv metrics_bigram_test.csv
mv pred_expc_bigram.csv pred_expc_bigram_test.csv
mv resultados_bigram.csv resultados_bigram_test.csv