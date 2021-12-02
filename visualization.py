import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

unigram = pd.read_csv('resultados_unigram_val.csv', index_col=0)
unigram = unigram.div(unigram.sum(axis=1), axis=0)
unigram.fillna(0, inplace=True)
fig, ax = plt.subplots(figsize=(12,10))
sns.heatmap(unigram, ax=ax)
plt.tight_layout()
plt.savefig("resultados_unigram_val.png")

unigram = pd.read_csv('resultados_unigram_test.csv', index_col=0)
unigram = unigram.div(unigram.sum(axis=1), axis=0)
unigram.fillna(0, inplace=True)
fig, ax = plt.subplots(figsize=(12,10))
sns.heatmap(unigram, ax=ax)
plt.tight_layout()
plt.savefig("resultados_unigram_test.png")

bigram = pd.read_csv('resultados_bigram_val.csv', index_col=0)
bigram = bigram.div(bigram.sum(axis=1), axis=0)
bigram.fillna(0, inplace=True)
fig, ax = plt.subplots(figsize=(12,10))
sns.heatmap(bigram, ax=ax)
plt.tight_layout()
plt.savefig("resultados_bigram_val.png")

unigram = pd.read_csv('resultados_bigram_test.csv', index_col=0)
unigram = unigram.div(unigram.sum(axis=1), axis=0)
unigram.fillna(0, inplace=True)
fig, ax = plt.subplots(figsize=(12,10))
sns.heatmap(unigram, ax=ax)
plt.tight_layout()
plt.savefig("resultados_bigram_test.png")