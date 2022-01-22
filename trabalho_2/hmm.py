import sys
import numpy as np
import time
import numba
import re
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
import metrics
import pandas as pd

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

if len(sys.argv) != 4:
    print("Informe o caminho para o arquivo de treino, seguido pelo caminho para o arquivo de teste e por fim a string utilizada para separar os pares tag e token")
    sys.exit()

sep = sys.argv[3]
train_pairs = []
train_vocab = set()
train_tags = set()

sentenceStartMarker = '__SS__'

primeira_tag = []
primeira = True
with open(sys.argv[1], 'r') as file:
    for line in file:
        token, tag = sentenceStartMarker, sentenceStartMarker
        train_pairs.append((token, tag))
        primeira = True
        for pair in re.split(' ', line):
            if len(pair) > 2:
                token, tag = pair.split(sep)
                tag = tag.strip()
                if primeira:
                    primeira = False
                    primeira_tag.append(tag)
                train_pairs.append((token, tag))
                train_vocab.add(token)
                train_tags.add(tag)

train_words_sequence = [pair[0] for pair in train_pairs]
train_tags_sequence = [pair[1] for pair in train_pairs]
train_tags = sorted(list(train_tags))
train_vocab = sorted(list(train_vocab))

train_tags_dict = numba.typed.Dict()
train_vocab_dict = numba.typed.Dict()
primeira_tag_dict = {}

for i, tag in enumerate(train_tags):
    train_tags_dict[tag] = i
    primeira_tag_dict[tag] = 0

for i, w in enumerate(train_vocab):
    train_vocab_dict[w] = i

for tag in primeira_tag:
    primeira_tag_dict[tag] += 1

for tag in primeira_tag_dict:
    primeira_tag_dict[tag] /= len(primeira_tag)

primeira_tag_prob = np.zeros(len(train_tags))
for i in range(len(train_tags)):
    primeira_tag_prob[i] = primeira_tag_dict[train_tags[i]]

test_pairs = []
test_vocab = set()
test_tags = set()
with open(sys.argv[2], 'r') as file:
    for line in file:
        token, tag = sentenceStartMarker, sentenceStartMarker
        test_pairs.append((token, tag))
        for pair in re.split(' ', line):
            if len(pair) > 2:
                token, tag = pair.split(sep)
                tag = tag.strip()
                test_pairs.append((token, tag))
                test_vocab.add(token)
                test_tags.add(tag)

test_words_sequence = [pair[0] for pair in test_pairs]
test_tags_sequence = [pair[1] for pair in test_pairs]
test_tags = sorted(list(test_tags))
test_vocab = sorted(list(test_vocab))

# Compute Emission Probability
@numba.jit(nopython=True, nogil=True)
def get_emission_matrix(train_tags_sequence, train_words_sequence, states, train_vocab):
    emission_matrix = np.zeros((len(states), len(train_vocab)))
    states_count = np.zeros(len(states))

    for i in numba.prange(len(train_tags_sequence)):
        if train_tags_sequence[i] != sentenceStartMarker:
            emission_matrix[states[train_tags_sequence[i]], train_vocab[train_words_sequence[i]]] += 1
            states_count[states[train_tags_sequence[i]]] += 1
    
    for i in numba.prange(len(states)):
        emission_matrix[i, :] /= states_count[i]
    return emission_matrix

a = numba.typed.Dict()
b = numba.typed.Dict()

for i, tag in enumerate(list(set(train_tags_sequence[:2]))):
    a[tag]=i

for i, w in enumerate(list(set(train_words_sequence[:2]))):
    b[w]=i

emission_matrix = get_emission_matrix(train_tags_sequence[:2], train_words_sequence[:2], a, b)

emission_matrix = get_emission_matrix(train_tags_sequence, train_words_sequence, train_tags_dict, train_vocab_dict)

# creating t x t transition matrix of tags, t= no of tags
# Matrix(i, j) represents P(jth tag after the ith tag)
# @numba.jit(nopython=True, nogil=True, parallel=True, fastmath=True)
def create_transition_matrix(train_tags:np.ndarray, train_tags_sequence:np.ndarray):
    tags_matrix = np.zeros((len(train_tags), len(train_tags)), dtype=np.float32)
    for i in numba.prange(len(train_tags)):
        for j in numba.prange(len(train_tags)):
            index = train_tags_sequence == train_tags[i]
            count_t1 = index.sum()
            s_index = np.roll(index, 1)
            s_index[0] = False
            count_t2_t1 = (train_tags_sequence[s_index] == train_tags[j]).sum()
            tags_matrix[i, j] = count_t2_t1/count_t1
    return tags_matrix

transition_matrix = create_transition_matrix(np.array(list(set(train_tags_sequence[:2]))), np.array(train_tags_sequence[:2]))

transition_matrix = create_transition_matrix(np.array(train_tags), np.array(train_tags_sequence))

@numba.jit(nopython=True, nogil=True)
def viterbi_log(transition_matrix, initial_probs, emission_matrix, words_sequence_index):
    number_states = transition_matrix.shape[0]
    number_words = len(words_sequence_index)
    tiny = np.finfo(0.).tiny
    transition_matrix_log = np.log(transition_matrix + tiny)
    initial_probs_log = np.log(initial_probs + tiny)
    emission_matrix_log = np.log(emission_matrix + tiny)

    accumulated_prob_matrix = np.zeros((number_states, number_words))
    backtracking_matrix = np.zeros((number_states, number_words-1)).astype(np.int32)
    accumulated_prob_matrix[:, 0] = initial_probs_log + emission_matrix_log[:, words_sequence_index[0]]

    for n in range(1, number_words):
        for i in range(number_states):
            temp_sum = transition_matrix_log[:, i] + accumulated_prob_matrix[:, n-1]
            if words_sequence_index[n] < 0:
                if ((-words_sequence_index[n])-1 == i):
                    emission = np.log(1+tiny)
                else:
                    emission = np.log(tiny)
            else:
                emission = emission_matrix_log[i, words_sequence_index[n]]
            accumulated_prob_matrix[i, n] = np.max(temp_sum) + emission
            backtracking_matrix[i, n-1] = np.argmax(temp_sum)

    optimal_tag_sequence = np.zeros(number_words).astype(np.int32)
    optimal_tag_sequence[-1] = np.argmax(accumulated_prob_matrix[:, -1])
    for n in range(number_words-2, -1, -1):
        optimal_tag_sequence[n] = backtracking_matrix[int(optimal_tag_sequence[n+1]), n]
    return optimal_tag_sequence#, backtracking_matrix, accumulated_prob_matrix

@numba.jit(nopython=True, nogil=True)
def handle_unknown_token(token):
    if token[-1] == '-':
        token = token[:-1]
    if token.replace('.','',1).isdigit() or token.replace(',','',1).isdigit():
        return 'num'
    if token[-3:] == 'ndo':
        return 'v-ger'
    if token[-3:] == 'ada' or token[-4:] == 'adas' or token[-3:] == 'ado' or token[-4:] == 'ados' or \
       token[-3:] == 'ida' or token[-4:] == 'idas' or token[-3:] == 'ido' or token[-4:] == 'idos' or \
       token[-3:] == 'ída' or token[-4:] == 'ídas' or token[-3:] == 'ído' or token[-4:] == 'ídos':
        return 'v-pcp'
    if token[-2:] == 'ar' or token[-2:] == 'er' or token[-2:] == 'ir' or token[-2:] == 'or' or \
       token[-4:] == 'arem' or token[-4:] == 'erem' or token[-4:] == 'irem' or token[-4:] == 'orem':
        return 'v-inf'
    if token[-3:] == 'ava' or token[-4:] == 'imos' or token[-4:] == 'aram' or token[-2:] == 'emos' or \
       token[-4:] == 'avam' or token[-4:] == 'amos' or token[-2:] == 'ou' or token[-4:] == 'ia':
        return 'v-fin'
    if token[-5:] == 'mente':
        return 'adv'
    if token[-4:] == 'ento' or token[-5:] == 'entos' or token[-4:] == 'ável' or token[-5:] == 'áveis' or \
       token[-4:] == 'ante' or token[-5:] == 'antes' or token[-4:] == 'esco' or token[-5:] == 'escos' or \
       token[-4:] == 'ível' or token[-5:] == 'íveis' or token[-3:] == 'ano' or token[-3:] == 'ino' or \
       token[-5:] == 'dente' or token[-4:] == 'ista':
        return 'adj'
    if token[-4:] == 'ismo' or token[-5:] == 'ismos' or token[-5:] == 'idade' or token[-6:] == 'idades':
        return 'n'
    if token[0].isupper():
        return 'prop'
    return 'n'


@numba.jit(nopython=True, nogil=True, parallel=True)
def prepare_viterbi_input(words_sequence, train_vocab_dict, train_tags_dict):
    result = np.zeros(len(words_sequence), dtype=np.int32)
    for i in numba.prange(len(words_sequence)):
        if words_sequence[i] in train_vocab_dict:
            result[i] = train_vocab_dict[words_sequence[i]]
        else:
            result[i] = -(train_tags_dict[handle_unknown_token(words_sequence[i])]+1)
    return result

# @numba.jit(nopython=True, nogil=True)
def process_results(expected_tags, predicted_tags):
    result = {}
    for i in numba.prange(len(expected_tags)):
        if expected_tags[i] in result:
            if 'pred_'+predicted_tags[i] in result[expected_tags[i]]:
                result[expected_tags[i]]['pred_'+predicted_tags[i]] += 1
            else:
                result[expected_tags[i]]['pred_'+predicted_tags[i]] = 1
        else:
            result[expected_tags[i]] = {'pred_'+predicted_tags[i]:1}
    return result

def save_results(result):
    df = pd.DataFrame.from_dict(result, dtype=np.int32)
    for tag in df.columns:
        p_tag = 'pred_'+tag
        if p_tag not in df.index:
            df.loc[p_tag,:] = [np.NaN] * (len(df.columns))
    df.sort_index(axis=0, inplace=True)
    df.sort_index(axis=1, inplace=True)
    df.fillna(0, inplace=True)
    df.to_csv('results_hmm.csv')

    metrics_dict = metrics.extract_metrics_from_confusion_matrix(df.values)
    metrics_df = pd.DataFrame.from_dict(metrics_dict)
    index = list(df.index)
    index.append('total')
    metrics_df.index = index
    metrics_df.to_csv('metrics_hmm.csv')

@numba.jit(nopython=True, nogil=True, parallel=True)
def viterbi_loop(sentences_array, tags_array, train_vocab_dict, train_tags_dict, transition_matrix, primeira_tag_prob, emission_matrix, train_tags, pred_tags_array, expc_tags_array):
    for i in numba.prange(len(sentences_array)):
        words_sequence_index = prepare_viterbi_input(list(sentences_array[i][1:]), train_vocab_dict, train_tags_dict)
        s = viterbi_log(transition_matrix, primeira_tag_prob, emission_matrix, words_sequence_index)
        preds_tags = numba.typed.List()
        for j in numba.prange(len(s)):
            preds_tags.append(train_tags[s[j]])
        pred_tags_array[i] = preds_tags
        expc_tags_array[i] = tags_array[i][1:]
    return pred_tags_array, expc_tags_array

def run_viterbi(word_sequence, tags_sequence, sentenceStartMarker, train_vocab_dict, train_tags_dict, transition_matrix, primeira_tag_prob, emission_matrix, train_tags):
    words_sequence_array = np.array(word_sequence)
    tags_sequence_array = np.array(tags_sequence)
    words_sequence_idx = np.where(words_sequence_array == sentenceStartMarker)[0][1:]

    sentences_lists = numba.typed.List()
    
    for sentence_array in np.split(words_sequence_array, words_sequence_idx):
        sentence = numba.typed.List()
        for token in sentence_array:
            sentence.append(token)
        sentences_lists.append(sentence)

    tags_array =  numba.typed.List()
    
    for tag_array in np.split(tags_sequence_array, words_sequence_idx):
        tags = numba.typed.List()
        for token in tag_array:
            tags.append(token)
        tags_array.append(tags)

    pred_tags_array = tags_array.copy()
    expc_tags_array = tags_array.copy()

    predictions_mat, expectations_mat = viterbi_loop(sentences_lists[:10], tags_array[:10], train_vocab_dict, train_tags_dict, transition_matrix, primeira_tag_prob, emission_matrix, train_tags, pred_tags_array, expc_tags_array)

    predictions_mat, expectations_mat = viterbi_loop(sentences_lists, tags_array, train_vocab_dict, train_tags_dict, transition_matrix, primeira_tag_prob, emission_matrix, train_tags, pred_tags_array, expc_tags_array)

    predictions = []
    for preds in predictions_mat:
        predictions.extend(preds)
    expectations = []
    for preds in expectations_mat:
        expectations.extend(preds)
    pred_expc = pd.DataFrame.from_dict({'predictions': predictions, 'expectations':expectations})
    pred_expc.to_csv('pred_expc_hmm.csv', index=False)
    save_results(process_results(expectations, predictions))


run_viterbi(test_words_sequence, test_tags_sequence, sentenceStartMarker, train_vocab_dict, train_tags_dict, transition_matrix, primeira_tag_prob, emission_matrix, train_tags)
