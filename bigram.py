# bigram.py

import sys
import re
from numpy import NaN
import pandas as pd
import operator

MIN_QTD = 10

token_count = {}
token_uni_count = {}

known_tags = set()

token_total = {}
token_af_token_total = {}
handle_unknown = ""

results_count = {}

def pair_is_valid(pair):
    return len(pair) < 3

def preprocess_token(token):
    return token.lower()

def token_is_known(token):
    return token in token_count

def token_is_known_after_token(token, previous_token):
    return previous_token in token_count[token]

def token_after_token_is_known_with_tag(token, previous_token, tag, previous_tag):
    return tag in token_count[token][previous_token]

def train_handle_token_with_known_tag(token, previous_token, tag, previous_tag):
    token_count[token][previous_token][tag] += 1
    token_af_token_total[token][previous_token] += 1
    token_total[token] += 1

def train_handle_token_with_unknown_tag(token, previous_token, tag, previous_tag):
    known_tags.add(tag)
    token_count[token][previous_token][tag] = 1
    token_af_token_total[token][previous_token] += 1
    token_total[token] += 1

def train_handle_known_token(token, previous_token, tag, previous_tag):
    if token_is_known_after_token(token, previous_token):
        if token_after_token_is_known_with_tag(token, previous_token, tag, previous_tag):
            train_handle_token_with_known_tag(token, previous_token, tag, previous_tag)
        else: 
            train_handle_token_with_unknown_tag(token, previous_token, tag, previous_tag)
    else:
        token_count[token][previous_token] = {}
        token_af_token_total[token][previous_token] = 0
        train_handle_token_with_unknown_tag(token, previous_token, tag, previous_tag)


def train_handle_unknown_token(token, previous_token, tag, previous_tag):
    token_count[token] = {}
    token_count[token][previous_token] = {}
    token_count[token][previous_token][tag] = 1
    token_af_token_total[token] = {}
    token_af_token_total[token][previous_token] = 1
    token_total[token] = 1
    known_tags.add(tag)

def compute_unknow_handle():
    tags = {}
    for token in token_total:
        token_uni_count[token] = {}
        if token_total[token] < MIN_QTD:
            for p_token in token_count[token]:
                for tag in token_count[token][p_token]:
                    if tag in tags:
                        tags[tag] += token_count[token][p_token][tag]
                    else:
                        tags[tag] = token_count[token][p_token][tag]
                    if tag in token_uni_count[token]:
                        token_uni_count[token][tag] += token_count[token][p_token][tag]
                    else:
                        token_uni_count[token][tag] = token_count[token][p_token][tag]
        else:
            for p_token in token_count[token]:
                for tag in token_count[token][p_token]:
                    if tag in token_uni_count[token]:
                        token_uni_count[token][tag] += token_count[token][p_token][tag]
                    else:
                        token_uni_count[token][tag] = token_count[token][p_token][tag]
        token_uni_count[token]['prediction'] = max(token_uni_count[token].items(), key=operator.itemgetter(1))[0]

            
    return max(tags.items(), key=operator.itemgetter(1))[0]

def predict_tag_for_token_with_known_previous_token(token, previous_token, previous_p_tag):
    return max(token_count[token][previous_token].items(), key=operator.itemgetter(1))[0]

def predict_tag_for_token_with_unknown_previous_token(token, previous_p_tag):
    return token_uni_count[token]['prediction']

def tag_in_result(tag):
    return tag in results_count

def p_tag_in_tag_results(p_tag, tag):
    return 'pred_'+p_tag in results_count[tag]

def handle_known_p_tag_as_tag(p_tag, tag):
    results_count[tag]['pred_'+p_tag] += 1

def handle_unknown_p_tag_as_tag(p_tag, tag):
    results_count[tag]['pred_'+p_tag] = 1

def handle_predicted_tag(p_tag, tag):
    if tag_in_result(tag):
        if p_tag_in_tag_results(p_tag, tag):
            handle_known_p_tag_as_tag(p_tag, tag)
        else:
            handle_unknown_p_tag_as_tag(p_tag, tag)
    else:
        known_tags.add(tag)
        results_count[tag] = {}
        handle_unknown_p_tag_as_tag(p_tag, tag)

def test_handle_token_with_known_tag(token, tag):
    pass

def test_handle_token_with_unknown_tag(token, tag):
    pass

def test_handle_known_token(token, previous_token, tag, previous_p_tag):
    if token_is_known_after_token(token, previous_token):
        p_tag = predict_tag_for_token_with_known_previous_token(token, previous_token, previous_p_tag)
    else:
        p_tag = predict_tag_for_token_with_unknown_previous_token(token, previous_p_tag)
    handle_predicted_tag(p_tag, tag)

def token_is_CD_tag(token, tag):
    return re.match("\d*\.?\d+", token)

def handle_CD_token(token, tag):
    p_tag = "CD"
    handle_predicted_tag(p_tag, tag)

def predict_unknow_tag_using_previous_token_p_tag(previous_token, previous_p_tag, handle_unknown):
    #TODO
    return handle_unknown

def test_handle_unknown_token(token, previous_token, tag, previous_p_tag, handle_unknown):
    if token_is_CD_tag(token, tag):
        handle_CD_token(token, tag)
    else:
        p_tag = predict_unknow_tag_using_previous_token_p_tag(previous_token, previous_p_tag, handle_unknown)
        handle_predicted_tag(p_tag, tag)

def main():

    if len(sys.argv) != 3:
        print("Informe apenas o nome do arquivo do corpus e o arquivo alvo do tagging")
        sys.exit()

    with open(sys.argv[1], 'r') as file:
        train = re.split('\s|\n', file.read())

    with open(sys.argv[2], 'r') as file:
        test = re.split('\s|\n', file.read())

    previous_token = '.'
    previous_tag = '.'

    for pair in train:
        if pair_is_valid(pair):
            continue
        token, tag = pair.split('_')
        token = preprocess_token(token)
        if token_is_known(token):
            train_handle_known_token(token, previous_token, tag, previous_tag)
        else:
            train_handle_unknown_token(token, previous_token, tag, previous_tag)
        previous_token = token
        previous_tag = tag

    handle_unknown = compute_unknow_handle()

    previous_token = '.'
    previous_p_tag = '.'

    for pair in test:
        if pair_is_valid(pair):
            continue
        token, tag = pair.split('_')
        token = preprocess_token(token)
        if token_is_known(token):
            p_tag = test_handle_known_token(token, previous_token, tag, previous_p_tag)
        else:
            p_tag = test_handle_unknown_token(token, previous_token, tag, previous_p_tag, handle_unknown)
        previous_token = token
        previous_p_tag = p_tag

    df = pd.DataFrame.from_dict(results_count)
    for tag in df.columns:
        p_tag = 'pred_'+tag
        if p_tag not in df.index:
            df.loc[p_tag,:] = [NaN] * (len(df.columns))
    df.sort_index(axis=0, inplace=True)
    df.sort_index(axis=1, inplace=True)
    df.to_csv('resultados.csv')
    print(df.head())

if __name__ == "__main__":
    main()