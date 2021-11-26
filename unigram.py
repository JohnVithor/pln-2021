# unigram.py

import sys
import re
import numpy as np
import pandas as pd
import operator
import json
import metrics

MIN_QTD = 10

token_uni_count = {}

known_tags = set()

token_uni_total = {}
handle_unknown = ""

results_count = {}

def pair_is_not_valid(pair):
    return len(pair) < 3

def preprocess_token(token):
    return token.lower()

def token_is_known(token):
    return token in token_uni_count

def token_is_known_with_tag(token, tag):
    return tag in token_uni_count[token]

def train_handle_token_with_known_tag(token, tag):
    token_uni_count[token][tag] += 1
    token_uni_total[token] += 1

def train_handle_token_with_unknown_tag(token, tag):
    known_tags.add(tag)
    token_uni_count[token][tag] = 1
    token_uni_total[token] += 1

def train_handle_known_token(token, tag):
    if token_is_known_with_tag(token, tag):
        train_handle_token_with_known_tag(token, tag)
    else: 
        train_handle_token_with_unknown_tag(token, tag)

def train_handle_unknown_token(token, tag):
    token_uni_count[token] = {}
    token_uni_count[token][tag] = 1
    token_uni_total[token] = 1
    known_tags.add(tag)

def compute_unknow_handle():
    tags = {}
    for token in token_uni_total:
        if token_uni_total[token] < MIN_QTD:
            for tag in token_uni_count[token]:
                if tag in tags:
                    tags[tag] += token_uni_count[token][tag]
                else:
                    tags[tag] = token_uni_count[token][tag]
            
    return max(tags.items(), key=operator.itemgetter(1))[0]

def predict_tag_for_token(token):
    return max(token_uni_count[token].items(), key=operator.itemgetter(1))[0]

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

def test_handle_known_token(token, tag):
    p_tag = predict_tag_for_token(token)
    handle_predicted_tag(p_tag, tag)
    return p_tag

def token_is_CD_tag(token, tag):
    return re.match("\d*\.?\d+", token)

def test_handle_unknown_token(token, tag, default):
    if token_is_CD_tag(token, tag):
        handle_predicted_tag("CD", tag)
        return "CD"
    else:
        handle_predicted_tag(default, tag)
        return default

def main():
    if len(sys.argv) != 3:
        print("Informe apenas o nome do arquivo do corpus e o arquivo alvo do tagging")
        sys.exit()

    with open(sys.argv[1], 'r') as file:
        train_lines = file.readlines()

    with open(sys.argv[2], 'r') as file:
        test_lines = file.readlines()

    for line in train_lines:
        for pair in re.split('\s', line):
            if pair_is_not_valid(pair):
                continue
            token, tag = pair.split('_')
            token = preprocess_token(token)
            if token_is_known(token):
                train_handle_known_token(token, tag)
            else:
                train_handle_unknown_token(token, tag)

    handle_unknown = compute_unknow_handle()

    predicted = []
    expected = []

    for line in test_lines:
        for pair in re.split('\s', line):
            if pair_is_not_valid(pair):
                continue
            token, tag = pair.split('_')
            token = preprocess_token(token)
            if token_is_known(token):
                p_tag = test_handle_known_token(token, tag)
            else:
                p_tag = test_handle_unknown_token(token, tag, handle_unknown)
            predicted.append(p_tag)
            expected.append(tag)

    pred_expc = pd.DataFrame.from_dict({'predicted': predicted, 'expected':expected})
    pred_expc.to_csv('pred_expc_unigram.csv', index=False)

    df = pd.DataFrame.from_dict(results_count, dtype=np.int32)
    for tag in df.columns:
        p_tag = 'pred_'+tag
        if p_tag not in df.index:
            df.loc[p_tag,:] = [np.NaN] * (len(df.columns))
    df.sort_index(axis=0, inplace=True)
    df.sort_index(axis=1, inplace=True)
    df.fillna(0, inplace=True)
    df.to_csv('resultados_unigram.csv')
    
    with open("metrics_unigram.json", 'w') as f:
        metrics_dict = metrics.extract_metrics_from_confusion_matrix(df.values)
        json.dump(metrics_dict, f,indent=4, sort_keys=True)
    metrics_df = pd.DataFrame.from_dict(metrics_dict)
    metrics_df.index = df.index
    metrics_df.to_csv('metrics_unigram.csv')

if __name__ == "__main__":
    main()