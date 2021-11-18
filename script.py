# unigram.py

import sys
import re
import pandas as pd
import operator

token_count = {}
results_count = {}

def pair_is_valid(pair):
    return len(pair) < 3

def token_is_known(token):
    return token in token_count

def token_is_known_with_tag(token, tag):
    return tag in token_count[token]

def train_handle_token_with_known_tag(token, tag):
    token_count[token][tag] += 1

def train_handle_token_with_unknown_tag(token, tag):
    token_count[token][tag] = 1

def train_handle_known_token(token, tag):
    if token_is_known_with_tag(token, tag):
        train_handle_token_with_known_tag(token, tag)
    else: 
        train_handle_token_with_unknown_tag(token, tag)

def train_handle_unknown_token(token, tag):
    token_count[token] = {}
    token_count[token][tag] = 1

def predict_tag_for_token(token):
    return max(token_count[token].items(), key=operator.itemgetter(1))[0]

def tag_in_result(tag):
    return tag in results_count

def p_tag_in_tag_results(p_tag, tag):
    return p_tag in results_count[tag]

def handle_known_p_tag_as_tag(p_tag, tag):
    results_count[tag][p_tag] += 1

def handle_unknown_p_tag_as_tag(p_tag, tag):
    results_count[tag][p_tag] = 1

def handle_predicted_tag(p_tag, tag):
    if tag_in_result(tag):
        if p_tag_in_tag_results(p_tag, tag):
            handle_known_p_tag_as_tag(p_tag, tag)
        else:
            handle_unknown_p_tag_as_tag(p_tag, tag)
    else:
        results_count[tag] = {}
        handle_unknown_p_tag_as_tag(p_tag, tag)

def test_handle_token_with_known_tag(token, tag):
    token_count[token][tag] += 1

def test_handle_token_with_unknown_tag(token, tag):
    token_count[token][tag] = 1

def test_handle_known_token(token, tag):
    p_tag = predict_tag_for_token(token)
    handle_predicted_tag(p_tag, tag)

def token_is_CD_tag(token, tag):
    return re.match("\d*\.?\d+", token)

def handle_CD_token(token, tag):
    p_tag = "CD"
    handle_predicted_tag(p_tag, tag)

def test_handle_unknown_token(token, tag):
    if token_is_CD_tag(token, tag):
        handle_CD_token(token, tag)
    else:
        print(f"I don't know how to classify this token: '{token}' its tag is: '{tag}'")

def main():

    if len(sys.argv) != 3:
        print("Informe apenas o nome do arquivo do corpus e o arquivo alvo do tagging")
        sys.exit()

    with open(sys.argv[1], 'r') as file:
        train = re.split('\s|\n', file.read())

    with open(sys.argv[2], 'r') as file:
        test = re.split('\s|\n', file.read())

    for pair in train:
        if pair_is_valid(pair):
            continue
        token, tag = pair.split('_')
        if token_is_known(token):
            train_handle_known_token(token, tag)
        else:
            train_handle_unknown_token(token, tag)

    for pair in test:
        if pair_is_valid(pair):
            continue
        token, tag = pair.split('_')
        if token_is_known(token):
            test_handle_known_token(token, tag)
        else:
            test_handle_unknown_token(token, tag)

    df = pd.DataFrame.from_dict(results_count)
    df.sort_index(axis=0, inplace=True)
    df.sort_index(axis=1, inplace=True)
    df.to_csv('resultados.csv')
    print(df.head())

if __name__ == "__main__":
    main()