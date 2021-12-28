import sys
import numpy as np
import pandas as pd

if len(sys.argv) != 2:
    print("Informe apenas o nome do arquivo alvo!")
    sys.exit()

pairs = []
tags = set()
with open(sys.argv[1], 'r') as file:
    for line in file:
        pair = tuple(line.strip().split(' '))
        pairs.append(pair)
        tags.add(pair[0])

tags = sorted(list(tags))
tags_sequence = [pair[0] for pair in pairs]

# Compute Emission Probability
def word_given_tag(word, tag, train_bag):
    tag_list = [pair for pair in train_bag if pair[0]==tag]
    #total number of times the passed tag occurred in train_bag
    count_tag = len(tag_list)
    w_given_tag_list = [pair[1] for pair in tag_list if pair[1]==word]
    # Now calculate the total number of times the passed word occurred as the passed tag.
    count_w_given_tag = len(w_given_tag_list)
    return (count_w_given_tag, count_tag)

# Compute  Transition Probability
def t2_given_t1(t2, t1, tags):
    count_t1 = len([t for t in tags if t==t1])
    count_t2_t1 = 0
    for index in range(len(tags)-1):
        if tags[index]==t1 and tags[index+1] == t2:
            count_t2_t1 += 1
    return (count_t2_t1, count_t1)


# creating t x t transition matrix of tags, t= no of tags
# Matrix(i, j) represents P(jth tag after the ith tag)
 
tags_matrix = np.zeros((len(tags), len(tags)), dtype='float32')
for i, t1 in enumerate(list(tags)):
    for j, t2 in enumerate(list(tags)): 
        r = t2_given_t1(t2, t1, tags_sequence)
        tags_matrix[i, j] = r[0]/r[1]
 
tags_df = pd.DataFrame(tags_matrix, columns = list(tags), index=list(tags))
print(tags_df.head())