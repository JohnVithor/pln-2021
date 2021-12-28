# extractor.py

import sys
import re
import numpy as np
import random

if len(sys.argv) != 2:
    print("Informe apenas o nome do arquivo alvo!")
    sys.exit()

frases = {}
text = {}

frase_id = ''
regex_frase_id = re.compile('(#\d+\s(\w+)-(\d+|\d+\w+)\s+)')
regex_tel_number = re.compile('(num_\d+-\d+)\'(\s\<\w+\>\s|\s)\w\s\w\}')

with open(sys.argv[1], 'r') as file:
    for line in file:
        if line.startswith('#'):
            code = regex_frase_id.findall(line)
            if len(code) > 0:
                frase_id = code[0][1]
                if frase_id in frases:
                    frases[frase_id].append(regex_frase_id.sub("", line))
                else:
                    frases[frase_id] = [regex_frase_id.sub("", line)]
        else:
            if frase_id != '':
                if "N<{'185/60_R_14'}" in line:
                    continue
                if "num_" in line:
                    line = regex_tel_number.sub("num", line)
                if frase_id in text:
                    text[frase_id].append(line)
                else:
                    text[frase_id] = [line]

with open(sys.argv[1]+'.text', 'w') as file:
    file.write('\n'.join(frases))

keys = list(text.keys())
random.seed(0)
random.shuffle(keys)

divider = np.ceil(len(keys)*0.8).astype(int)

train_keys = keys[:divider]
test_keys = keys[divider:]

regex = re.compile('\(([^)^(]*)\)')

train_text = []
test_text = []

for key in train_keys:
    for line in text[key]:
        train_text.append(line)

train_text = '\n'.join(train_text)
train_result = regex.findall(train_text)

for key in test_keys:
    for line in text[key]:
        test_text.append(line)

test_text = '\n'.join(test_text)
test_result = regex.findall(test_text)

with open(sys.argv[1]+'_train.pairs', 'w') as file:
    for s in train_result:
        s = str(s)
        s1 = s.split(' ')
        if ('+' in s1[0]):
            s = s[s1[0].rfind('+')+1:]
        file.write(s)
        if len(s) < 4:
            file.write(" ")
            file.write(s)
        file.write("\n")

with open(sys.argv[1]+'_test.pairs', 'w') as file:
    for s in test_result:
        s = str(s)
        s1 = s.split(' ')
        if ('+' in s1[0]):
            s = s[s1[0].rfind('+')+1:]
        file.write(s)
        if len(s) < 4:
            file.write(" ")
            file.write(s)
        file.write("\n")