# extractor.py

import sys
import re
import numpy as np
import random

if len(sys.argv) != 3:
    print("Informe o nome do arquivo alvo seguido da seed para criar os splits train, validation e test")
    sys.exit()

frases = {}
text = {}

sep='@'
frase_id = ''
new_frase = False
regex_frase_id = re.compile('(#\d+\s(\w+)-(\d+|\d+\w+)\s+)')
regex_tel_number = re.compile('(num_\d+-\d+)\'(\s\<\w+\>\s|\s)\w\s\w\}')

regex = re.compile('\(([^)^(]*)\)')

with open(sys.argv[1], 'r') as file:
    for line in file:
        if line.startswith('#'):
            new_frase = True
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
                
                line = regex.findall(line)
                if len(line) > 0:
                    line = line[0]
                else:
                    continue
                if frase_id in text:
                    if new_frase:
                        text[frase_id].append([line])
                    else:
                        text[frase_id][-1].append(line)   
                else:
                    text[frase_id] = [[line]]
            new_frase = False

with open(sys.argv[1]+'.text', 'w') as file:
    for id in frases:
        file.write('#'+id+'\n')
        for f in frases[id]:
            file.write(f)

keys = list(text.keys())
random.seed(int(sys.argv[2]))
random.shuffle(keys)

train_test_divider = np.ceil(len(keys)*0.8).astype(int)
train_keys = keys[:train_test_divider]
test_keys = keys[train_test_divider:]

train_val_divider = np.ceil(len(train_keys)*0.8).astype(int)
val_keys = train_keys[train_val_divider:]
train_keys = train_keys[:train_val_divider]

print(len(train_keys))
print(len(val_keys))
print(len(test_keys))

with open(sys.argv[1]+'_train.pairs', 'w') as file:
    for key in train_keys:
        for frase in text[key]:
            for s in frase:
                s = str(s)
                s1 = s.split(' ')
                if ('+' in s1[0]):
                    s1[0] = s1[0][s1[0].rfind('+')+1:]
                file.write(s1[-1])
                file.write(sep)
                file.write(s1[0])
                file.write(" ")
            file.write("\n")
        
with open(sys.argv[1]+'_val.pairs', 'w') as file:
    for key in val_keys:
        for frase in text[key]:
            for s in frase:
                s = str(s)
                s1 = s.split(' ')
                if ('+' in s1[0]):
                    s1[0] = s1[0][s1[0].rfind('+')+1:]
                file.write(s1[-1])
                file.write(sep)
                file.write(s1[0])
                file.write(" ")
            file.write("\n")

with open(sys.argv[1]+'_test.pairs', 'w') as file:
    for key in test_keys:
        for frase in text[key]:
            for s in frase:
                s = str(s)
                s1 = s.split(' ')
                if ('+' in s1[0]):
                    s1[0] = s1[0][s1[0].rfind('+')+1:]
                file.write(s1[-1])
                file.write(sep)
                file.write(s1[0])
                file.write(" ")
            file.write("\n")
