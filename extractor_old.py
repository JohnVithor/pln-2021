# extractor.py

import sys
import re

regex = re.compile('\(([^)^(]*)\)')

if len(sys.argv) != 2:
    print("Informe apenas o nome do arquivo alvo!")
    sys.exit()

with open(sys.argv[1], 'r') as file:
    text = file.read()
    result = regex.findall(text)

with open(sys.argv[1]+'.pairs', 'w') as file:
    for s in result:
        file.write(s+'\n')