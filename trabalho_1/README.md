# pln-2021

Como usar:

Para utilizar individualmente a versão com unigramas:
python3 unigram.py caminho_para_dados_treino caminho_para_dados_teste_ou_validacao "caractere_separador_do_par_word_tag"

Para utilizar individualmente a versão com bigramas:
python3 bigram.py caminho_para_dados_treino caminho_para_dados_teste_ou_validacao "caractere_separador_do_par_word_tag"

Para automaticamente computar os resultados para validação e teste do corpus em ingles foi criado um script em .sh: run.sh
Que executa tanto o unigram.py como o bigram.py nos devidos arquivos, renomeia os resultados corretamente e já cria os gráficos usados no relatorio.

As seguintes dependencias foram usadas:

metrics.py: numpy
unigram.py: re numpy pandas metrics
bigram.py: re numpy pandas metrics
visualization.py: pandas seaborn matplotlib.pyplot
