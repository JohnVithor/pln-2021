# pln-2021 - Unidade 2 HMM

Como usar:

Para aplicar o algoritmo desenvolvido em um arquivo de teste:
python3 hmm.py caminho_para_dados_treino caminho_para_dados_teste_ou_validacao "caractere_separador_do_par_word_tag"

Para automaticamente computar os resultados para validação e teste foi criado um script em .sh: run.sh
Que executa tanto o hmm.py nos devidos arquivos e renomeia os resultados corretamente

As seguintes dependencias foram usadas:

metrics.py: numpy
hmm.py: re numpy pandas metrics numba

Detalhe: a library numba foi utilizada para otimizar o código, sem a mesma o tempo de execução fica muito longo.
