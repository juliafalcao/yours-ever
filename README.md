

*yours ever*\
*virginia woolf*

Projeto de processamento de linguagem natural e aprendizado de representações, utilizando cartas escritas por Virginia Woolf.
Feito em Python (3.6).

#### Bibliotecas utilizadas
* numpy
* pandas
* matplotlib
* beautiful soup 4
* dateutil
* nltk
* gensim
* scipy
* sklearn
* wordcloud

#### Arquivos

* ```scraping.py```: leitura dos arquivos HTML das cartas e scraping para obter o conteúdo e dados como destinatário e data.
* ```analysis.py```: análise de dados preliminar.
* ```preprocessing.py```: remoção e substituição de símbolos e letras em caixa alta, divisão em parágrafos, substituição das datas por somente anos, tokenização, limpeza de stop-words e lematização.
* ```embedding.py```: cálculo das taxas de TF-IDF por parágrafo, criação e treinamento do modelo Word2Vec para geração de embeddings, criação dos embeddings de parágrafos.
* ```clustering.py```: tarefa de clusterização para avaliar os embeddings gerados.
* ```const.py```: reúne variáveis utilizadas nos arquivos acima.

#### Denominações das tabelas de dados
* ```VW```: dataset original de cartas.
* ```VWP```: dataset de cartas divididas em parágrafos.
* ```VWB```: dataset de livros usados para treinar o Word2Vec.
