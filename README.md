Grupo 5

Nome: Ana Julia Soares de Oliveira
Nome: Arthur Pires Antunes
Nome: Diego Rian da Silva
Nome: Rafael Roberto de Jesus Ribeiro


# Projeto: Comparação de Algoritmos Apriori e FP-Growth

Este projeto tem como objetivo realizar a mineração de regras de associação e análise exploratória de dados, comparando o desempenho e a execução dos algoritmos **Apriori** e **FP-Growth** aplicados a conjuntos de dados transacionais.

---

## 📂 Estrutura do Repositório

O projeto está organizado da seguinte forma:

* `algoritmos/`: Contém os módulos específicos e scripts de execução dos algoritmos de mineração (`apriori.py` e `fp_growth.py`).
* `dados/`: Contém o conjunto de dados utilizado (`Base_Analise.csv`).
* `resultados/`: Guarda as tabelas processadas, relatórios de resumo (`eda_resumo.txt`) e os gráficos gerados durante a Análise Exploratória (EDA), como distribuição de países, compras por mês e tamanho das cestas.
* `app.py`: Script principal para execução do pipeline da aplicação.
* `eda_baseline.py`: Script responsável pela Análise Exploratória de Dados inicial e pré-processamento.

---

# Tecnologias utilizadas

Python
Pandas
Streamlit
Mlxtend
Matplotlib
Seaborn

##  Instruções do Dataset

1. **Origem e Formato:** O conjunto de dados principal encontra-se em `dados/Base_Analise.csv`.
2. **Pré-processamento e EDA:** O script `eda_baseline.py` processa este ficheiro base para gerar insights visuais e estatísticos guardados automaticamente na pasta `resultados/`.

---

# Como executar o projeto

Instale o Python em uma versão compatível com as bibliotecas utilizadas no projeto.

1. Clona este repositório:
   ```bash
   git clone [https://github.com/teu-utilizador/Projeto_Apriori_FPGrowth.git](https://github.com/teu-utilizador/Projeto_Apriori_FPGrowth.git)
   cd Projeto_Apriori_FPGrowth

2. Instale as dependências

pip install pandas streamlit mlxtend matplotlib seaborn

3. Execute a aplicação
Na pasta principal do projeto, execute:
streamlit run app.py

A aplicação será aberta no navegador, utilizando o endereço local informado pelo Streamlit no terminal.