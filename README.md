Grupo 5

Nome: Ana Julia Soares de Oliveira
Nome: Arthur Pires Antunes
Nome: Diego Rian da Silva
Nome: Rafael Roberto de Jesus Ribeiro


## Identificação do problema

### O problema
Lojas de varejo online registram milhares de compras, mas não sabem **quais produtos costumam ser comprados juntos**. Sem essa informação, não dá para sugerir produtos associados, montar combos ou organizar promoções com base em dados.

### Pergunta que o projeto responde
> **Quais itens aparecem juntos nas compras, e com que força essa associação acontece?**

### Por que não é trivial
- **Muitos produtos e poucas combinações repetidas:** na base usada, são 4.002 produtos em 19.773 compras, e a maioria dos produtos é rara (1.148 aparecem em menos de 20 compras). Por isso, a matriz de compras é muito esparsa.
- **Número de combinações possível enorme:** testar todas as combinações de produtos é inviável, e é por isso que usamos algoritmos de mineração de padrões frequentes.
- **Regras enganosas:** um suporte mínimo muito baixo gera milhares de regras, e parte delas se apoia em poucas compras, é óbvia (variantes do mesmo produto) ou só reflete a popularidade de um item. Distinguir regras úteis de ruído faz parte do problema.
- **Dados reais com ruído:** a base tem cancelamentos, taxas, fretes, preços zerados e quantidades negativas, que precisam ser tratados para não gerar associações falsas.


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
