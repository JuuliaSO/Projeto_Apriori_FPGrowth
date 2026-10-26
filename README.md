# Projeto: Comparação de Algoritmos Apriori e FP-Growth

Este projeto tem como objetivo realizar a mineração de regras de associação e análise exploratória de dados, comparando o desempenho e a execução dos algoritmos **Apriori** e **FP-Growth** aplicados a conjuntos de dados transacionais.

---

## 📂 Estrutura do Repositório

O projeto está organizado da seguinte forma:

* `algoritmos/`: Contém os módulos específicos e scripts de execução dos algoritmos de mineração (`apriori.py` e `fp_growth.py`)[cite: 8].
* `dados/`: Contém o conjunto de dados utilizado (`Base_Analise.csv`)[cite: 8].
* `resultados/`: Guarda as tabelas processadas, relatórios de resumo (`eda_resumo.txt`) e os gráficos gerados durante a Análise Exploratória (EDA), como distribuição de países, compras por mês e tamanho das cestas[cite: 8].
* `app.py`: Script principal para execução do pipeline da aplicação[cite: 8].
* `eda_baseline.py`: Script responsável pela Análise Exploratória de Dados inicial e pré-processamento[cite: 8].

---

## 📊 Instruções do Dataset

1. **Origem e Formato:** O conjunto de dados principal encontra-se em `dados/Base_Analise.csv`[cite: 8].
2. **Pré-processamento e EDA:** O script `eda_baseline.py` processa este ficheiro base para gerar insights visuais e estatísticos guardados automaticamente na pasta `resultados/`[cite: 8].

---

## ⚙️ Pré-requisitos e Instalação

Certifica-te de que tens o **Python** instalado na tua máquina. 

1. Clona este repositório:
   ```bash
   git clone [https://github.com/teu-utilizador/Projeto_Apriori_FPGrowth.git](https://github.com/teu-utilizador/Projeto_Apriori_FPGrowth.git)
   cd Projeto_Apriori_FPGrowth