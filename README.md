# Projeto IA II: Apriori e FP-Growth

**Quais itens aparecem juntos nas compras?** Mineração de regras de associação em compras de varejo online, comparando os algoritmos **Apriori** e **FP-Growth**.

**Grupo 5** · Ciência da Computação · Faculdades Integradas de Bauru (FIB) · Inteligência Artificial II · 2026
Professor: Ronaldo Cesar Dametto

**Integrantes**
- Ana Júlia Soares de Oliveira
- Arthur Pires Antunes
- Diego Rian da Silva
- Rafael Roberto de Jesus Ribeiro

---

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

### Objetivo
Identificar associações entre produtos nas compras e **comparar Apriori e FP-Growth** (resultados e tempo de execução), usando suporte, confiança e lift e filtrando regras espúrias.

### Entrada e saída do sistema
| | |
|---|---|
| **Entrada** | Compras (uma cesta de produtos por fatura), suporte mínimo e confiança mínima definidos pelo usuário |
| **Saída** | Conjuntos de produtos frequentes e regras de associação (se compra A, tende a comprar B), com suporte, confiança e lift |

### Público
Analistas de dados e responsáveis por supermercados ou lojas virtuais.

---

## Dataset

- **Nome:** Online Retail
- **Fonte:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/352/online%2Bretail)
- **Licença:** CC BY 4.0
- **Conteúdo:** compras de uma loja online do Reino Unido entre 01/12/2010 e 09/12/2011, com 8 colunas (InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country).
- **Arquivo do projeto:** `dados/Base_Analise.csv` (separador `;` e decimal `,`), já tratado.

**Tratamento:** a base original tinha 541.909 registros. Foram removidos cancelamentos (fatura iniciada por C), descrições vazias ou inválidas, códigos que não são produto (taxas, fretes, ajustes, descontos), itens com preço zero e quantidades negativas. Restaram **527.761 linhas, 19.773 compras e 4.002 produtos**. Cada compra (`InvoiceNo`) vira uma cesta com os produtos distintos.

---

## Técnicas e métricas

- **Apriori:** gera combinações candidatas e descarta as que não são frequentes.
- **FP-Growth:** evita a geração de candidatos e organiza as compras em uma árvore compacta (FP-Tree).
- **Suporte:** em que fração das compras o conjunto de produtos aparece.
- **Confiança:** entre as compras que têm A, em quantas também tem B.
- **Lift:** quantas vezes B aparece mais quando A está na compra do que no geral.
- **Baseline:** ranking dos produtos que aparecem em mais compras.

---

## Estrutura do repositório

```
.
├── app.py                  # protótipo web (Streamlit)
├── eda_baseline.py         # análise exploratória e baseline
├── experimentos.py         # experimentos E1 a E4
├── requirements.txt        # dependências
├── algoritmos/
│   ├── apriori.py
│   └── fp_growth.py
├── dados/
│   └── Base_Analise.csv
├── resultados/             # tabelas, gráficos e resumos gerados pelos scripts
├── Docs/                   # resumo expandido, diário de IA e checklist
└── Referencias_artigo/     # artigos de referência
```

---

## Tecnologias

Python · pandas · mlxtend · matplotlib · Streamlit

---

## Como executar

Use o Python em uma versão compatível com as bibliotecas (o grupo desenvolveu com Python 3.14). Rode todos os comandos a partir da pasta principal do projeto.

1. Clone o repositório:
   ```bash
   git clone https://github.com/JuuliaSO/Projeto_Apriori_FPGrowth.git
   cd Projeto_Apriori_FPGrowth
   ```

2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

3. Execute o protótipo:
   ```bash
   streamlit run app.py
   ```
   A aplicação abre no navegador, no endereço local informado pelo Streamlit no terminal. Na tela, escolha o algoritmo, o suporte mínimo e a confiança mínima.

### Reprodução da análise e dos experimentos

```bash
python eda_baseline.py              # EDA e baseline
python experimentos.py              # experimentos completos (leva alguns minutos)
python experimentos.py --rapido     # teste curto: 1 repetição e sem o suporte de 0,3%
```

Os resultados são salvos em `resultados/`:

| Arquivo | Conteúdo |
|---|---|
| `eda_resumo.txt` e `eda_*.png` | números e gráficos da análise exploratória |
| `baseline_top_produtos.csv` e `baseline_top10.png` | ranking dos mais vendidos |
| `e1_tempo.csv` e `.png` | E1: tempo de Apriori e FP-Growth por suporte |
| `e2_suporte.csv` e `.png` | E2: conjuntos e regras por suporte |
| `e3_top20_regras_por_lift.csv` | E3: regras de maior lift contra o baseline |
| `e4_regras_frageis.csv` e `e4_regras_mesma_familia.csv` | E4: casos de erro e limitação |
| `experimentos_resumo.txt` | resumo de todos os experimentos |

A contagem de conjuntos e regras é a mesma em qualquer computador. Os **tempos** mudam conforme a máquina e as versões das bibliotecas.

---

## Resultados principais

**Análise exploratória**
- Mediana de 15 itens por compra, e 1.500 compras (7,6%) com um item só, que não geram regras.
- O Reino Unido concentra 90,5% das compras (38 países no total).
- Calda longa: 1.148 dos 4.002 produtos aparecem em menos de 20 compras, e a matriz compra × produto tem só 0,65% das células ocupadas.

**E1: tempo (mesmos dados e parâmetros)**

Apriori e FP-Growth encontram exatamente os mesmos conjuntos e regras. O tempo (em segundos) muda com o suporte:

| Suporte | Conjuntos | Apriori | FP-Growth | Mais rápido |
|---|---|---|---|---|
| 3% | 141 | 0,14 | 0,53 | Apriori |
| 2% | 382 | 0,44 | 0,94 | Apriori |
| 1% | 1.874 | 2,75 | 2,41 | FP-Growth |
| 0,5% | 15.877 | 10,72 | 6,33 | FP-Growth |
| 0,3% | 157.933 | 47,25 | 17,13 | FP-Growth |

**E2: efeito do suporte mínimo (confiança 50%)**

| Suporte | Conjuntos frequentes | Regras |
|---|---|---|
| 5% | 33 | 0 |
| 3% | 141 | 13 |
| 2% | 382 | 60 |
| 1% | 1.874 | 841 |
| 0,5% | 15.877 | 20.725 |
| 0,3% | 157.933 | não gerado (volume muito grande) |

**E3: regras contra o baseline (suporte 1%, confiança 50%)**
- 841 regras, todas com lift maior que 1 (mediana 12,5; máximo 78,8).
- Só 21% têm consequente entre os 10 produtos mais vendidos, e 79% dos produtos das regras estão fora do top 50 do baseline. As regras mostram relações que o ranking não mostra.

**E4: erros e limitações**
- **Suporte muito baixo:** o número de regras explode (de 841 em 1% para 20.725 em 0,5%).
- **Evidência frágil:** 73% das regras de confiança acima de 90% (suporte 0,5%) se apoiam em menos de 150 compras.
- **Regras redundantes:** 38% das regras ligam variantes do mesmo produto, e as 20 regras de maior lift são todas de uma única família (marcadores de ervas).

---

## Limitações conhecidas

- Um único dataset: loja britânica, com 90,5% das compras do Reino Unido e muitos clientes atacadistas. Os resultados não se generalizam para outros varejos.
- Um ano de dados, com sazonalidade não tratada.
- Os tempos dependem da implementação (mlxtend), das versões das bibliotecas e da máquina. O suporte em que os algoritmos se cruzam muda entre ambientes.
- As regras mostram associação, não causa, e não foram validadas com resultados de vendas.
- O agrupamento de variantes do mesmo produto usa uma regra simples (50% ou mais de palavras em comum).

---

## Responsabilidades do grupo

| Integrante | Contribuição |
|---|---|
| Ana Júlia Soares de Oliveira | (preencher) |
| Arthur Pires Antunes | (preencher) |
| Diego Rian da Silva | (preencher) |
| Rafael Roberto de Jesus Ribeiro | (preencher) |

---

## Materiais

- Resumo expandido: [`Docs/Grupo_5_Resumo.pdf`](Docs/Grupo_5_Resumo.pdf)
- Diário de uso de IA: [`Docs/Grupo_5_Diario_de_IA.pdf`](Docs/Grupo_5_Diario_de_IA.pdf)
- Checklist do Checkpoint 1: [`Docs/Grupo_5_Checklist_27_10.pdf`](Docs/Grupo_5_Checklist_27_10.pdf)
- Referências: [`Referencias_artigo/`](Referencias_artigo/)

<!-- Adicionar aqui prints ou GIF do protótipo, por exemplo: ![Protótipo](Docs/img/app.png) -->
