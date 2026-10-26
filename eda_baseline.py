"""
EDA e Baseline - Projeto IA II (Apriori / FP-Growth) - Grupo 5

Como usar (na raiz do projeto, a mesma pasta do app.py):
    python eda_baseline.py

O que o script faz:
  1. Carrega dados/Base_Analise.csv (separador ";" e decimal ",").
  2. Aplica os filtros finais (Quantity > 0, UnitPrice > 0, sem cancelamentos).
  3. EDA: imprime os numeros e salva graficos em resultados/.
  4. BASELINE: ranking dos produtos mais vendidos (salvo em CSV).

Saidas em resultados/:
  eda_resumo.txt, baseline_top_produtos.csv e graficos .png

A funcao baseline_mais_vendidos() tambem pode ser importada pelo app.py.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # gera as imagens sem abrir janela
import matplotlib.pyplot as plt
import pandas as pd

CAMINHO_CSV = Path("dados") / "Base_Analise.csv"
PASTA_SAIDA = Path("resultados")


# ---------------------------------------------------------------------------
# Carga e preparo
# ---------------------------------------------------------------------------
def carregar_dados(caminho=CAMINHO_CSV):
    """Le o CSV corretamente (separador ; e decimal ,)."""
    df = pd.read_csv(caminho, encoding="utf-8-sig", sep=";", decimal=",")
    df["Description"] = df["Description"].astype("string").str.strip()
    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"], dayfirst=True, errors="coerce"
    )
    return df


def filtrar_validos(df):
    """Filtros finais definidos pelo grupo para formar as cestas."""
    return df[
        (df["Quantity"] > 0)
        & (df["UnitPrice"] > 0)
        & (df["Description"].notna())
        & (df["Description"] != "")
        & (~df["InvoiceNo"].astype(str).str.startswith("C"))
    ].copy()


def montar_cestas(df):
    """Uma cesta (lista de produtos sem repeticao) por compra."""
    return df.groupby("InvoiceNo")["Description"].apply(
        lambda produtos: sorted(set(produtos))
    )


# ---------------------------------------------------------------------------
# BASELINE: ranking dos produtos mais vendidos
# ---------------------------------------------------------------------------
def baseline_mais_vendidos(df, n=10):
    """
    Ranking por popularidade individual.

    - Compras: em quantas compras (InvoiceNo) o produto aparece.
    - Pct_compras: Compras / total de compras. E o SUPORTE de 1 item,
      por isso e comparavel direto com o suporte do Apriori/FP-Growth.
    - Unidades: soma de Quantity (volume vendido).

    O ranking principal usa 'Compras', porque Apriori e FP-Growth tambem
    contam presenca na cesta, nao quantidade.
    """
    total_compras = df["InvoiceNo"].nunique()
    ranking = (
        df.groupby("Description")
        .agg(Compras=("InvoiceNo", "nunique"), Unidades=("Quantity", "sum"))
        .sort_values("Compras", ascending=False)
        .head(n)
        .reset_index()
        .rename(columns={"Description": "Produto"})
    )
    ranking["Pct_compras"] = (ranking["Compras"] / total_compras * 100).round(2)
    ranking.insert(0, "Posicao", range(1, len(ranking) + 1))
    return ranking


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------
def grafico_barras_h(serie, titulo, xlabel, arquivo, cor="#2a6f97"):
    fig, ax = plt.subplots(figsize=(9, 5))
    serie.sort_values().plot.barh(ax=ax, color=cor)
    ax.set_title(titulo)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(arquivo, dpi=150)
    plt.close(fig)


def executar_eda(bruto, df, cestas, linhas):
    """Calcula as descobertas, imprime e salva graficos."""

    def log(texto=""):
        print(texto)
        linhas.append(texto)

    # -- Visao geral ---------------------------------------------------------
    log("=== VISAO GERAL ===")
    log(f"Linhas no CSV de trabalho:      {len(bruto):,}")
    log(f"Linhas validas (filtros finais): {len(df):,}")
    log(f"Compras (InvoiceNo):             {df['InvoiceNo'].nunique():,}")
    log(f"Produtos distintos (Description): {df['Description'].nunique():,}")
    log(f"Clientes identificados:          {df['CustomerID'].nunique():,}")
    log(
        f"Periodo: {df['InvoiceDate'].min():%d/%m/%Y} a "
        f"{df['InvoiceDate'].max():%d/%m/%Y}"
    )

    # -- Descoberta 1: tamanho das cestas -----------------------------------
    tamanhos = cestas.apply(len)
    unitarias = int((tamanhos == 1).sum())
    log("\n=== DESCOBERTA 1: TAMANHO DAS CESTAS ===")
    log(f"Mediana de itens por compra: {tamanhos.median():.0f}")
    log(f"Media de itens por compra:   {tamanhos.mean():.1f}")
    log(f"Maior cesta:                 {tamanhos.max()} itens")
    log(
        f"Compras com 1 item so:       {unitarias:,} "
        f"({unitarias / len(cestas) * 100:.1f}%) -> nao geram regra"
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(tamanhos.clip(upper=100), bins=50, color="#2a6f97")
    ax.set_title("Itens distintos por compra (valores acima de 100 agrupados em 100)")
    ax.set_xlabel("Itens por compra")
    ax.set_ylabel("Numero de compras")
    fig.tight_layout()
    fig.savefig(PASTA_SAIDA / "eda_tamanho_cestas.png", dpi=150)
    plt.close(fig)

    # -- Descoberta 2: concentracao por pais --------------------------------
    por_pais = df.groupby("Country")["InvoiceNo"].nunique().sort_values(ascending=False)
    pct_uk = por_pais.get("United Kingdom", 0) / por_pais.sum() * 100
    log("\n=== DESCOBERTA 2: PAISES ===")
    log(f"Paises distintos: {len(por_pais)}")
    log(f"Reino Unido: {pct_uk:.1f}% das compras")
    log("Top 5 paises (compras):")
    for pais, qtd in por_pais.head(5).items():
        log(f"  {pais}: {qtd:,}")
    grafico_barras_h(
        por_pais.head(8),
        "Compras por pais (top 8)",
        "Numero de compras",
        PASTA_SAIDA / "eda_paises.png",
    )

    # -- Descoberta 3: calda longa dos produtos -----------------------------
    freq = df.groupby("Description")["InvoiceNo"].nunique().sort_values(ascending=False)
    raros = int((freq < 20).sum())
    freq_1pct = int((freq / len(cestas) >= 0.01).sum())
    freq_2pct = int((freq / len(cestas) >= 0.02).sum())
    log("\n=== DESCOBERTA 3: CALDA LONGA DE PRODUTOS ===")
    log(f"Produtos em menos de 20 compras: {raros:,} de {len(freq):,}")
    log(f"Produtos com suporte >= 1%:      {freq_1pct:,}")
    log(f"Produtos com suporte >= 2%:      {freq_2pct:,}")
    log("-> a maioria dos produtos e rara; a matriz e muito esparsa.")

    n_itens = len(freq)
    densidade = freq.sum() / (len(cestas) * n_itens) * 100
    log(f"Densidade da matriz compra x produto: {densidade:.2f}% de celulas ocupadas")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(1, len(freq) + 1), freq.values, color="#2a6f97")
    ax.set_title("Popularidade dos produtos (compras em que aparecem)")
    ax.set_xlabel("Posicao no ranking")
    ax.set_ylabel("Numero de compras")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(PASTA_SAIDA / "eda_calda_longa.png", dpi=150)
    plt.close(fig)

    # -- Qualidade dos dados -------------------------------------------------
    log("\n=== QUALIDADE DOS DADOS ===")
    sem_cliente = bruto["CustomerID"].isna().sum()
    log(
        f"Linhas sem CustomerID: {sem_cliente:,} "
        f"({sem_cliente / len(bruto) * 100:.1f}%) "
        "-> nao atrapalha: as cestas sao por InvoiceNo"
    )
    duplicadas = int(bruto.duplicated().sum())
    log(
        f"Linhas totalmente duplicadas: {duplicadas:,} "
        "-> viram 1 item na cesta (usamos set); decidir se remove ou mantem"
    )
    n_canc = int(bruto["InvoiceNo"].astype(str).str.startswith("C").sum())
    log(f"Cancelamentos (InvoiceNo com C) ainda presentes: {n_canc:,}")
    n_qtd = int((bruto["Quantity"] <= 0).sum())
    n_preco = int((bruto["UnitPrice"] <= 0).sum())
    log(f"Quantity <= 0: {n_qtd:,} | UnitPrice <= 0: {n_preco:,}")

    # -- Volume ao longo do tempo (extra) -----------------------------------
    por_mes = (
        df.set_index("InvoiceDate").resample("MS")["InvoiceNo"].nunique()
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(por_mes.index.strftime("%m/%y"), por_mes.values, color="#2a6f97")
    ax.set_title("Compras por mes")
    ax.set_ylabel("Numero de compras")
    plt.setp(ax.get_xticklabels(), rotation=45)
    fig.tight_layout()
    fig.savefig(PASTA_SAIDA / "eda_compras_por_mes.png", dpi=150)
    plt.close(fig)
    pico = por_mes.idxmax()
    log(f"\nMes com mais compras: {pico:%m/%Y} ({por_mes.max():,} compras)")


# ---------------------------------------------------------------------------
# Execucao
# ---------------------------------------------------------------------------
def main():
    if not CAMINHO_CSV.exists():
        raise SystemExit(f"CSV nao encontrado em {CAMINHO_CSV}. Rode na raiz do projeto.")

    PASTA_SAIDA.mkdir(exist_ok=True)
    linhas = []

    bruto = carregar_dados()
    df = filtrar_validos(bruto)
    cestas = montar_cestas(df)

    executar_eda(bruto, df, cestas, linhas)

    # Baseline
    ranking = baseline_mais_vendidos(df, n=10)
    print("\n=== BASELINE: TOP 10 PRODUTOS MAIS VENDIDOS ===")
    print(ranking.to_string(index=False))
    linhas.append("\n=== BASELINE: TOP 10 PRODUTOS MAIS VENDIDOS ===")
    linhas.append(ranking.to_string(index=False))

    ranking.to_csv(
        PASTA_SAIDA / "baseline_top_produtos.csv",
        index=False,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
    )
    grafico_barras_h(
        ranking.set_index("Produto")["Compras"],
        "Baseline: 10 produtos em mais compras",
        "Numero de compras",
        PASTA_SAIDA / "baseline_top10.png",
        cor="#bc4749",
    )

    (PASTA_SAIDA / "eda_resumo.txt").write_text("\n".join(linhas), encoding="utf-8")
    print(f"\nArquivos salvos em: {PASTA_SAIDA.resolve()}")


if __name__ == "__main__":
    main()
