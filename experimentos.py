"""
Experimentos E1 a E3 + analise de erros (E4) - Projeto IA II - Grupo 5
Apriori x FP-Growth no Online Retail

Como usar (raiz do projeto, junto do app.py e do eda_baseline.py):
    python experimentos.py            # completo (leva alguns minutos)
    python experimentos.py --rapido   # 1 repeticao e sem o suporte de 0,3%

Cada experimento imprime a HIPOTESE antes de rodar (a Ficha de Experimentos
exige hipotese registrada antes) e salva tabelas e graficos em resultados/.

IMPORTANTE: o script gera NUMEROS. A coluna "Interpretacao" da Ficha precisa
ser escrita pelo grupo, olhando os resultados reais.
"""

import argparse
import statistics
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

from eda_baseline import (
    baseline_mais_vendidos,
    carregar_dados,
    filtrar_validos,
    montar_cestas,
)

PASTA = Path("resultados")
CONFIANCA = 0.50          # confianca minima fixa nos experimentos
SUPORTE_REGRAS = 0.01     # suporte usado no E3 e nos erros 3
SUPORTES_E1 = [0.03, 0.02, 0.01, 0.005, 0.003]
SUPORTES_E2 = [0.05, 0.03, 0.02, 0.01, 0.005, 0.003]
LIMITE_ITEMSETS_REGRAS = 50_000  # acima disso nao gera regras (muito pesado)

linhas_resumo = []


def log(texto=""):
    print(texto, flush=True)
    linhas_resumo.append(texto)


# ---------------------------------------------------------------------------
# Preparo
# ---------------------------------------------------------------------------
def preparar():
    df = filtrar_validos(carregar_dados())
    cestas = montar_cestas(df)
    enc = TransactionEncoder()
    matriz = enc.fit(cestas).transform(cestas, sparse=True)
    binaria = pd.DataFrame.sparse.from_spmatrix(matriz, columns=enc.columns_)
    return df, cestas, binaria


def minerar(nome, binaria, suporte):
    """Roda so a mineracao de conjuntos frequentes e devolve (itemsets, segundos)."""
    inicio = time.perf_counter()
    if nome == "Apriori":
        fr = apriori(binaria, min_support=suporte, use_colnames=True, low_memory=True)
    else:
        fr = fpgrowth(binaria, min_support=suporte, use_colnames=True)
    return fr, time.perf_counter() - inicio


_cache = {}


def frequentes_cache(binaria, suporte):
    """Conjuntos frequentes (FP-Growth) reaproveitados entre experimentos."""
    if suporte not in _cache:
        _cache[suporte], _ = minerar("FP-Growth", binaria, suporte)
    return _cache[suporte]


def regras_de(fr):
    if len(fr) > LIMITE_ITEMSETS_REGRAS:
        return None
    return association_rules(fr, metric="confidence", min_threshold=CONFIANCA)


def nome_itens(conjunto):
    return ", ".join(sorted(conjunto))


# ---------------------------------------------------------------------------
# E1 - tempo: Apriori x FP-Growth
# ---------------------------------------------------------------------------
def experimento_1(binaria, repeticoes, suportes):
    log("\n" + "=" * 70)
    log("E1 - Apriori x FP-Growth: tempo de execucao")
    log("HIPOTESE: Apriori e FP-Growth encontram os mesmos conjuntos frequentes,")
    log("mas o tempo de execucao pode diferir conforme o suporte minimo.")
    log("ALTERACAO: o algoritmo (mesmos dados e mesmo suporte).")
    log(f"METODO: mediana de {repeticoes} execucao(oes) (1 para suporte < 0,5%); mede so a mineracao.")
    log("=" * 70)

    minerar("FP-Growth", binaria, 0.05)  # aquecimento (evita vies da 1a chamada)

    linhas = []
    for s in suportes:
        tempos, resultado = {}, {}
        # suportes muito baixos sao pesados (tempo e memoria): 1 execucao so
        reps = 1 if s < 0.005 else repeticoes
        for nome in ("Apriori", "FP-Growth"):
            ts = []
            for _ in range(reps):
                fr, t = minerar(nome, binaria, s)
                ts.append(t)
            tempos[nome] = statistics.median(ts)
            resultado[nome] = fr
        a, f = resultado["Apriori"], resultado["FP-Growth"]
        iguais = set(a["itemsets"]) == set(f["itemsets"])
        _cache[s] = f

        regras_iguais = "n/a"
        ra, rf = regras_de(a), regras_de(f)
        if ra is not None and rf is not None:
            par = lambda r: set(zip(r["antecedents"], r["consequents"]))
            regras_iguais = "sim" if par(ra) == par(rf) else "NAO"

        mais_rapido = "Apriori" if tempos["Apriori"] < tempos["FP-Growth"] else "FP-Growth"
        linhas.append(
            {
                "Suporte (%)": round(s * 100, 4),
                "Conjuntos frequentes": len(f),
                "Tempo Apriori (s)": round(tempos["Apriori"], 2),
                "Tempo FP-Growth (s)": round(tempos["FP-Growth"], 2),
                "Mais rapido": mais_rapido,
                "Mesmos conjuntos": "sim" if iguais else "NAO",
                "Mesmas regras": regras_iguais,
            }
        )
        log(
            f"suporte {s:.2%}: Apriori {tempos['Apriori']:.2f}s | "
            f"FP-Growth {tempos['FP-Growth']:.2f}s | conjuntos={len(f):,} | "
            f"iguais={'sim' if iguais else 'NAO'}"
        )

    tabela = pd.DataFrame(linhas)
    tabela.to_csv(PASTA / "e1_tempo.csv", index=False, sep=";", decimal=",", encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(9, 5))
    x = [f"{v:g}%" for v in tabela["Suporte (%)"]]
    ax.plot(x, tabela["Tempo Apriori (s)"], marker="o", label="Apriori", color="#bc4749")
    ax.plot(x, tabela["Tempo FP-Growth (s)"], marker="o", label="FP-Growth", color="#2a6f97")
    ax.set_title("E1: tempo de mineracao por suporte minimo")
    ax.set_xlabel("Suporte minimo")
    ax.set_ylabel("Tempo (s)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PASTA / "e1_tempo.png", dpi=150)
    plt.close(fig)
    return tabela


# ---------------------------------------------------------------------------
# E2 - variacao do suporte minimo
# ---------------------------------------------------------------------------
def experimento_2(binaria, suportes):
    log("\n" + "=" * 70)
    log("E2 - Variacao do suporte minimo")
    log("HIPOTESE: diminuir o suporte minimo aumenta a quantidade de conjuntos")
    log("frequentes e de regras (e que o crescimento seja bem mais que proporcional).")
    log(f"ALTERACAO: suporte minimo; confianca fixa em {CONFIANCA:.0%}.")
    log("=" * 70)

    linhas = []
    for s in suportes:
        fr = frequentes_cache(binaria, s)
        tamanhos = fr["itemsets"].apply(len)
        regras = regras_de(fr)
        linhas.append(
            {
                "Suporte (%)": round(s * 100, 4),
                "Compras minimas": int(round(s * len(binaria))),
                "Conjuntos frequentes": len(fr),
                "Com 1 item": int((tamanhos == 1).sum()),
                "Com 2 itens": int((tamanhos == 2).sum()),
                "Com 3+ itens": int((tamanhos >= 3).sum()),
                "Regras": len(regras) if regras is not None else "n/a (muito grande)",
            }
        )
        log(
            f"suporte {s:.2%}: conjuntos={len(fr):,} | 3+ itens={(tamanhos >= 3).sum():,} | "
            f"regras={len(regras) if regras is not None else 'n/a'}"
        )

    tabela = pd.DataFrame(linhas)
    tabela.to_csv(PASTA / "e2_suporte.csv", index=False, sep=";", decimal=",", encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(9, 5))
    x = [f"{v:g}%" for v in tabela["Suporte (%)"]]
    ax.bar(x, tabela["Conjuntos frequentes"], color="#2a6f97")
    ax.set_yscale("log")
    ax.set_title("E2: conjuntos frequentes por suporte minimo (escala log)")
    ax.set_xlabel("Suporte minimo")
    ax.set_ylabel("Conjuntos frequentes")
    fig.tight_layout()
    fig.savefig(PASTA / "e2_suporte.png", dpi=150)
    plt.close(fig)
    return tabela


# ---------------------------------------------------------------------------
# E3 - regras x baseline
# ---------------------------------------------------------------------------
def experimento_3(df, binaria):
    log("\n" + "=" * 70)
    log("E3 - Regras de associacao x baseline (mais vendidos)")
    log("HIPOTESE: as regras revelam relacoes entre produtos que o ranking dos")
    log("mais vendidos nao mostra.")
    log(f"ALTERACAO: comparar regras (suporte {SUPORTE_REGRAS:.0%}, confianca {CONFIANCA:.0%})")
    log("com o ranking de popularidade, usando suporte, confianca e lift.")
    log("=" * 70)

    fr = frequentes_cache(binaria, SUPORTE_REGRAS)
    regras = association_rules(fr, metric="confidence", min_threshold=CONFIANCA)
    top10 = set(baseline_mais_vendidos(df, 10)["Produto"])
    top50 = set(baseline_mais_vendidos(df, 50)["Produto"])

    regras = regras.copy()
    regras["cons_no_top10"] = regras["consequents"].apply(lambda s: bool(s & top10))
    regras["itens"] = regras.apply(lambda r: r["antecedents"] | r["consequents"], axis=1)
    regras["fora_top50"] = regras["itens"].apply(lambda s: not (s & top50))

    itens_regras = set().union(*regras["itens"])
    fora = itens_regras - top50
    log(f"Regras geradas: {len(regras):,}")
    log(f"Lift: mediana {regras['lift'].median():.1f} | minimo {regras['lift'].min():.2f} | maximo {regras['lift'].max():.1f}")
    log(f"Regras com lift > 1: {(regras['lift'] > 1).sum():,} ({(regras['lift'] > 1).mean():.0%})")
    log(f"Regras cujo consequente esta no top 10 do baseline: {regras['cons_no_top10'].sum():,} ({regras['cons_no_top10'].mean():.0%})")
    log(f"Produtos distintos nas regras: {len(itens_regras):,} | fora do top 50 do baseline: {len(fora):,} ({len(fora) / len(itens_regras):.0%})")
    log(f"Regras formadas so por produtos fora do top 50: {regras['fora_top50'].sum():,}")

    top = regras.sort_values("lift", ascending=False).head(20).copy()
    saida = pd.DataFrame(
        {
            "Se comprar": top["antecedents"].apply(nome_itens),
            "Tambem compra": top["consequents"].apply(nome_itens),
            "Suporte (%)": (top["support"] * 100).round(2),
            "Confianca (%)": (top["confidence"] * 100).round(1),
            "Lift": top["lift"].round(1),
            "Consequente no top10 baseline": top["cons_no_top10"].map({True: "sim", False: "nao"}),
        }
    )
    saida.to_csv(PASTA / "e3_top20_regras_por_lift.csv", index=False, sep=";", decimal=",", encoding="utf-8-sig")
    log("\nTop 5 regras por lift:")
    for _, r in saida.head(5).iterrows():
        log(f"  {r['Se comprar']}  ->  {r['Tambem compra']}  (conf {r['Confianca (%)']}%, lift {r['Lift']})")
    return regras


# ---------------------------------------------------------------------------
# E4 - tres erros / limitacoes
# ---------------------------------------------------------------------------
def palavras(texto):
    return set(texto.replace(",", " ").split())


def jaccard(a, b):
    pa, pb = palavras(a), palavras(b)
    return len(pa & pb) / len(pa | pb) if pa | pb else 0.0


def experimento_4(binaria, regras_1pct, tabela_e2):
    log("\n" + "=" * 70)
    log("E4 - Analise de tres erros / limitacoes")
    log("=" * 70)

    # Erro 1: suporte minimo muito baixo
    log("\n[Erro 1] Suporte minimo muito baixo")
    log("HIPOTESE: com suporte muito baixo, o numero de conjuntos e regras explode")
    log("e a lista fica grande demais para alguem analisar.")
    t = tabela_e2.set_index("Suporte (%)")
    for s in (1.0, 0.5, 0.3):
        if s in t.index:
            log(f"  suporte {s:g}%: {t.loc[s, 'Conjuntos frequentes']:,} conjuntos | regras: {t.loc[s, 'Regras']}")

    # Erro 2: dados esparsos / evidencia fragil
    log("\n[Erro 2] Dados esparsos: regras fortes no papel, mas com pouca evidencia")
    log("HIPOTESE: com suporte baixo, aparecem regras com confianca alta baseadas")
    log("em poucas compras, que podem ser coincidencia.")
    fr = frequentes_cache(binaria, 0.005)
    regras = association_rules(fr, metric="confidence", min_threshold=CONFIANCA)
    n = len(binaria)
    regras = regras.copy()
    regras["compras"] = (regras["support"] * n).round().astype(int)
    fortes = regras[regras["confidence"] >= 0.90]
    frageis = fortes[fortes["compras"] < 150]
    log(f"  Regras com confianca >= 90% (suporte 0,5%): {len(fortes):,}")
    log(f"  Dessas, apoiadas em menos de 150 compras:   {len(frageis):,} ({len(frageis) / max(len(fortes), 1):.0%})")
    exemplo = frageis.sort_values("lift", ascending=False).head(15)
    pd.DataFrame(
        {
            "Se comprar": exemplo["antecedents"].apply(nome_itens),
            "Tambem compra": exemplo["consequents"].apply(nome_itens),
            "Compras em que aparece": exemplo["compras"],
            "Confianca (%)": (exemplo["confidence"] * 100).round(1),
            "Lift": exemplo["lift"].round(1),
        }
    ).to_csv(PASTA / "e4_regras_frageis.csv", index=False, sep=";", decimal=",", encoding="utf-8-sig")

    # Erro 3: regras obvias / redundantes
    log("\n[Erro 3] Regras obvias ou redundantes")
    log("HIPOTESE: parte das regras liga variantes do mesmo produto (cores/modelos)")
    log("ou aparece duplicada em pares A->B e B->A, sem informacao nova.")
    r = regras_1pct.copy()

    def mesma_familia(linha):
        return any(
            jaccard(a, c) >= 0.5 for a in linha["antecedents"] for c in linha["consequents"]
        )

    r["mesma_familia"] = r.apply(mesma_familia, axis=1)
    pares = set(zip(r["antecedents"], r["consequents"]))
    r["espelho"] = [(c, a) in pares for a, c in zip(r["antecedents"], r["consequents"])]
    log(f"  Regras (suporte 1%, confianca 50%): {len(r):,}")
    log(f"  Regras entre variantes do mesmo produto (heuristica: >= 50% das palavras em comum): {r['mesma_familia'].sum():,} ({r['mesma_familia'].mean():.0%})")
    log(f"  Regras que tem a regra espelho (B->A tambem existe): {r['espelho'].sum():,} ({r['espelho'].mean():.0%})")
    ex = r[r["mesma_familia"]].sort_values("lift", ascending=False).head(15)
    pd.DataFrame(
        {
            "Se comprar": ex["antecedents"].apply(nome_itens),
            "Tambem compra": ex["consequents"].apply(nome_itens),
            "Confianca (%)": (ex["confidence"] * 100).round(1),
            "Lift": ex["lift"].round(1),
        }
    ).to_csv(PASTA / "e4_regras_mesma_familia.csv", index=False, sep=";", decimal=",", encoding="utf-8-sig")


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rapido", action="store_true", help="1 repeticao e sem suporte de 0,3%%")
    args = parser.parse_args()

    PASTA.mkdir(exist_ok=True)
    repeticoes = 1 if args.rapido else 3
    s1 = [s for s in SUPORTES_E1 if not (args.rapido and s < 0.005)]
    s2 = [s for s in SUPORTES_E2 if not (args.rapido and s < 0.005)]

    df, cestas, binaria = preparar()
    log(f"Compras: {len(cestas):,} | Produtos: {binaria.shape[1]:,} | Confianca fixa: {CONFIANCA:.0%}")

    experimento_1(binaria, repeticoes, s1)
    tabela_e2 = experimento_2(binaria, s2)
    regras_1pct = experimento_3(df, binaria)
    experimento_4(binaria, regras_1pct, tabela_e2)

    (PASTA / "experimentos_resumo.txt").write_text("\n".join(linhas_resumo), encoding="utf-8")
    print(f"\nArquivos salvos em: {PASTA.resolve()}")


if __name__ == "__main__":
    main()
