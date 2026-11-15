"""
Protótipo web - Projeto IA II - Grupo 5
Quais produtos aparecem juntos? Apriori x FP-Growth no Online Retail.

Como executar (na pasta principal do projeto):
    streamlit run app.py

Abas:
  🛍️ Loja         escolha um produto e veja o que costuma ser comprado junto,
                  com a explicação de cada sugestão.
  🔬 Laboratório  rode Apriori, FP-Growth ou os dois, ajustando suporte e
                  confiança, e veja conjuntos, regras e tempo de execução.
  📊 Os dados     visão geral da base e baseline (produtos mais populares).
"""

import time
from pathlib import Path

import pandas as pd
import streamlit as st
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

from algoritmos.apriori import executar_apriori
from algoritmos.fp_growth import executar_fp_growth
from eda_baseline import (
    CAMINHO_CSV,
    baseline_mais_vendidos,
    carregar_dados,
    filtrar_validos,
    montar_cestas,
)

# Regras com menos compras que isso são marcadas como "pouca evidência"
# (mesmo critério do experimento E4).
MIN_COMPRAS_EVIDENCIA = 150

# Parâmetros fixos da aba Loja: FP-Growth, regras de 1 produto -> 1 produto.
LOJA_SUPORTE = 0.005
LOJA_CONFIANCA = 0.30


# ---------------------------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Apriori e FP-Growth",
    page_icon="🛒",
    layout="wide",
)

st.title("🛒 Quais produtos aparecem juntos?")
st.write(
    "Descubra quais produtos costumam ser comprados na mesma compra "
    "usando os algoritmos **Apriori** e **FP-Growth**."
)


# ---------------------------------------------------------------------------
# Carga dos dados (feita uma vez e guardada em cache)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Carregando as compras...")
def carregar_base():
    df = filtrar_validos(carregar_dados())
    cestas = montar_cestas(df)
    return df, cestas


@st.cache_resource(show_spinner="Preparando a matriz de compras...")
def matriz_binaria(_cestas):
    encoder = TransactionEncoder()
    matriz = encoder.fit(_cestas).transform(_cestas, sparse=True)
    return pd.DataFrame.sparse.from_spmatrix(matriz, columns=encoder.columns_)


if not Path(CAMINHO_CSV).exists():
    st.error(
        "CSV não encontrado. Confira se Base_Analise.csv "
        "está dentro da pasta dados."
    )
    st.stop()

try:
    df, compras = carregar_base()
except Exception as erro:
    st.error(f"Não foi possível ler o CSV: {erro}")
    st.stop()

binaria = matriz_binaria(compras)
total_compras = len(compras)

# Em quantas compras cada produto aparece (usado nas explicações)
compras_por_produto = (
    df.groupby("Description")["InvoiceNo"].nunique().sort_values(ascending=False)
)


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------
def texto_itens(itens):
    return ", ".join(sorted(itens))


def palavras(texto):
    return set(texto.replace(",", " ").split())


def mesma_familia(a, b):
    """Variantes do mesmo produto: 50% ou mais de palavras em comum (igual ao E4)."""
    pa, pb = palavras(a), palavras(b)
    return len(pa & pb) / len(pa | pb) >= 0.5 if pa | pb else False


def regra_mesma_familia(antecedentes, consequentes):
    return any(mesma_familia(a, c) for a in antecedentes for c in consequentes)


def formatar_numero(valor):
    return f"{valor:,}".replace(",", ".")


def formatar_pct(valor, casas=1):
    return f"{valor * 100:.{casas}f}".replace(".", ",") + "%"


@st.cache_resource(show_spinner="Procurando produtos que combinam (FP-Growth)...")
def regras_da_loja(_binaria):
    """Regras de 1 produto -> 1 produto, calculadas uma vez para a aba Loja."""
    conjuntos = fpgrowth(
        _binaria, min_support=LOJA_SUPORTE, use_colnames=True, max_len=2
    )
    regras = association_rules(
        conjuntos, metric="confidence", min_threshold=LOJA_CONFIANCA
    )
    regras = regras.copy()
    regras["produto"] = regras["antecedents"].apply(lambda s: next(iter(s)))
    regras["sugestao"] = regras["consequents"].apply(lambda s: next(iter(s)))
    return regras[
        ["produto", "sugestao", "support", "confidence", "lift", "consequent support"]
    ]


# ---------------------------------------------------------------------------
# Abas
# ---------------------------------------------------------------------------
aba_loja, aba_lab, aba_dados = st.tabs(
    ["🛍️ Loja", "🔬 Laboratório", "📊 Os dados"]
)


# ===========================================================================
# ABA 1 - LOJA
# ===========================================================================
with aba_loja:
    st.header("🛍️ O que levar junto?")
    st.write(
        "Escolha um produto como se estivesse colocando no carrinho. "
        "O sistema mostra o que outros clientes costumam comprar junto e "
        "explica cada sugestão."
    )

    regras_loja = regras_da_loja(binaria)
    com_sugestao = set(regras_loja["produto"])

    so_com_regra = st.toggle(
        "Mostrar só produtos que têm sugestões",
        value=True,
        help=(
            "Desligue para escolher qualquer um dos produtos da loja, "
            "inclusive os que não geram regras."
        ),
    )

    if so_com_regra:
        opcoes = [p for p in compras_por_produto.index if p in com_sugestao]
    else:
        opcoes = list(compras_por_produto.index)

    produto = st.selectbox(
        f"Produto no carrinho ({formatar_numero(len(opcoes))} opções, "
        "dá para digitar para buscar)",
        opcoes,
    )

    n_produto = int(compras_por_produto.get(produto, 0))
    st.caption(
        f"**{produto}** aparece em {formatar_numero(n_produto)} "
        f"{'compra' if n_produto == 1 else 'compras'} "
        f"({formatar_pct(n_produto / total_compras)} de todas)."
    )

    sugestoes = (
        regras_loja[regras_loja["produto"] == produto]
        .sort_values("lift", ascending=False)
        .head(5)
    )

    if sugestoes.empty:
        minimo = int(round(LOJA_SUPORTE * total_compras))
        if n_produto < minimo:
            motivo = (
                f"Esse produto aparece em só {formatar_numero(n_produto)} "
                f"{'compra' if n_produto == 1 else 'compras'}, "
                f"abaixo do suporte mínimo usado aqui ({formatar_pct(LOJA_SUPORTE)}, "
                f"ou seja, {formatar_numero(minimo)} compras). Com tão poucos "
                "dados, qualquer regra seria pura coincidência."
            )
        else:
            motivo = (
                "Esse produto é frequente, mas é comprado com muitos produtos "
                "diferentes: nenhum deles aparece junto em pelo menos "
                f"{formatar_pct(LOJA_CONFIANCA, 0)} das compras."
            )
        st.warning(f"**Nenhuma regra forte para este produto.** {motivo}")

        st.write("Quando não há regra, a alternativa é o baseline, os mais populares da loja:")
        populares = [p for p in compras_por_produto.index if p != produto][:5]
        for p in populares:
            st.write(
                f"- {p} ({formatar_pct(compras_por_produto[p] / total_compras)} "
                "das compras)"
            )
        st.caption(
            "Repare que o baseline sugere a mesma coisa para qualquer produto. "
            "As regras de associação, quando existem, mudam conforme o carrinho."
        )

    else:
        st.subheader("Quem compra este produto também leva:")

        for _, r in sugestoes.iterrows():
            n_juntos = int(round(r["support"] * total_compras))
            fragil = n_juntos < MIN_COMPRAS_EVIDENCIA
            variante = mesma_familia(produto, r["sugestao"])
            lift_txt = f"{r['lift']:.1f}".replace(".", ",")

            with st.container(border=True):
                c1, c2, c3 = st.columns([3, 1, 1])
                c1.markdown(f"#### {r['sugestao']}")
                c2.metric("Confiança", formatar_pct(r["confidence"], 0))
                c3.metric("Lift", f"{lift_txt}×")

                st.write(
                    f"Em **{formatar_pct(r['confidence'], 0)}** das compras com "
                    f"o produto escolhido, este também estava "
                    f"({formatar_numero(n_juntos)} compras). No geral, ele aparece "
                    f"em só {formatar_pct(r['consequent support'])} das compras, então a "
                    f"chance de vir junto é **{lift_txt} vezes maior** que o normal."
                )

                avisos = []
                if fragil:
                    avisos.append(
                        f"⚠️ Pouca evidência: baseada em menos de "
                        f"{MIN_COMPRAS_EVIDENCIA} compras."
                    )
                if variante:
                    avisos.append(
                        "🔁 Parece outra versão do mesmo produto (cor ou modelo), "
                        "então a regra é meio óbvia."
                    )
                for aviso in avisos:
                    st.caption(aviso)

    with st.expander("Como estas sugestões são calculadas?"):
        st.write(
            f"- Algoritmo: **FP-Growth**, com suporte mínimo de "
            f"{formatar_pct(LOJA_SUPORTE)} e confiança mínima de "
            f"{formatar_pct(LOJA_CONFIANCA, 0)}.\n"
            "- Só regras de 1 produto para 1 produto, ordenadas por lift.\n"
            "- **Confiança:** entre as compras com o produto escolhido, em quantas "
            "a sugestão também estava.\n"
            "- **Lift:** quantas vezes mais a sugestão aparece quando o produto "
            "está no carrinho, comparado com o normal. Acima de 1 = associação "
            "positiva.\n"
            "- As regras mostram **associação, não causa**: comprar um produto "
            "não faz a pessoa comprar o outro."
        )
        st.caption(
            f"{formatar_numero(len(regras_loja))} regras, cobrindo "
            f"{formatar_numero(len(com_sugestao))} dos "
            f"{formatar_numero(len(compras_por_produto))} produtos."
        )


# ===========================================================================
# ABA 2 - LABORATÓRIO
# ===========================================================================
def tabela_regras(regras):
    tabela = pd.DataFrame(
        {
            "Se comprar": regras["antecedents"].apply(texto_itens),
            "Também pode comprar": regras["consequents"].apply(texto_itens),
            "Suporte (%)": (regras["support"] * 100).round(2),
            "Confiança (%)": (regras["confidence"] * 100).round(1),
            "Lift": regras["lift"].round(2),
            "Compras": (regras["support"] * total_compras).round().astype(int),
        }
    )
    tabela["Pouca evidência"] = tabela["Compras"] < MIN_COMPRAS_EVIDENCIA
    tabela["Mesmo produto"] = [
        regra_mesma_familia(a, c)
        for a, c in zip(regras["antecedents"], regras["consequents"])
    ]
    return tabela.sort_values("Lift", ascending=False).reset_index(drop=True)


def minerar(nome, suporte, confianca):
    inicio = time.perf_counter()
    if nome == "Apriori":
        frequentes, regras = executar_apriori(binaria, suporte, confianca)
    else:
        frequentes, regras = executar_fp_growth(binaria, suporte, confianca)
    return frequentes, regras, time.perf_counter() - inicio


with aba_lab:
    st.header("🔬 Laboratório: compare os algoritmos")
    st.write(
        "Ajuste os parâmetros e rode a análise completa. Os dois algoritmos "
        "devem encontrar os mesmos resultados; o que muda é o tempo."
    )

    col_a, col_b, col_c = st.columns(3)

    modo = col_a.radio(
        "Algoritmo",
        ["Apriori", "FP-Growth", "Comparar os dois"],
    )

    suporte = col_b.slider(
        "Suporte mínimo (%)",
        min_value=0.5,
        max_value=10.0,
        value=1.0,
        step=0.1,
        help=(
            "Abaixo de 0,5% o número de combinações explode (157.933 conjuntos "
            "em 0,3%) e a análise fica lenta demais para uma demonstração."
        ),
    )

    confianca = col_c.slider(
        "Confiança mínima (%)",
        min_value=10,
        max_value=100,
        value=50,
        step=5,
    )

    st.caption(
        f"Suporte de {str(suporte).replace('.', ',')}% = os produtos precisam aparecer "
        f"juntos em pelo menos {formatar_numero(int(round(suporte / 100 * total_compras)))} "
        "compras. Confiança = em quantas compras com o antecedente o consequente "
        "também aparece."
    )

    if suporte < 0.8:
        st.warning(
            "Suporte baixo: a análise pode levar alguns segundos e gerar "
            "milhares de regras, muitas com pouca evidência."
        )

    f1, f2 = st.columns(2)
    esconder_frageis = f1.checkbox(
        f"Esconder regras com pouca evidência (menos de {MIN_COMPRAS_EVIDENCIA} compras)"
    )
    esconder_variantes = f2.checkbox(
        "Esconder regras entre versões do mesmo produto"
    )

    if st.button("Executar análise", type="primary"):
        s, c = suporte / 100, confianca / 100

        try:
            with st.spinner("Analisando as compras..."):
                if modo == "Comparar os dois":
                    fa, ra, ta = minerar("Apriori", s, c)
                    ff, rf, tf = minerar("FP-Growth", s, c)
                    frequentes, regras = ff, rf
                else:
                    frequentes, regras, tempo = minerar(modo, s, c)

            st.divider()
            st.subheader("⏱️ Desempenho")

            if modo == "Comparar os dois":
                mesmos = set(fa["itemsets"]) == set(ff["itemsets"])
                m1, m2, m3 = st.columns(3)
                m1.metric("Apriori", f"{ta:.2f} s".replace(".", ","))
                m2.metric("FP-Growth", f"{tf:.2f} s".replace(".", ","))
                m3.metric("Mesmos conjuntos?", "Sim ✅" if mesmos else "Não ❌")
                mais_rapido = "Apriori" if ta < tf else "FP-Growth"
                razao = max(ta, tf) / min(ta, tf) if min(ta, tf) > 0 else 1
                razao_txt = f"{razao:.1f}".replace(".", ",")
                st.info(
                    f"Com suporte de {str(suporte).replace('.', ',')}%, o "
                    f"**{mais_rapido}** foi {razao_txt} vezes mais rápido nesta execução. "
                    "Nos nossos experimentos, o Apriori ganha com suporte alto e o "
                    "FP-Growth passa à frente por volta de 1%. Os tempos mudam de "
                    "máquina para máquina."
                )
            else:
                m1, m2 = st.columns(2)
                m1.metric("Algoritmo", modo)
                m2.metric("Tempo de execução", f"{tempo:.2f} s".replace(".", ","))

            if frequentes is None or frequentes.empty:
                st.warning(
                    "Nenhum conjunto frequente encontrado. Tente diminuir o suporte mínimo."
                )
            else:
                st.subheader(f"📦 Conjuntos frequentes: {formatar_numero(len(frequentes))}")
                tabela_freq = pd.DataFrame(
                    {
                        "Produtos": frequentes["itemsets"].apply(texto_itens),
                        "Nº de produtos": frequentes["itemsets"].apply(len),
                        "Suporte (%)": (frequentes["support"] * 100).round(2),
                    }
                ).sort_values("Suporte (%)", ascending=False)
                with st.expander("Ver conjuntos frequentes"):
                    st.dataframe(tabela_freq, width="stretch", hide_index=True)

                st.subheader("🔗 Regras de associação")

                if regras is None or regras.empty:
                    st.warning(
                        "Nenhuma regra atingiu a confiança mínima. "
                        "Tente diminuir a confiança ou o suporte."
                    )
                else:
                    tabela = tabela_regras(regras)
                    total = len(tabela)
                    n_frageis = int(tabela["Pouca evidência"].sum())
                    n_variantes = int(tabela["Mesmo produto"].sum())

                    if esconder_frageis:
                        tabela = tabela[~tabela["Pouca evidência"]]
                    if esconder_variantes:
                        tabela = tabela[~tabela["Mesmo produto"]]

                    r1, r2, r3 = st.columns(3)
                    r1.metric("Regras geradas", formatar_numero(total))
                    r2.metric(
                        "Com pouca evidência",
                        f"{formatar_numero(n_frageis)} ({n_frageis / total:.0%})",
                    )
                    r3.metric(
                        "Entre versões do mesmo produto",
                        f"{formatar_numero(n_variantes)} ({n_variantes / total:.0%})",
                    )

                    st.caption(
                        f"Mostrando {formatar_numero(len(tabela))} regras, ordenadas por lift."
                    )
                    st.dataframe(
                        tabela,
                        width="stretch",
                        hide_index=True,
                        column_config={
                            "Pouca evidência": st.column_config.CheckboxColumn(
                                "⚠️ Pouca evidência"
                            ),
                            "Mesmo produto": st.column_config.CheckboxColumn(
                                "🔁 Mesmo produto"
                            ),
                        },
                    )

                    if not tabela.empty:
                        primeira = tabela.iloc[0]
                        st.subheader("💡 Como ler a primeira regra")
                        st.write(
                            f"Quando alguém compra **{primeira['Se comprar']}**, "
                            f"a regra sugere **{primeira['Também pode comprar']}**."
                        )
                        st.write(
                            f"- **Suporte:** {primeira['Suporte (%)']}% das compras "
                            f"({formatar_numero(int(primeira['Compras']))}) têm todos "
                            "os produtos da regra.\n"
                            f"- **Confiança:** em {primeira['Confiança (%)']}% das "
                            "compras com o antecedente, o consequente também aparece.\n"
                            f"- **Lift:** {primeira['Lift']}. Acima de 1, os produtos "
                            "aparecem juntos mais do que o esperado por acaso."
                        )

        except Exception as erro:
            st.error(f"Erro durante a análise: {erro}")


# ===========================================================================
# ABA 3 - OS DADOS
# ===========================================================================
with aba_dados:
    st.header("📊 Conhecendo as compras")
    st.write(
        "Antes de procurar combinações de produtos, vamos conhecer os dados "
        "que estamos analisando: a base **Online Retail** (UCI), com compras de "
        "uma loja online do Reino Unido entre 12/2010 e 12/2011."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Registros na base", formatar_numero(len(df)))
    col2.metric("Compras analisadas", formatar_numero(total_compras))
    col3.metric("Produtos diferentes", formatar_numero(df["Description"].nunique()))
    col4.metric("Itens por compra (mediana)", f"{compras.apply(len).median():.0f}")

    st.info(
        "Cada compra pode conter vários produtos. Nosso objetivo é descobrir "
        "quais produtos aparecem juntos, e não apenas quais são os mais populares."
    )

    with st.expander("Ver prévia dos dados"):
        st.dataframe(df.head(10), width="stretch")

    with st.expander("Ver prévia das cestas de compras"):
        st.dataframe(
            compras.reset_index(name="Produtos").head(10), width="stretch"
        )

    if "Country" in df.columns:
        with st.expander("🌎 De onde vêm as compras?"):
            por_pais = (
                df.groupby("Country")["InvoiceNo"].nunique().sort_values(ascending=False)
            )
            pct_uk = por_pais.get("United Kingdom", 0) / por_pais.sum() * 100
            st.write(f"**{pct_uk:.1f}%** das compras são do Reino Unido.".replace(".", ",", 1))
            st.bar_chart(por_pais.head(5))

    with st.expander("📦 Como os produtos se distribuem?"):
        st.write(
            "Alguns produtos aparecem em muitas compras, enquanto a maioria é rara. "
            "Os algoritmos só encontram combinações que atingem o suporte mínimo."
        )
        st.bar_chart(compras_por_produto.head(10))

    st.divider()
    st.header("🛍️ Baseline: produtos mais populares")
    st.write(
        "Estes são os produtos que aparecem no maior número de compras. "
        "Isso não significa que sejam comprados juntos."
    )

    ranking = baseline_mais_vendidos(df, n=10).rename(
        columns={
            "Pct_compras": "Compras (%)",
            "Compras": "Nº de compras",
            "Unidades": "Unidades vendidas",
            "Posicao": "Posição",
        }
    )
    st.dataframe(ranking, width="stretch", hide_index=True)
