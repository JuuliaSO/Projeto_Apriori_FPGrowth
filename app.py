
import streamlit as st
import pandas as pd
import time
from pathlib import Path
from mlxtend.preprocessing import TransactionEncoder

from eda_baseline import baseline_mais_vendidos
from algoritmos.apriori import executar_apriori
from algoritmos.fp_growth import executar_fp_growth


# Configuração da página

st.set_page_config(
    page_title="Apriori e FP-Growth",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 Quais produtos aparecem juntos?")

st.write(
    "Descubra quais produtos costumam ser comprados "
    "na mesma compra usando Apriori e FP-Growth."
)


# Carregar o dataset já preparado

caminho_csv = Path("dados") / "Base_Analise.csv"

if not caminho_csv.exists():
    st.error(
        "CSV não encontrado. Confira se Base_Analise.csv "
        "está dentro da pasta dados."
    )
    st.stop()

try:
    df = pd.read_csv(
        caminho_csv,
        encoding="utf-8-sig",
        sep=";",
        decimal=","
    )
except Exception as erro:
    st.error(f"Não foi possível ler o CSV: {erro}")
    st.stop()


# Conferir as colunas necessárias

colunas = [
    "InvoiceNo",
    "Description",
    "Quantity",
    "UnitPrice"
]

if not all(coluna in df.columns for coluna in colunas):
    st.error(
        "Confira se o CSV contém as colunas InvoiceNo, "
        "Description, Quantity e UnitPrice."
    )
    st.stop()


# Criar uma cesta por compra
# Não aplicamos novos filtros ao dataset.

compras = (
    df.groupby("InvoiceNo")["Description"]
    .apply(lambda produtos: sorted(set(produtos)))
)


# Visão geral das compras

st.header("📊 Conhecendo as compras")

st.write(
    "Antes de procurar combinações de produtos, "
    "vamos conhecer os dados que estamos analisando."
)

col1, col2 = st.columns(2)

col1.metric(
    "Registros na base",
    f"{len(df):,}".replace(",", ".")
)

col2.metric(
    "Compras analisadas",
    f"{df['InvoiceNo'].nunique():,}".replace(",", ".")
)

col3, col4 = st.columns(2)

col3.metric(
    "Produtos diferentes",
    f"{df['Description'].nunique():,}".replace(",", ".")
)

col4.metric(
    "Itens por compra (mediana)",
    f"{compras.apply(len).median():.0f}"
)

st.info(
    "Cada compra pode conter vários produtos. "
    "Nosso objetivo é descobrir quais produtos aparecem juntos, "
    "e não apenas quais são os mais populares."
)


# Prévia dos dados

with st.expander("Ver prévia dos dados"):
    st.dataframe(df.head(10), use_container_width=True)

with st.expander("Ver prévia das cestas de compras"):
    cestas = compras.reset_index(name="Produtos")
    st.dataframe(cestas.head(10), use_container_width=True)


# Distribuição das compras por país

if "Country" in df.columns:
    with st.expander("🌎 De onde vêm as compras?"):

        compras_por_pais = (
            df.groupby("Country")["InvoiceNo"]
            .nunique()
            .sort_values(ascending=False)
        )

        total_compras_pais = compras_por_pais.sum()

        if total_compras_pais > 0:
            percentual_uk = (
                compras_por_pais.get("United Kingdom", 0)
                / total_compras_pais
                * 100
            )

            st.write(
                f"**{percentual_uk:.1f}%** das compras "
                "são do Reino Unido."
            )

        st.bar_chart(compras_por_pais.head(5))


# Frequência dos produtos

with st.expander("📦 Como os produtos se distribuem?"):

    frequencia_produtos = (
        df.groupby("Description")["InvoiceNo"]
        .nunique()
        .sort_values(ascending=False)
    )

    st.write(
        "Alguns produtos aparecem em muitas compras, "
        "enquanto outros são menos frequentes. "
        "Os algoritmos procuram combinações que atendam "
        "ao suporte mínimo escolhido."
    )

    st.bar_chart(frequencia_produtos.head(10))


# Baseline: produtos mais populares

st.divider()

st.header("🛍️ Produtos mais populares")

st.write(
    "Estes são os produtos que aparecem no maior número de compras. "
    "Isso não significa necessariamente que sejam comprados juntos."
)

ranking = baseline_mais_vendidos(df, n=10)

ranking = ranking.rename(
    columns={
        "Pct_compras": "Compras (%)",
        "Compras": "Nº de compras",
        "Unidades": "Unidades vendidas",
        "Posicao": "Posição"
    }
)

st.dataframe(ranking, use_container_width=True)


# Configurações da análise

st.divider()

st.header("⚙️ Configurações da análise")

algoritmo = st.selectbox(
    "Escolha o algoritmo",
    ["Apriori", "FP-Growth"]
)

suporte = st.slider(
    "Suporte mínimo (%)",
    min_value=0.1,
    max_value=10.0,
    value=1.0,
    step=0.1
)

confianca = st.slider(
    "Confiança mínima (%)",
    min_value=10,
    max_value=100,
    value=50,
    step=5
)

st.caption(
    "O suporte indica em quantas compras os produtos aparecem juntos. "
    "A confiança indica a frequência com que o produto consequente "
    "aparece quando o antecedente está presente."
)


# Executar a análise

if st.button("Executar análise", type="primary"):

    if len(compras) < 2:
        st.error("São necessárias pelo menos duas compras.")
        st.stop()

    with st.spinner("Analisando as compras..."):

        try:
            # Preparar a matriz binária

            encoder = TransactionEncoder()

            matriz = encoder.fit(compras).transform(
                compras,
                sparse=True
            )

            cesta_binaria = pd.DataFrame.sparse.from_spmatrix(
                matriz,
                columns=encoder.columns_
            )

            suporte_decimal = suporte / 100
            confianca_decimal = confianca / 100

            # Executar o algoritmo e medir o tempo

            inicio = time.perf_counter()

            if algoritmo == "Apriori":
                frequentes, regras = executar_apriori(
                    cesta_binaria,
                    suporte_decimal,
                    confianca_decimal
                )
            else:
                frequentes, regras = executar_fp_growth(
                    cesta_binaria,
                    suporte_decimal,
                    confianca_decimal
                )

            tempo_execucao = time.perf_counter() - inicio

            # Desempenho

            st.divider()
            st.header("⏱️ Desempenho do algoritmo")

            col1, col2 = st.columns(2)

            col1.metric("Algoritmo executado", algoritmo)

            col2.metric(
                "Tempo de execução",
                f"{tempo_execucao:.4f} segundos"
            )

            # Conjuntos frequentes

            if frequentes is None or frequentes.empty:
                st.warning(
                    "Nenhum conjunto frequente encontrado. "
                    "Tente diminuir o suporte mínimo."
                )
                st.stop()

            frequentes = frequentes.copy()

            frequentes["Quantidade de produtos"] = (
                frequentes["itemsets"].apply(len)
            )

            frequentes["Produtos"] = frequentes["itemsets"].apply(
                lambda itens: ", ".join(sorted(itens))
            )

            st.header("📦 Conjuntos frequentes")

            st.write(
                f"Total encontrado: {len(frequentes)} conjuntos."
            )

            tabela_frequentes = frequentes[
                [
                    "Produtos",
                    "support",
                    "Quantidade de produtos"
                ]
            ].rename(
                columns={
                    "support": "Suporte (%)",
                    "Quantidade de produtos": "Nº de produtos"
                }
            )

            tabela_frequentes["Suporte (%)"] = (
                tabela_frequentes["Suporte (%)"] * 100
            ).round(2)

            tabela_frequentes = tabela_frequentes.sort_values(
                "Suporte (%)",
                ascending=False
            )

            st.dataframe(
                tabela_frequentes,
                use_container_width=True
            )

            # Regras de associação

            st.header("🔗 Regras de associação")

            if regras is None or regras.empty:
                st.warning(
                    "Nenhuma regra atingiu a confiança mínima. "
                    "Tente diminuir a confiança ou o suporte."
                )

            else:
                regras = regras.copy()

                regras["Se comprar"] = regras["antecedents"].apply(
                    lambda itens: ", ".join(sorted(itens))
                )

                regras["Também pode comprar"] = regras[
                    "consequents"
                ].apply(
                    lambda itens: ", ".join(sorted(itens))
                )

                regras["Suporte (%)"] = (
                    regras["support"] * 100
                ).round(2)

                regras["Confiança (%)"] = (
                    regras["confidence"] * 100
                ).round(2)

                regras["Lift"] = regras["lift"].round(2)

                tabela_regras = regras[
                    [
                        "Se comprar",
                        "Também pode comprar",
                        "Suporte (%)",
                        "Confiança (%)",
                        "Lift"
                    ]
                ].sort_values(
                    "Confiança (%)",
                    ascending=False
                )

                st.dataframe(
                    tabela_regras,
                    use_container_width=True
                )

                # Explicação da primeira regra

                st.subheader("💡 Como interpretar")

                primeira = tabela_regras.iloc[0]

                st.write(
                    f"Quando alguém compra **{primeira['Se comprar']}**, "
                    f"a regra sugere **{primeira['Também pode comprar']}**."
                )

                st.write(
                    f"**Suporte:** {primeira['Suporte (%)']}% "
                    "das compras contêm os produtos da regra."
                )

                st.write(
                    f"**Confiança:** em {primeira['Confiança (%)']}% "
                    "das compras com o antecedente, o consequente "
                    "também aparece."
                )

                st.write(
                    f"**Lift:** {primeira['Lift']}. "
                    "Valores acima de 1 indicam uma associação positiva "
                    "em relação à frequência esperada do consequente."
                )

        except Exception as erro:
            st.error(f"Erro durante a análise: {erro}")

            st.info(
                "Confira a mensagem de erro acima. "
                "Se necessário, envie-a para verificarmos o problema."
            )
