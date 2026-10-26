
from mlxtend.frequent_patterns import apriori, association_rules


def executar_apriori(cesta_binaria, suporte_minimo, confianca_minima):
    conjuntos = apriori(
        cesta_binaria,
        min_support=suporte_minimo,
        use_colnames=True,
        low_memory=True
    )

    if conjuntos.empty:
        return conjuntos, None

    regras = association_rules(
        conjuntos,
        metric="confidence",
        min_threshold=confianca_minima
    )

    return conjuntos, regras
