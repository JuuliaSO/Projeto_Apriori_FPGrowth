
from mlxtend.frequent_patterns import fpgrowth, association_rules


def executar_fp_growth(cesta_binaria, suporte_minimo, confianca_minima):
    conjuntos = fpgrowth(
        cesta_binaria,
        min_support=suporte_minimo,
        use_colnames=True
    )

    if conjuntos.empty:
        return conjuntos, None

    regras = association_rules(
        conjuntos,
        metric="confidence",
        min_threshold=confianca_minima
    )

    return conjuntos, regras
