from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CAMINHOS
# ============================================================

PASTA_SAIDA = Path("resultados/automotivo")

CAMINHO_SAIDA = (
    PASTA_SAIDA
    / "comparacao_formulas_health.csv"
)

PASTA_SAIDA.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# FÓRMULA
# ============================================================

def calcular_health(
    score_medio: float,
    persistencia: float,
    peso_score: float,
) -> float:
    """
    Automotive Health demonstrativo.

    Risco =
        peso_score * intensidade recente
        +
        (1 - peso_score) * persistência

    Health = 100 - risco

    score_medio:
        média recente do Anomaly Score [0, 100]

    persistencia:
        percentual recente de janelas anômalas [0, 100]

    IMPORTANTE:
    Health representa proximidade ao padrão aprendido.
    Não representa percentual de vida mecânica.
    """

    score_medio = float(
        np.clip(
            score_medio,
            0.0,
            100.0,
        )
    )

    persistencia = float(
        np.clip(
            persistencia,
            0.0,
            100.0,
        )
    )

    peso_persistencia = (
        1.0 - peso_score
    )

    risco = (
        peso_score * score_medio
        + peso_persistencia * persistencia
    )

    health = 100.0 - risco

    return float(
        np.clip(
            health,
            0.0,
            100.0,
        )
    )


# ============================================================
# CONFIGURAÇÕES TESTADAS
# ============================================================

FORMULAS = {
    "80_20": 0.80,
    "70_30": 0.70,
    "60_40": 0.60,
    "50_50": 0.50,
}

SCORES_MEDIOS = [
    0,
    10,
    25,
    50,
    75,
    100,
]

PERSISTENCIAS = [
    0,
    10,
    25,
    50,
    75,
    100,
]


# ============================================================
# EXECUÇÃO
# ============================================================

def main():

    print()
    print("=" * 78)
    print("TESTE DE SENSIBILIDADE — AUTOMOTIVE HEALTH")
    print("=" * 78)

    linhas = []

    for score_medio in SCORES_MEDIOS:

        for persistencia in PERSISTENCIAS:

            linha = {
                "score_medio": score_medio,
                "persistencia": persistencia,
            }

            for nome, peso_score in FORMULAS.items():

                linha[
                    f"health_{nome}"
                ] = calcular_health(
                    score_medio=score_medio,
                    persistencia=persistencia,
                    peso_score=peso_score,
                )

            linhas.append(
                linha
            )

    df = pd.DataFrame(
        linhas
    )

    df.to_csv(
        CAMINHO_SAIDA,
        index=False,
    )

    print()
    print("TABELA COMPLETA")
    print("-" * 78)

    print(
        df.to_string(
            index=False,
            formatters={
                "health_80_20": lambda x: f"{x:.1f}",
                "health_70_30": lambda x: f"{x:.1f}",
                "health_60_40": lambda x: f"{x:.1f}",
                "health_50_50": lambda x: f"{x:.1f}",
            },
        )
    )

    # ========================================================
    # CASOS DIDÁTICOS
    # ========================================================

    casos = [
        {
            "caso": "Sem desvio",
            "score": 0,
            "persistencia": 0,
        },
        {
            "caso": "Desvio isolado",
            "score": 50,
            "persistencia": 10,
        },
        {
            "caso": "Moderado persistente",
            "score": 50,
            "persistencia": 50,
        },
        {
            "caso": "Forte pouco persistente",
            "score": 80,
            "persistencia": 20,
        },
        {
            "caso": "Moderado muito persistente",
            "score": 50,
            "persistencia": 90,
        },
        {
            "caso": "Forte persistente",
            "score": 80,
            "persistencia": 90,
        },
        {
            "caso": "Extremo persistente",
            "score": 100,
            "persistencia": 100,
        },
    ]

    linhas_casos = []

    for caso in casos:

        linha = {
            "caso": caso["caso"],
            "score": caso["score"],
            "persistencia": caso["persistencia"],
        }

        for nome, peso_score in FORMULAS.items():

            linha[
                f"health_{nome}"
            ] = calcular_health(
                score_medio=caso["score"],
                persistencia=caso["persistencia"],
                peso_score=peso_score,
            )

        linhas_casos.append(
            linha
        )

    df_casos = pd.DataFrame(
        linhas_casos
    )

    print()
    print("=" * 78)
    print("CASOS DIDÁTICOS")
    print("=" * 78)

    print()
    print(
        df_casos.to_string(
            index=False,
            formatters={
                "health_80_20": lambda x: f"{x:.1f}",
                "health_70_30": lambda x: f"{x:.1f}",
                "health_60_40": lambda x: f"{x:.1f}",
                "health_50_50": lambda x: f"{x:.1f}",
            },
        )
    )

    print()
    print("=" * 78)
    print("INTERPRETAÇÃO")
    print("=" * 78)

    print(
        "Quanto maior o peso do score médio, "
        "mais o Health responde à intensidade."
    )

    print(
        "Quanto maior o peso da persistência, "
        "mais o Health penaliza desvios repetidos."
    )

    print(
        "Nenhuma fórmula é selecionada automaticamente."
    )

    print()
    print(
        "Arquivo salvo em:"
    )

    print(
        CAMINHO_SAIDA
    )

    print()
    print("=" * 78)
    print("TESTE CONCLUÍDO")
    print("=" * 78)


if __name__ == "__main__":
    main()