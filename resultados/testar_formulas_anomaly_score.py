from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CAMINHOS
# ============================================================

CAMINHO_CALIBRACAO = Path(
    "resultados/automotivo/"
    "calibracao_anomaly_score_normal.csv"
)

CAMINHO_RESUMO = Path(
    "resultados/automotivo/"
    "resumo_calibracao_anomaly_score.json"
)

CAMINHO_SAIDA = Path(
    "resultados/automotivo/"
    "comparacao_formulas_anomaly_score.csv"
)


# ============================================================
# CARREGAMENTO
# ============================================================

def carregar_dados():

    if not CAMINHO_CALIBRACAO.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: "
            f"{CAMINHO_CALIBRACAO}"
        )

    if not CAMINHO_RESUMO.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: "
            f"{CAMINHO_RESUMO}"
        )

    df = pd.read_csv(
        CAMINHO_CALIBRACAO
    )

    with open(
        CAMINHO_RESUMO,
        "r",
        encoding="utf-8",
    ) as arquivo:
        resumo = json.load(arquivo)

    return df, resumo


# ============================================================
# PARTE COMUM: MEDIANA -> P95
# ============================================================

def score_ate_threshold(
    erro,
    mediana,
    threshold,
):
    """
    Mediana -> 0
    P95      -> 50
    """

    if erro <= mediana:
        return 0.0

    if erro >= threshold:
        return 50.0

    score = (
        (erro - mediana)
        / (threshold - mediana)
        * 50.0
    )

    return float(
        np.clip(
            score,
            0.0,
            50.0,
        )
    )


# ============================================================
# FÓRMULA A
# ============================================================

def formula_a(
    erro,
    mediana,
    threshold,
):
    """
    Acima do P95:

    score = 50 + 50 * (
        1 - exp(
            -ln(2) * ln(erro / threshold)
        )
    )

    Crescimento relativamente suave.
    """

    if erro <= threshold:
        return score_ate_threshold(
            erro,
            mediana,
            threshold,
        )

    razao = erro / threshold

    score = (
        50.0
        + 50.0
        * (
            1.0
            - np.exp(
                -np.log(2.0)
                * np.log(razao)
            )
        )
    )

    return float(
        np.clip(
            score,
            0.0,
            100.0,
        )
    )


# ============================================================
# FÓRMULA B
# ============================================================

def formula_b(
    erro,
    mediana,
    threshold,
):
    """
    Mesmo formato da A, mas crescimento
    intermediário acima do threshold.
    """

    if erro <= threshold:
        return score_ate_threshold(
            erro,
            mediana,
            threshold,
        )

    razao = erro / threshold

    score = (
        50.0
        + 50.0
        * (
            1.0
            - np.exp(
                -1.0
                * np.log(razao)
            )
        )
    )

    return float(
        np.clip(
            score,
            0.0,
            100.0,
        )
    )


# ============================================================
# FÓRMULA C
# ============================================================

def formula_c(
    erro,
    mediana,
    threshold,
):
    """
    Crescimento mais agressivo acima
    do threshold.
    """

    if erro <= threshold:
        return score_ate_threshold(
            erro,
            mediana,
            threshold,
        )

    razao = erro / threshold

    score = (
        50.0
        + 50.0
        * (
            1.0
            - np.exp(
                -1.5
                * np.log(razao)
            )
        )
    )

    return float(
        np.clip(
            score,
            0.0,
            100.0,
        )
    )


# ============================================================
# APLICAR FÓRMULA
# ============================================================

def aplicar_formula(
    valores,
    funcao,
    mediana,
    threshold,
):

    return np.asarray(
        [
            funcao(
                float(valor),
                mediana,
                threshold,
            )
            for valor in valores
        ],
        dtype=float,
    )


# ============================================================
# EXECUÇÃO
# ============================================================

def main():

    print()
    print("=" * 74)
    print("TESTE DE FÓRMULAS — ANOMALY SCORE AUTOMOTIVO")
    print("=" * 74)

    df, resumo = carregar_dados()

    erros = (
        df["reconstruction_error"]
        .to_numpy(
            dtype=float
        )
    )

    percentis = resumo[
        "percentis"
    ]

    mediana = float(
        percentis["50"]
    )

    threshold = float(
        resumo[
            "threshold_original"
        ]
    )

    print()
    print(
        f"Mediana NORMAL: "
        f"{mediana:.8f}"
    )
    print(
        f"Threshold P95:  "
        f"{threshold:.8f}"
    )

    # --------------------------------------------------------
    # Aplicação no NORMAL de validação
    # --------------------------------------------------------

    formulas = {
        "A_SUAVE": formula_a,
        "B_INTERMEDIARIA": formula_b,
        "C_AGRESSIVA": formula_c,
    }

    resultados = []

    print()
    print("=" * 74)
    print("COMPORTAMENTO NO NORMAL DE VALIDAÇÃO")
    print("=" * 74)

    for nome, funcao in formulas.items():

        scores = aplicar_formula(
            erros,
            funcao,
            mediana,
            threshold,
        )

        linha = {
            "formula": nome,
            "score_medio_normal": float(
                np.mean(scores)
            ),
            "score_mediano_normal": float(
                np.median(scores)
            ),
            "score_p90_normal": float(
                np.percentile(
                    scores,
                    90,
                )
            ),
            "score_p95_normal": float(
                np.percentile(
                    scores,
                    95,
                )
            ),
            "score_p99_normal": float(
                np.percentile(
                    scores,
                    99,
                )
            ),
            "score_max_normal": float(
                np.max(scores)
            ),
        }

        resultados.append(
            linha
        )

        print()
        print(nome)
        print(
            f"  Média:   "
            f"{linha['score_medio_normal']:.2f}"
        )
        print(
            f"  Mediana: "
            f"{linha['score_mediano_normal']:.2f}"
        )
        print(
            f"  P90:     "
            f"{linha['score_p90_normal']:.2f}"
        )
        print(
            f"  P95:     "
            f"{linha['score_p95_normal']:.2f}"
        )
        print(
            f"  P99:     "
            f"{linha['score_p99_normal']:.2f}"
        )
        print(
            f"  Máximo:  "
            f"{linha['score_max_normal']:.2f}"
        )

    # --------------------------------------------------------
    # Pontos de referência
    # --------------------------------------------------------

    pontos = {
        "MEDIANA_NORMAL": mediana,
        "P90_NORMAL": float(
            percentis["90"]
        ),
        "P95_THRESHOLD": threshold,
        "P99_NORMAL": float(
            percentis["99"]
        ),
        "P99_9_NORMAL": float(
            percentis["99.9"]
        ),
        "MAX_NORMAL": float(
            percentis["100"]
        ),

        # Valores puramente matemáticos para visualizar
        # o comportamento da curva acima da região NORMAL.
        # NÃO são usados para calibrar a fórmula.
        "2X_THRESHOLD": (
            2.0 * threshold
        ),
        "3X_THRESHOLD": (
            3.0 * threshold
        ),
        "5X_THRESHOLD": (
            5.0 * threshold
        ),
        "10X_THRESHOLD": (
            10.0 * threshold
        ),
        "50X_THRESHOLD": (
            50.0 * threshold
        ),
        "100X_THRESHOLD": (
            100.0 * threshold
        ),
    }

    print()
    print("=" * 74)
    print("COMPARAÇÃO EM PONTOS DE REFERÊNCIA")
    print("=" * 74)

    tabela_pontos = []

    for nome_ponto, erro in pontos.items():

        linha = {
            "ponto": nome_ponto,
            "erro": erro,
        }

        for nome_formula, funcao in formulas.items():

            linha[
                nome_formula
            ] = funcao(
                erro,
                mediana,
                threshold,
            )

        tabela_pontos.append(
            linha
        )

    df_pontos = pd.DataFrame(
        tabela_pontos
    )

    print()
    print(
        df_pontos.to_string(
            index=False,
            formatters={
                "erro": (
                    lambda x:
                    f"{x:.6f}"
                ),
                "A_SUAVE": (
                    lambda x:
                    f"{x:.2f}"
                ),
                "B_INTERMEDIARIA": (
                    lambda x:
                    f"{x:.2f}"
                ),
                "C_AGRESSIVA": (
                    lambda x:
                    f"{x:.2f}"
                ),
            },
        )
    )

    # --------------------------------------------------------
    # Salvar comparação
    # --------------------------------------------------------

    df_resultados = pd.DataFrame(
        resultados
    )

    # Salva as duas tabelas no mesmo CSV,
    # separadas por uma linha identificadora.
    with open(
        CAMINHO_SAIDA,
        "w",
        encoding="utf-8",
        newline="",
    ) as arquivo:

        arquivo.write(
            "RESUMO_NORMAL\n"
        )

        df_resultados.to_csv(
            arquivo,
            index=False,
        )

        arquivo.write(
            "\nPONTOS_REFERENCIA\n"
        )

        df_pontos.to_csv(
            arquivo,
            index=False,
        )

    print()
    print("=" * 74)
    print("OBSERVAÇÃO METODOLÓGICA")
    print("=" * 74)

    print(
        "Os cenários anormais reservados NÃO "
        "foram utilizados para escolher "
        "os parâmetros das curvas."
    )

    print(
        "Os múltiplos do threshold servem apenas "
        "para visualizar matematicamente a "
        "progressão do score."
    )

    print()
    print(
        "Arquivo salvo em:"
    )
    print(
        CAMINHO_SAIDA
    )

    print()
    print("=" * 74)
    print("TESTE CONCLUÍDO")
    print("=" * 74)


if __name__ == "__main__":
    main()