from pathlib import Path

import json
import numpy as np
import pandas as pd


# =========================================================
# CAMINHOS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

ANOMALIAS_PATH = (
    BASE_DIR
    / "resultados"
    / "fd001_anomalia"
    / "anomalias_validacao.csv"
)

RESULTADOS_DIR = (
    BASE_DIR
    / "resultados"
    / "fd001_anomalia"
)

ANOMALY_SCORE_PATH = (
    RESULTADOS_DIR
    / "anomaly_score_validacao.csv"
)

RESUMO_PATH = (
    RESULTADOS_DIR
    / "resumo_anomaly_score.csv"
)

CONFIG_PATH = (
    RESULTADOS_DIR
    / "config_anomaly_score.json"
)


# =========================================================
# CONFIGURAÇÕES
# =========================================================

PERCENTIL_THRESHOLD = 95
PERCENTIL_MAXIMO = 99

FAIXAS_RUL = [
    "RUL > 125",
    "76-125",
    "31-75",
    "0-30"
]


# =========================================================
# CARREGAMENTO
# =========================================================

def carregar_dados():

    if not ANOMALIAS_PATH.exists():

        raise FileNotFoundError(
            "Arquivo não encontrado:\n"
            f"{ANOMALIAS_PATH}"
        )

    df = pd.read_csv(
        ANOMALIAS_PATH
    )

    colunas_necessarias = [
        "unit_number",
        "cycle",
        "RUL_linear",
        "erro_reconstrucao"
    ]

    faltantes = [
        coluna
        for coluna in colunas_necessarias
        if coluna not in df.columns
    ]

    if faltantes:

        raise ValueError(
            f"Colunas ausentes: {faltantes}"
        )

    return df


# =========================================================
# FAIXAS DE RUL
# =========================================================

def definir_faixa_rul(rul):

    if rul > 125:
        return "RUL > 125"

    if rul > 75:
        return "76-125"

    if rul > 30:
        return "31-75"

    return "0-30"


# =========================================================
# CALIBRAÇÃO
# =========================================================

def calcular_referencias(df):

    erros_saudaveis = (
        df[
            df["RUL_linear"] > 125
        ]["erro_reconstrucao"]
        .dropna()
        .values
    )

    erros_totais = (
        df[
            "erro_reconstrucao"
        ]
        .dropna()
        .values
    )

    if len(erros_saudaveis) == 0:

        raise ValueError(
            "Nenhum erro saudável encontrado."
        )

    referencia_zero = float(
        np.median(
            erros_saudaveis
        )
    )

    referencia_threshold = float(
        np.percentile(
            erros_saudaveis,
            PERCENTIL_THRESHOLD
        )
    )

    referencia_maxima = float(
        np.percentile(
            erros_totais,
            PERCENTIL_MAXIMO
        )
    )

    if referencia_threshold <= referencia_zero:

        raise ValueError(
            "Threshold inválido: "
            "P95 saudável <= mediana saudável."
        )

    if referencia_maxima <= referencia_threshold:

        raise ValueError(
            "Referência máxima inválida: "
            "P99 geral <= threshold."
        )

    return (
        referencia_zero,
        referencia_threshold,
        referencia_maxima
    )


# =========================================================
# ANOMALY SCORE
# =========================================================

def calcular_anomaly_score(
    erro,
    referencia_zero,
    referencia_threshold,
    referencia_maxima
):

    # -----------------------------------------------------
    # REGIÃO 1
    # Mediana saudável -> 0
    # Threshold P95 saudável -> 50
    # -----------------------------------------------------

    if erro <= referencia_threshold:

        score = (
            (
                erro
                - referencia_zero
            )
            /
            (
                referencia_threshold
                - referencia_zero
            )
        ) * 50

    # -----------------------------------------------------
    # REGIÃO 2
    # Threshold -> 50
    # P99 geral -> 100
    # -----------------------------------------------------

    else:

        score = (
            50
            +
            (
                (
                    erro
                    - referencia_threshold
                )
                /
                (
                    referencia_maxima
                    - referencia_threshold
                )
            )
            * 50
        )

    score = np.clip(
        score,
        0,
        100
    )

    return float(score)


# =========================================================
# GERAR SCORES
# =========================================================

def gerar_scores(
    df,
    referencia_zero,
    referencia_threshold,
    referencia_maxima
):

    df = df.copy()

    df["faixa_RUL"] = (
        df["RUL_linear"]
        .apply(
            definir_faixa_rul
        )
    )

    df["anomaly_score"] = (
        df["erro_reconstrucao"]
        .apply(
            lambda erro:
            calcular_anomaly_score(
                erro,
                referencia_zero,
                referencia_threshold,
                referencia_maxima
            )
        )
    )

    df["anomalia_threshold"] = (
        df["erro_reconstrucao"]
        > referencia_threshold
    ).astype(int)

    return df


# =========================================================
# RESUMO
# =========================================================

def gerar_resumo(df):

    resumo = (
        df
        .groupby(
            "faixa_RUL"
        )
        .agg(
            quantidade=(
                "anomaly_score",
                "count"
            ),

            anomaly_score_medio=(
                "anomaly_score",
                "mean"
            ),

            anomaly_score_mediano=(
                "anomaly_score",
                "median"
            ),

            anomaly_score_p75=(
                "anomaly_score",
                lambda x:
                np.percentile(
                    x,
                    75
                )
            ),

            anomaly_score_p90=(
                "anomaly_score",
                lambda x:
                np.percentile(
                    x,
                    90
                )
            ),

            taxa_acima_threshold=(
                "anomalia_threshold",
                "mean"
            )
        )
        .reindex(
            FAIXAS_RUL
        )
        .reset_index()
    )

    resumo[
        "taxa_acima_threshold_percentual"
    ] = (
        resumo[
            "taxa_acima_threshold"
        ]
        * 100
    )

    return resumo


# =========================================================
# EXECUÇÃO
# =========================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("ANOMALY SCORE - NASA FD001")
    print("=" * 70)

    # -----------------------------------------------------
    # 1. CARREGAR
    # -----------------------------------------------------

    df = carregar_dados()

    print(
        f"\nLinhas carregadas: "
        f"{len(df)}"
    )

    print(
        f"Motores: "
        f"{df['unit_number'].nunique()}"
    )

    # -----------------------------------------------------
    # 2. CALIBRAÇÃO
    # -----------------------------------------------------

    (
        referencia_zero,
        referencia_threshold,
        referencia_maxima
    ) = calcular_referencias(
        df
    )

    print("\n" + "=" * 70)
    print("CALIBRAÇÃO DO ANOMALY SCORE")
    print("=" * 70)

    print(
        f"\nReferência 0 "
        f"(mediana saudável): "
        f"{referencia_zero:.8f}"
    )

    print(
        f"Referência 50 "
        f"(P{PERCENTIL_THRESHOLD} saudável): "
        f"{referencia_threshold:.8f}"
    )

    print(
        f"Referência 100 "
        f"(P{PERCENTIL_MAXIMO} geral): "
        f"{referencia_maxima:.8f}"
    )

    # -----------------------------------------------------
    # 3. GERAR SCORE
    # -----------------------------------------------------

    df_score = gerar_scores(
        df,
        referencia_zero,
        referencia_threshold,
        referencia_maxima
    )

    # -----------------------------------------------------
    # 4. RESUMO
    # -----------------------------------------------------

    resumo = gerar_resumo(
        df_score
    )

    print("\n" + "=" * 70)
    print("ANOMALY SCORE POR FAIXA DE RUL")
    print("=" * 70)

    colunas_exibicao = [
        "faixa_RUL",
        "quantidade",
        "anomaly_score_medio",
        "anomaly_score_mediano",
        "anomaly_score_p75",
        "anomaly_score_p90",
        "taxa_acima_threshold_percentual"
    ]

    print(
        "\n"
        + resumo[
            colunas_exibicao
        ]
        .round(2)
        .to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # 5. SALVAR
    # -----------------------------------------------------

    RESULTADOS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df_score.to_csv(
        ANOMALY_SCORE_PATH,
        index=False
    )

    resumo.to_csv(
        RESUMO_PATH,
        index=False
    )

    config = {

        "dataset": "FD001",

        "fonte": (
            "erro de reconstrucao "
            "do LSTM Autoencoder"
        ),

        "score_min": 0,
        "score_threshold": 50,
        "score_max": 100,

        "calibracao": {

            "referencia_zero": (
                referencia_zero
            ),

            "referencia_zero_metodo": (
                "mediana dos erros "
                "com RUL_linear > 125"
            ),

            "referencia_threshold": (
                referencia_threshold
            ),

            "referencia_threshold_metodo": (
                f"percentil "
                f"{PERCENTIL_THRESHOLD} "
                f"dos erros com "
                f"RUL_linear > 125"
            ),

            "referencia_maxima": (
                referencia_maxima
            ),

            "referencia_maxima_metodo": (
                f"percentil "
                f"{PERCENTIL_MAXIMO} "
                f"dos erros da validacao completa"
            )
        },

        "interpretacao": {
            "0": (
                "comportamento proximo "
                "da mediana saudavel"
            ),

            "50": (
                "limiar estatistico "
                "de anomalia"
            ),

            "100": (
                "desvio severo do "
                "padrao saudavel"
            )
        },

        "observacao": (
            "RUL e utilizado apenas durante "
            "a calibracao experimental. "
            "Em inferencia, o score depende "
            "somente do erro de reconstrucao "
            "produzido pelo Autoencoder."
        )
    }

    with open(
        CONFIG_PATH,
        "w",
        encoding="utf-8"
    ) as arquivo:

        json.dump(
            config,
            arquivo,
            indent=4,
            ensure_ascii=False
        )

    print("\n" + "=" * 70)
    print("ARQUIVOS SALVOS")
    print("=" * 70)

    print(
        f"\nAnomaly Score:\n"
        f"{ANOMALY_SCORE_PATH}"
    )

    print(
        f"\nResumo:\n"
        f"{RESUMO_PATH}"
    )

    print(
        f"\nConfig:\n"
        f"{CONFIG_PATH}"
    )