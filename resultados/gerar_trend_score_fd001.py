from pathlib import Path

import json
import numpy as np
import pandas as pd


# =========================================================
# CAMINHOS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TENDENCIA_PATH = (
    BASE_DIR
    / "resultados"
    / "fd001_tendencia"
    / "tendencia_horizonte_10.csv"
)

RESULTADOS_DIR = (
    BASE_DIR
    / "resultados"
    / "fd001_tendencia"
)

TREND_SCORE_PATH = (
    RESULTADOS_DIR
    / "trend_score_horizonte_10.csv"
)

RESUMO_PATH = (
    RESULTADOS_DIR
    / "resumo_trend_score.csv"
)

CONFIG_PATH = (
    RESULTADOS_DIR
    / "config_trend_score.json"
)


# =========================================================
# CONFIGURAÇÃO
# =========================================================

HORIZONTE = 10

PERCENTIL_SUPERIOR = 99

FAIXAS_RUL = [
    "RUL > 125",
    "76-125",
    "31-75",
    "0-30"
]


# =========================================================
# CARREGAR
# =========================================================

def carregar_dados():

    if not TENDENCIA_PATH.exists():

        raise FileNotFoundError(
            "Arquivo não encontrado:\n"
            f"{TENDENCIA_PATH}"
        )

    df = pd.read_csv(
        TENDENCIA_PATH
    )

    colunas = [
        "unit_number",
        "cycle",
        "RUL_linear",
        "erro_reconstrucao",
        "slope"
    ]

    faltantes = [
        coluna
        for coluna in colunas
        if coluna not in df.columns
    ]

    if faltantes:

        raise ValueError(
            f"Colunas ausentes: {faltantes}"
        )

    return df


# =========================================================
# FAIXA DE RUL
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

def calcular_calibracao(df):

    slopes_saudaveis = (
        df[
            df["RUL_linear"] > 125
        ]["slope"]
        .dropna()
        .values
    )

    if len(slopes_saudaveis) == 0:

        raise ValueError(
            "Nenhum slope saudável encontrado."
        )

    referencia_zero = float(
        np.median(
            slopes_saudaveis
        )
    )

    referencia_maxima = float(
        np.percentile(
            slopes_saudaveis,
            PERCENTIL_SUPERIOR
        )
    )

    if referencia_maxima <= referencia_zero:

        raise ValueError(
            "Calibração inválida: "
            "percentil superior <= mediana."
        )

    return (
        referencia_zero,
        referencia_maxima
    )


# =========================================================
# TREND SCORE
# =========================================================

def calcular_trend_score(
    slope,
    referencia_zero,
    referencia_maxima
):

    score = (
        (
            slope
            - referencia_zero
        )
        /
        (
            referencia_maxima
            - referencia_zero
        )
    ) * 100

    score = np.clip(
        score,
        0,
        100
    )

    return float(score)


# =========================================================
# PROCESSAMENTO
# =========================================================

def gerar_scores(
    df,
    referencia_zero,
    referencia_maxima
):

    df = df.copy()

    df["faixa_RUL"] = (
        df["RUL_linear"]
        .apply(
            definir_faixa_rul
        )
    )

    df["trend_score"] = (
        df["slope"]
        .apply(
            lambda slope:
            calcular_trend_score(
                slope,
                referencia_zero,
                referencia_maxima
            )
        )
    )

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
                "trend_score",
                "count"
            ),

            trend_score_medio=(
                "trend_score",
                "mean"
            ),

            trend_score_mediano=(
                "trend_score",
                "median"
            ),

            trend_score_p75=(
                "trend_score",
                lambda x:
                np.percentile(
                    x,
                    75
                )
            ),

            trend_score_p90=(
                "trend_score",
                lambda x:
                np.percentile(
                    x,
                    90
                )
            )
        )
        .reindex(
            FAIXAS_RUL
        )
        .reset_index()
    )

    return resumo


# =========================================================
# EXECUÇÃO
# =========================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("TREND SCORE - NASA FD001")
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
    # 2. CALIBRAR
    # -----------------------------------------------------

    (
        referencia_zero,
        referencia_maxima
    ) = calcular_calibracao(
        df
    )

    print("\n" + "=" * 70)
    print("CALIBRAÇÃO DO TREND SCORE")
    print("=" * 70)

    print(
        f"\nReferência 0 "
        f"(mediana saudável): "
        f"{referencia_zero:.10f}"
    )

    print(
        f"Referência 100 "
        f"(percentil {PERCENTIL_SUPERIOR} saudável): "
        f"{referencia_maxima:.10f}"
    )

    # -----------------------------------------------------
    # 3. CALCULAR SCORE
    # -----------------------------------------------------

    df_score = gerar_scores(
        df,
        referencia_zero,
        referencia_maxima
    )

    # -----------------------------------------------------
    # 4. RESUMO
    # -----------------------------------------------------

    resumo = gerar_resumo(
        df_score
    )

    print("\n" + "=" * 70)
    print("TREND SCORE POR FAIXA DE RUL")
    print("=" * 70)

    print(
        "\n"
        + resumo
        .round(2)
        .to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # 5. SALVAR RESULTADOS
    # -----------------------------------------------------

    RESULTADOS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df_score.to_csv(
        TREND_SCORE_PATH,
        index=False
    )

    resumo.to_csv(
        RESUMO_PATH,
        index=False
    )

    config = {

        "dataset": "FD001",

        "fonte": (
            "slope do erro de reconstrucao "
            "do LSTM Autoencoder"
        ),

        "horizonte": HORIZONTE,

        "metodo_slope": (
            "regressao linear "
            "erro_reconstrucao x ciclo"
        ),

        "calibracao_score": {
            "score_min": 0,
            "score_max": 100,

            "referencia_zero": (
                referencia_zero
            ),

            "referencia_maxima": (
                referencia_maxima
            ),

            "referencia_zero_metodo": (
                "mediana dos slopes "
                "com RUL_linear > 125"
            ),

            "referencia_maxima_metodo": (
                f"percentil "
                f"{PERCENTIL_SUPERIOR} "
                f"dos slopes com "
                f"RUL_linear > 125"
            )
        },

        "observacao": (
            "RUL foi utilizado apenas durante "
            "a calibracao experimental do score. "
            "Em inferencia, o Trend Score depende "
            "do slope calculado a partir do erro "
            "de reconstrucao do Autoencoder."
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
        f"\nTrend Score:\n"
        f"{TREND_SCORE_PATH}"
    )

    print(
        f"\nResumo:\n"
        f"{RESUMO_PATH}"
    )

    print(
        f"\nConfig:\n"
        f"{CONFIG_PATH}"
    )