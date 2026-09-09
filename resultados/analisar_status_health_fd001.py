from pathlib import Path

import json
import numpy as np
import pandas as pd


# =========================================================
# CAMINHOS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

HEALTH_PATH = (
    BASE_DIR
    / "resultados"
    / "fd001_health_score"
    / "base_health_score.csv"
)

RESULTADOS_DIR = (
    BASE_DIR
    / "resultados"
    / "fd001_health_score"
)

RESULTADO_STATUS_PATH = (
    RESULTADOS_DIR
    / "health_status_validacao.csv"
)

RESUMO_STATUS_PATH = (
    RESULTADOS_DIR
    / "resumo_status_por_faixa_rul.csv"
)

CONFIG_PATH = (
    RESULTADOS_DIR
    / "config_health_status.json"
)


# =========================================================
# CONFIGURAÇÃO ESCOLHIDA
# =========================================================

COLUNA_HEALTH = "health_D_60_25_15"


# =========================================================
# LIMITES INICIAIS
# =========================================================

LIMITE_SAUDAVEL = 80
LIMITE_ATENCAO = 60
LIMITE_RISCO = 30


FAIXAS_RUL = [
    "RUL > 125",
    "76-125",
    "31-75",
    "0-30"
]

STATUS_ORDEM = [
    "SAUDAVEL",
    "ATENCAO",
    "RISCO",
    "CRITICO"
]


# =========================================================
# CARREGAMENTO
# =========================================================

def carregar_dados():

    if not HEALTH_PATH.exists():

        raise FileNotFoundError(
            "Arquivo não encontrado:\n"
            f"{HEALTH_PATH}"
        )

    df = pd.read_csv(
        HEALTH_PATH
    )

    colunas = [
        "unit_number",
        "cycle",
        "RUL_linear",
        "faixa_RUL",
        COLUNA_HEALTH
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
# STATUS
# =========================================================

def classificar_status(
    health
):

    if health >= LIMITE_SAUDAVEL:
        return "SAUDAVEL"

    if health >= LIMITE_ATENCAO:
        return "ATENCAO"

    if health >= LIMITE_RISCO:
        return "RISCO"

    return "CRITICO"


# =========================================================
# GERAR STATUS
# =========================================================

def gerar_status(
    df
):

    df = df.copy()

    df["health_score"] = (
        df[COLUNA_HEALTH]
    )

    df["status"] = (
        df["health_score"]
        .apply(
            classificar_status
        )
    )

    return df


# =========================================================
# RESUMO
# =========================================================

def gerar_resumo(
    df
):

    tabela = (
        df
        .groupby(
            [
                "faixa_RUL",
                "status"
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    tabela = (
        tabela
        .reindex(
            index=FAIXAS_RUL,
            columns=STATUS_ORDEM,
            fill_value=0
        )
    )

    percentuais = (
        tabela
        .div(
            tabela.sum(
                axis=1
            ),
            axis=0
        )
        * 100
    )

    resumo = []

    for faixa in FAIXAS_RUL:

        linha = {
            "faixa_RUL": faixa,
            "quantidade": int(
                tabela
                .loc[
                    faixa
                ]
                .sum()
            )
        }

        for status in STATUS_ORDEM:

            linha[
                f"{status}_quantidade"
            ] = int(
                tabela.loc[
                    faixa,
                    status
                ]
            )

            linha[
                f"{status}_percentual"
            ] = float(
                percentuais.loc[
                    faixa,
                    status
                ]
            )

        resumo.append(
            linha
        )

    return pd.DataFrame(
        resumo
    )


# =========================================================
# EXECUÇÃO
# =========================================================

if __name__ == "__main__":

    print("\n" + "=" * 78)
    print("VALIDAÇÃO DOS STATUS DO HEALTH SCORE - FD001")
    print("=" * 78)

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
    # 2. STATUS
    # -----------------------------------------------------

    df = gerar_status(
        df
    )

    print("\n" + "=" * 78)
    print("LIMITES TESTADOS")
    print("=" * 78)

    print(
        f"\nSAUDAVEL: "
        f"{LIMITE_SAUDAVEL}–100"
    )

    print(
        f"ATENCAO: "
        f"{LIMITE_ATENCAO}–"
        f"{LIMITE_SAUDAVEL - 0.01}"
    )

    print(
        f"RISCO: "
        f"{LIMITE_RISCO}–"
        f"{LIMITE_ATENCAO - 0.01}"
    )

    print(
        f"CRITICO: "
        f"0–{LIMITE_RISCO - 0.01}"
    )

    # -----------------------------------------------------
    # 3. DISTRIBUIÇÃO GERAL
    # -----------------------------------------------------

    print("\n" + "=" * 78)
    print("DISTRIBUIÇÃO GERAL DOS STATUS")
    print("=" * 78)

    distribuicao = (
        df["status"]
        .value_counts()
        .reindex(
            STATUS_ORDEM,
            fill_value=0
        )
    )

    percentual = (
        distribuicao
        / len(df)
        * 100
    )

    tabela_geral = pd.DataFrame(
        {
            "quantidade": distribuicao,
            "percentual": percentual
        }
    )

    print(
        "\n"
        + tabela_geral
        .round(2)
        .to_string()
    )

    # -----------------------------------------------------
    # 4. RESUMO POR FAIXA DE RUL
    # -----------------------------------------------------

    resumo = gerar_resumo(
        df
    )

    print("\n" + "=" * 78)
    print("STATUS POR FAIXA DE RUL")
    print("=" * 78)

    colunas_exibicao = [
        "faixa_RUL",
        "quantidade",
        "SAUDAVEL_percentual",
        "ATENCAO_percentual",
        "RISCO_percentual",
        "CRITICO_percentual"
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
    # 5. HEALTH SCORE MÉDIO POR STATUS
    # -----------------------------------------------------

    print("\n" + "=" * 78)
    print("HEALTH SCORE POR STATUS")
    print("=" * 78)

    resumo_health = (
        df
        .groupby(
            "status"
        )
        ["health_score"]
        .agg(
            [
                "count",
                "mean",
                "median",
                "min",
                "max"
            ]
        )
        .reindex(
            STATUS_ORDEM
        )
    )

    print(
        "\n"
        + resumo_health
        .round(2)
        .to_string()
    )

    # -----------------------------------------------------
    # 6. SALVAR
    # -----------------------------------------------------

    RESULTADOS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        RESULTADO_STATUS_PATH,
        index=False
    )

    resumo.to_csv(
        RESUMO_STATUS_PATH,
        index=False
    )

    config = {

        "dataset": "FD001",

        "health_score_config": (
            "D_60_25_15"
        ),

        "pesos": {
            "rul": 0.60,
            "anomalia": 0.25,
            "trend": 0.15
        },

        "status_testados": {

            "SAUDAVEL": {
                "min": LIMITE_SAUDAVEL,
                "max": 100
            },

            "ATENCAO": {
                "min": LIMITE_ATENCAO,
                "max": LIMITE_SAUDAVEL
            },

            "RISCO": {
                "min": LIMITE_RISCO,
                "max": LIMITE_ATENCAO
            },

            "CRITICO": {
                "min": 0,
                "max": LIMITE_RISCO
            }
        },

        "observacao": (
            "Limites ainda em fase de validacao. "
            "Devem ser congelados somente apos "
            "analise da distribuicao por faixa de RUL."
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

    print("\n" + "=" * 78)
    print("ARQUIVOS SALVOS")
    print("=" * 78)

    print(
        f"\nResultados:\n"
        f"{RESULTADO_STATUS_PATH}"
    )

    print(
        f"\nResumo:\n"
        f"{RESUMO_STATUS_PATH}"
    )

    print(
        f"\nConfiguração:\n"
        f"{CONFIG_PATH}"
    )