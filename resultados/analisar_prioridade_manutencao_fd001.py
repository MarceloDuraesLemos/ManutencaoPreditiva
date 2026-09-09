from pathlib import Path

import json
import numpy as np
import pandas as pd


# =========================================================
# CAMINHOS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

HEALTH_STATUS_PATH = (
    BASE_DIR
    / "resultados"
    / "fd001_health_score"
    / "health_status_validacao.csv"
)

RESULTADOS_DIR = (
    BASE_DIR
    / "resultados"
    / "fd001_prioridade"
)

PRIORIDADE_PATH = (
    RESULTADOS_DIR
    / "prioridade_manutencao_validacao.csv"
)

RESUMO_RUL_PATH = (
    RESULTADOS_DIR
    / "resumo_prioridade_por_faixa_rul.csv"
)

RESUMO_STATUS_PATH = (
    RESULTADOS_DIR
    / "resumo_prioridade_por_status.csv"
)

RESUMO_GERAL_PATH = (
    RESULTADOS_DIR
    / "resumo_geral_prioridade.csv"
)

CONFIG_PATH = (
    RESULTADOS_DIR
    / "config_prioridade.json"
)


# =========================================================
# LIMITES INICIAIS
# =========================================================

RUL_P1 = 15
RUL_P2 = 45
RUL_P3 = 90

HEALTH_P1 = 30
HEALTH_P2 = 60
HEALTH_P3 = 80


PRIORIDADES = [
    "P1_IMEDIATA",
    "P2_ALTA",
    "P3_MEDIA",
    "P4_BAIXA"
]

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

    if not HEALTH_STATUS_PATH.exists():

        raise FileNotFoundError(
            "Arquivo não encontrado:\n"
            f"{HEALTH_STATUS_PATH}"
        )

    df = pd.read_csv(
        HEALTH_STATUS_PATH
    )

    colunas_necessarias = [
        "unit_number",
        "cycle",
        "RUL_linear",
        "faixa_RUL",
        "anomaly_score",
        "trend_score",
        "health_score",
        "status"
    ]

    faltantes = [
        coluna
        for coluna in colunas_necessarias
        if coluna not in df.columns
    ]

    if faltantes:

        raise ValueError(
            "Colunas ausentes:\n"
            f"{faltantes}"
        )

    return df


# =========================================================
# PRIORIDADE
# =========================================================

def classificar_prioridade(
    rul,
    health
):

    # -----------------------------------------------------
    # P1 - IMEDIATA
    # -----------------------------------------------------

    if (
        health < HEALTH_P1
        or rul <= RUL_P1
    ):
        return "P1_IMEDIATA"

    # -----------------------------------------------------
    # P2 - ALTA
    # -----------------------------------------------------

    if (
        health < HEALTH_P2
        or rul <= RUL_P2
    ):
        return "P2_ALTA"

    # -----------------------------------------------------
    # P3 - MÉDIA
    # -----------------------------------------------------

    if (
        health < HEALTH_P3
        or rul <= RUL_P3
    ):
        return "P3_MEDIA"

    # -----------------------------------------------------
    # P4 - BAIXA
    # -----------------------------------------------------

    return "P4_BAIXA"


# =========================================================
# MOTIVO DA PRIORIDADE
# =========================================================

def gerar_motivo(
    rul,
    health,
    anomaly_score,
    trend_score
):

    motivos = []

    if health < HEALTH_P1:
        motivos.append(
            "Health crítico"
        )

    elif health < HEALTH_P2:
        motivos.append(
            "Health em risco"
        )

    elif health < HEALTH_P3:
        motivos.append(
            "Health em atenção"
        )

    if rul <= RUL_P1:
        motivos.append(
            "RUL muito baixo"
        )

    elif rul <= RUL_P2:
        motivos.append(
            "RUL baixo"
        )

    elif rul <= RUL_P3:
        motivos.append(
            "RUL moderado"
        )

    if anomaly_score >= 50:
        motivos.append(
            "Anomalia detectada"
        )

    if trend_score >= 70:
        motivos.append(
            "Tendência forte de degradação"
        )

    if len(motivos) == 0:
        motivos.append(
            "Condição estável"
        )

    return " | ".join(
        motivos
    )


# =========================================================
# GERAR PRIORIDADE
# =========================================================

def gerar_prioridades(
    df
):

    df = df.copy()

    df["prioridade"] = df.apply(
        lambda linha:
        classificar_prioridade(
            linha["RUL_linear"],
            linha["health_score"]
        ),
        axis=1
    )

    df["motivo_prioridade"] = df.apply(
        lambda linha:
        gerar_motivo(
            linha["RUL_linear"],
            linha["health_score"],
            linha["anomaly_score"],
            linha["trend_score"]
        ),
        axis=1
    )

    return df


# =========================================================
# RESUMO POR FAIXA DE RUL
# =========================================================

def gerar_resumo_por_rul(
    df
):

    tabela = (
        df
        .groupby(
            [
                "faixa_RUL",
                "prioridade"
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
        .reindex(
            index=FAIXAS_RUL,
            columns=PRIORIDADES,
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

    linhas = []

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

        for prioridade in PRIORIDADES:

            linha[
                f"{prioridade}_percentual"
            ] = float(
                percentuais.loc[
                    faixa,
                    prioridade
                ]
            )

        linhas.append(
            linha
        )

    return pd.DataFrame(
        linhas
    )


# =========================================================
# RESUMO POR STATUS
# =========================================================

def gerar_resumo_por_status(
    df
):

    tabela = (
        df
        .groupby(
            [
                "status",
                "prioridade"
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
        .reindex(
            index=STATUS_ORDEM,
            columns=PRIORIDADES,
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

    linhas = []

    for status in STATUS_ORDEM:

        linha = {
            "status": status,
            "quantidade": int(
                tabela
                .loc[
                    status
                ]
                .sum()
            )
        }

        for prioridade in PRIORIDADES:

            linha[
                f"{prioridade}_percentual"
            ] = float(
                percentuais.loc[
                    status,
                    prioridade
                ]
            )

        linhas.append(
            linha
        )

    return pd.DataFrame(
        linhas
    )


# =========================================================
# RESUMO GERAL
# =========================================================

def gerar_resumo_geral(
    df
):

    resumo = (
        df
        .groupby(
            "prioridade"
        )
        .agg(
            quantidade=(
                "prioridade",
                "count"
            ),

            RUL_medio=(
                "RUL_linear",
                "mean"
            ),

            RUL_mediano=(
                "RUL_linear",
                "median"
            ),

            health_medio=(
                "health_score",
                "mean"
            ),

            health_mediano=(
                "health_score",
                "median"
            ),

            anomaly_medio=(
                "anomaly_score",
                "mean"
            ),

            trend_medio=(
                "trend_score",
                "mean"
            )
        )
        .reindex(
            PRIORIDADES
        )
        .reset_index()
    )

    resumo["percentual"] = (
        resumo["quantidade"]
        / len(df)
        * 100
    )

    return resumo


# =========================================================
# VALIDAÇÕES LÓGICAS
# =========================================================

def validar_logica(
    df
):

    print("\n" + "=" * 78)
    print("VALIDAÇÕES LÓGICAS")
    print("=" * 78)

    p1 = df[
        df["prioridade"]
        == "P1_IMEDIATA"
    ]

    p4 = df[
        df["prioridade"]
        == "P4_BAIXA"
    ]

    print(
        f"\nMenor RUL em P4: "
        f"{p4['RUL_linear'].min():.2f}"
        if len(p4) > 0
        else
        "\nNenhuma observação P4."
    )

    print(
        f"Maior RUL em P1: "
        f"{p1['RUL_linear'].max():.2f}"
        if len(p1) > 0
        else
        "Nenhuma observação P1."
    )

    print(
        f"\nHealth máximo em P1: "
        f"{p1['health_score'].max():.2f}"
        if len(p1) > 0
        else
        "\nNenhuma observação P1."
    )

    print(
        f"Health mínimo em P4: "
        f"{p4['health_score'].min():.2f}"
        if len(p4) > 0
        else
        "Nenhuma observação P4."
    )


# =========================================================
# EXECUÇÃO
# =========================================================

if __name__ == "__main__":

    print("\n" + "=" * 78)
    print("VALIDAÇÃO DA PRIORIDADE DE MANUTENÇÃO - FD001")
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
    # 2. GERAR PRIORIDADES
    # -----------------------------------------------------

    df = gerar_prioridades(
        df
    )

    # -----------------------------------------------------
    # 3. DISTRIBUIÇÃO GERAL
    # -----------------------------------------------------

    print("\n" + "=" * 78)
    print("DISTRIBUIÇÃO GERAL DAS PRIORIDADES")
    print("=" * 78)

    distribuicao = (
        df["prioridade"]
        .value_counts()
        .reindex(
            PRIORIDADES,
            fill_value=0
        )
    )

    tabela_distribuicao = pd.DataFrame(
        {
            "quantidade": distribuicao,
            "percentual": (
                distribuicao
                / len(df)
                * 100
            )
        }
    )

    print(
        "\n"
        + tabela_distribuicao
        .round(2)
        .to_string()
    )

    # -----------------------------------------------------
    # 4. POR FAIXA DE RUL
    # -----------------------------------------------------

    resumo_rul = (
        gerar_resumo_por_rul(
            df
        )
    )

    print("\n" + "=" * 78)
    print("PRIORIDADE POR FAIXA DE RUL")
    print("=" * 78)

    print(
        "\n"
        + resumo_rul
        .round(2)
        .to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # 5. POR STATUS
    # -----------------------------------------------------

    resumo_status = (
        gerar_resumo_por_status(
            df
        )
    )

    print("\n" + "=" * 78)
    print("PRIORIDADE POR STATUS")
    print("=" * 78)

    print(
        "\n"
        + resumo_status
        .round(2)
        .to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # 6. RESUMO GERAL
    # -----------------------------------------------------

    resumo_geral = (
        gerar_resumo_geral(
            df
        )
    )

    print("\n" + "=" * 78)
    print("CARACTERÍSTICAS POR PRIORIDADE")
    print("=" * 78)

    print(
        "\n"
        + resumo_geral
        .round(2)
        .to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # 7. VALIDAÇÕES
    # -----------------------------------------------------

    validar_logica(
        df
    )

    # -----------------------------------------------------
    # 8. SALVAR
    # -----------------------------------------------------

    RESULTADOS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        PRIORIDADE_PATH,
        index=False
    )

    resumo_rul.to_csv(
        RESUMO_RUL_PATH,
        index=False
    )

    resumo_status.to_csv(
        RESUMO_STATUS_PATH,
        index=False
    )

    resumo_geral.to_csv(
        RESUMO_GERAL_PATH,
        index=False
    )

    config = {

        "dataset": "FD001",

        "regras_testadas": {

            "P1_IMEDIATA": (
                f"health < {HEALTH_P1} "
                f"OU RUL <= {RUL_P1}"
            ),

            "P2_ALTA": (
                f"health < {HEALTH_P2} "
                f"OU RUL <= {RUL_P2}"
            ),

            "P3_MEDIA": (
                f"health < {HEALTH_P3} "
                f"OU RUL <= {RUL_P3}"
            ),

            "P4_BAIXA": (
                "demais casos"
            )
        },

        "observacao": (
            "Regras ainda em fase de validacao. "
            "A prioridade definitiva deve ser congelada "
            "somente apos analise da distribuicao."
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
        f"\nPrioridades:\n"
        f"{PRIORIDADE_PATH}"
    )

    print(
        f"\nResumo por RUL:\n"
        f"{RESUMO_RUL_PATH}"
    )

    print(
        f"\nResumo por status:\n"
        f"{RESUMO_STATUS_PATH}"
    )

    print(
        f"\nResumo geral:\n"
        f"{RESUMO_GERAL_PATH}"
    )

    print(
        f"\nConfiguração:\n"
        f"{CONFIG_PATH}"
    )