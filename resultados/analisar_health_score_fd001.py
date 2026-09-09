from pathlib import Path

import json
import numpy as np
import pandas as pd


# =========================================================
# CAMINHOS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

ANOMALY_PATH = (
    BASE_DIR
    / "resultados"
    / "fd001_anomalia"
    / "anomaly_score_validacao.csv"
)

TREND_PATH = (
    BASE_DIR
    / "resultados"
    / "fd001_tendencia"
    / "trend_score_horizonte_10.csv"
)

RESULTADOS_DIR = (
    BASE_DIR
    / "resultados"
    / "fd001_health_score"
)

BASE_HEALTH_PATH = (
    RESULTADOS_DIR
    / "base_health_score.csv"
)

COMPARACAO_PATH = (
    RESULTADOS_DIR
    / "comparacao_pesos_health_score.csv"
)

RESUMO_PATH = (
    RESULTADOS_DIR
    / "resumo_configuracoes_health_score.csv"
)

CONFIG_PATH = (
    RESULTADOS_DIR
    / "config_experimento_health_score.json"
)


# =========================================================
# CONFIGURAÇÕES
# =========================================================

RUL_REFERENCIA = 125.0


CONFIGURACOES = {

    "A_50_30_20": {
        "rul": 0.50,
        "anomalia": 0.30,
        "trend": 0.20
    },

    "B_40_40_20": {
        "rul": 0.40,
        "anomalia": 0.40,
        "trend": 0.20
    },

    "C_50_25_25": {
        "rul": 0.50,
        "anomalia": 0.25,
        "trend": 0.25
    },

    "D_60_25_15": {
        "rul": 0.60,
        "anomalia": 0.25,
        "trend": 0.15
    },

    "E_40_35_25": {
        "rul": 0.40,
        "anomalia": 0.35,
        "trend": 0.25
    }
}


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

    if not ANOMALY_PATH.exists():

        raise FileNotFoundError(
            "Arquivo de Anomaly Score não encontrado:\n"
            f"{ANOMALY_PATH}"
        )

    if not TREND_PATH.exists():

        raise FileNotFoundError(
            "Arquivo de Trend Score não encontrado:\n"
            f"{TREND_PATH}"
        )

    df_anomaly = pd.read_csv(
        ANOMALY_PATH
    )

    df_trend = pd.read_csv(
        TREND_PATH
    )

    colunas_anomaly = [
        "unit_number",
        "cycle",
        "RUL_linear",
        "anomaly_score"
    ]

    colunas_trend = [
        "unit_number",
        "cycle",
        "RUL_linear",
        "trend_score"
    ]

    faltantes_anomaly = [
        coluna
        for coluna in colunas_anomaly
        if coluna not in df_anomaly.columns
    ]

    faltantes_trend = [
        coluna
        for coluna in colunas_trend
        if coluna not in df_trend.columns
    ]

    if faltantes_anomaly:

        raise ValueError(
            "Colunas ausentes no arquivo de anomalia:\n"
            f"{faltantes_anomaly}"
        )

    if faltantes_trend:

        raise ValueError(
            "Colunas ausentes no arquivo de tendência:\n"
            f"{faltantes_trend}"
        )

    return (
        df_anomaly,
        df_trend
    )


# =========================================================
# MERGE
# =========================================================

def juntar_dados(
    df_anomaly,
    df_trend
):

    anomaly = (
        df_anomaly[
            [
                "unit_number",
                "cycle",
                "RUL_linear",
                "anomaly_score"
            ]
        ]
        .rename(
            columns={
                "RUL_linear": "RUL_anomaly"
            }
        )
    )

    trend = (
        df_trend[
            [
                "unit_number",
                "cycle",
                "RUL_linear",
                "trend_score"
            ]
        ]
        .rename(
            columns={
                "RUL_linear": "RUL_trend"
            }
        )
    )

    df = pd.merge(
        anomaly,
        trend,
        on=[
            "unit_number",
            "cycle"
        ],
        how="inner",
        validate="one_to_one"
    )

    if len(df) == 0:

        raise ValueError(
            "O merge não encontrou observações em comum."
        )

    diferenca_rul = np.abs(
        df["RUL_anomaly"]
        - df["RUL_trend"]
    )

    if (
        diferenca_rul
        > 1e-9
    ).any():

        raise ValueError(
            "Foram encontradas divergências de RUL "
            "entre os arquivos de Anomaly e Trend."
        )

    df["RUL_linear"] = (
        df["RUL_anomaly"]
    )

    df = df.drop(
        columns=[
            "RUL_anomaly",
            "RUL_trend"
        ]
    )

    return df


# =========================================================
# RUL SCORE
# =========================================================

def calcular_rul_score(
    rul
):

    score = (
        rul
        / RUL_REFERENCIA
    ) * 100

    score = np.clip(
        score,
        0,
        100
    )

    return float(
        score
    )


# =========================================================
# FAIXA DE RUL
# =========================================================

def definir_faixa_rul(
    rul
):

    if rul > 125:
        return "RUL > 125"

    if rul > 75:
        return "76-125"

    if rul > 30:
        return "31-75"

    return "0-30"


# =========================================================
# PREPARAR COMPONENTES
# =========================================================

def preparar_componentes(
    df
):

    df = df.copy()

    df["rul_score"] = (
        df["RUL_linear"]
        .apply(
            calcular_rul_score
        )
    )

    df["saude_anomalia"] = (
        100
        - df["anomaly_score"]
    )

    df["saude_trend"] = (
        100
        - df["trend_score"]
    )

    df["faixa_RUL"] = (
        df["RUL_linear"]
        .apply(
            definir_faixa_rul
        )
    )

    return df


# =========================================================
# HEALTH SCORE
# =========================================================

def calcular_health_score(
    df,
    nome_config,
    pesos
):

    coluna = (
        f"health_{nome_config}"
    )

    df[coluna] = (

        pesos["rul"]
        * df["rul_score"]

        +

        pesos["anomalia"]
        * df["saude_anomalia"]

        +

        pesos["trend"]
        * df["saude_trend"]
    )

    df[coluna] = np.clip(
        df[coluna],
        0,
        100
    )

    return df


# =========================================================
# RESUMO DE UMA CONFIGURAÇÃO
# =========================================================

def gerar_resumo_configuracao(
    df,
    nome_config
):

    coluna = (
        f"health_{nome_config}"
    )

    resumo = (
        df
        .groupby(
            "faixa_RUL"
        )
        .agg(
            quantidade=(
                coluna,
                "count"
            ),

            health_medio=(
                coluna,
                "mean"
            ),

            health_mediano=(
                coluna,
                "median"
            ),

            health_p10=(
                coluna,
                lambda x:
                np.percentile(
                    x,
                    10
                )
            ),

            health_p25=(
                coluna,
                lambda x:
                np.percentile(
                    x,
                    25
                )
            ),

            health_p75=(
                coluna,
                lambda x:
                np.percentile(
                    x,
                    75
                )
            ),

            health_p90=(
                coluna,
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

    resumo.insert(
        0,
        "configuracao",
        nome_config
    )

    return resumo


# =========================================================
# INDICADORES DE QUALIDADE
# =========================================================

def gerar_indicadores(
    resumo,
    nome_config,
    pesos
):

    tabela = (
        resumo
        .set_index(
            "faixa_RUL"
        )
    )

    medianas = []

    medias = []

    for faixa in FAIXAS_RUL:

        medianas.append(
            float(
                tabela.loc[
                    faixa,
                    "health_mediano"
                ]
            )
        )

        medias.append(
            float(
                tabela.loc[
                    faixa,
                    "health_medio"
                ]
            )
        )

    # -----------------------------------------------------
    # Queremos:
    #
    # >125     maior Health
    # 76-125   menor
    # 31-75    menor
    # 0-30     menor
    # -----------------------------------------------------

    monotonia_mediana = all(
        medianas[i]
        >= medianas[i + 1]
        for i in range(
            len(medianas) - 1
        )
    )

    monotonia_media = all(
        medias[i]
        >= medias[i + 1]
        for i in range(
            len(medias) - 1
        )
    )

    queda_mediana_total = (
        medianas[0]
        - medianas[-1]
    )

    queda_media_total = (
        medias[0]
        - medias[-1]
    )

    separacao_1 = (
        medianas[0]
        - medianas[1]
    )

    separacao_2 = (
        medianas[1]
        - medianas[2]
    )

    separacao_3 = (
        medianas[2]
        - medianas[3]
    )

    menor_separacao_mediana = min(
        separacao_1,
        separacao_2,
        separacao_3
    )

    return {

        "configuracao": nome_config,

        "peso_rul": (
            pesos["rul"]
        ),

        "peso_anomalia": (
            pesos["anomalia"]
        ),

        "peso_trend": (
            pesos["trend"]
        ),

        "monotonia_mediana": (
            monotonia_mediana
        ),

        "monotonia_media": (
            monotonia_media
        ),

        "queda_mediana_total": (
            queda_mediana_total
        ),

        "queda_media_total": (
            queda_media_total
        ),

        "separacao_mediana_1": (
            separacao_1
        ),

        "separacao_mediana_2": (
            separacao_2
        ),

        "separacao_mediana_3": (
            separacao_3
        ),

        "menor_separacao_mediana": (
            menor_separacao_mediana
        )
    }


# =========================================================
# EXECUÇÃO
# =========================================================

if __name__ == "__main__":

    print("\n" + "=" * 78)
    print("EXPERIMENTO DE HEALTH SCORE - NASA FD001")
    print("=" * 78)

    # -----------------------------------------------------
    # 1. CARREGAR
    # -----------------------------------------------------

    (
        df_anomaly,
        df_trend
    ) = carregar_dados()

    print(
        f"\nLinhas Anomaly Score: "
        f"{len(df_anomaly)}"
    )

    print(
        f"Linhas Trend Score: "
        f"{len(df_trend)}"
    )

    # -----------------------------------------------------
    # 2. MERGE
    # -----------------------------------------------------

    df = juntar_dados(
        df_anomaly,
        df_trend
    )

    print(
        f"\nLinhas após merge: "
        f"{len(df)}"
    )

    print(
        f"Motores após merge: "
        f"{df['unit_number'].nunique()}"
    )

    # -----------------------------------------------------
    # 3. COMPONENTES
    # -----------------------------------------------------

    df = preparar_componentes(
        df
    )

    print("\n" + "=" * 78)
    print("COMPONENTES DO HEALTH SCORE")
    print("=" * 78)

    print(
        "\nRUL Score:"
        "\n100 = RUL >= 125"
        "\n0 = RUL 0"
    )

    print(
        "\nSaúde de anomalia:"
        "\n100 - Anomaly Score"
    )

    print(
        "\nSaúde de tendência:"
        "\n100 - Trend Score"
    )

    # -----------------------------------------------------
    # 4. TESTAR CONFIGURAÇÕES
    # -----------------------------------------------------

    resumos = []

    indicadores = []

    for (
        nome_config,
        pesos
    ) in CONFIGURACOES.items():

        soma_pesos = sum(
            pesos.values()
        )

        if not np.isclose(
            soma_pesos,
            1.0
        ):

            raise ValueError(
                f"Pesos de {nome_config} "
                f"somam {soma_pesos}, "
                "mas deveriam somar 1."
            )

        df = calcular_health_score(
            df,
            nome_config,
            pesos
        )

        resumo = (
            gerar_resumo_configuracao(
                df,
                nome_config
            )
        )

        indicador = (
            gerar_indicadores(
                resumo,
                nome_config,
                pesos
            )
        )

        resumos.append(
            resumo
        )

        indicadores.append(
            indicador
        )

        print("\n" + "=" * 78)

        print(
            f"CONFIGURAÇÃO {nome_config}"
        )

        print(
            f"RUL={pesos['rul']:.0%} | "
            f"Anomalia={pesos['anomalia']:.0%} | "
            f"Trend={pesos['trend']:.0%}"
        )

        print("=" * 78)

        colunas_exibicao = [
            "faixa_RUL",
            "quantidade",
            "health_medio",
            "health_mediano",
            "health_p10",
            "health_p25",
            "health_p75",
            "health_p90"
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
    # 5. TABELAS FINAIS
    # -----------------------------------------------------

    df_resumos = pd.concat(
        resumos,
        ignore_index=True
    )

    df_indicadores = pd.DataFrame(
        indicadores
    )

    # -----------------------------------------------------
    # Primeiro priorizamos monotonia.
    # Depois a menor separação entre faixas.
    # Depois a queda total.
    # -----------------------------------------------------

    df_indicadores = (
        df_indicadores
        .sort_values(
            by=[
                "monotonia_mediana",
                "monotonia_media",
                "menor_separacao_mediana",
                "queda_mediana_total"
            ],
            ascending=[
                False,
                False,
                False,
                False
            ]
        )
        .reset_index(
            drop=True
        )
    )

    print("\n" + "=" * 78)
    print("COMPARAÇÃO FINAL DAS CONFIGURAÇÕES")
    print("=" * 78)

    colunas_final = [
        "configuracao",
        "peso_rul",
        "peso_anomalia",
        "peso_trend",
        "monotonia_mediana",
        "monotonia_media",
        "queda_mediana_total",
        "menor_separacao_mediana"
    ]

    print(
        "\n"
        + df_indicadores[
            colunas_final
        ]
        .round(2)
        .to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # 6. MELHOR CANDIDATO AUTOMÁTICO
    # -----------------------------------------------------

    melhor = (
        df_indicadores
        .iloc[0]
    )

    print("\n" + "=" * 78)
    print("MELHOR CANDIDATO PELOS CRITÉRIOS ATUAIS")
    print("=" * 78)

    print(
        f"\nConfiguração: "
        f"{melhor['configuracao']}"
    )

    print(
        f"Peso RUL: "
        f"{melhor['peso_rul']:.0%}"
    )

    print(
        f"Peso Anomalia: "
        f"{melhor['peso_anomalia']:.0%}"
    )

    print(
        f"Peso Trend: "
        f"{melhor['peso_trend']:.0%}"
    )

    print(
        f"\nQueda mediana total: "
        f"{melhor['queda_mediana_total']:.2f}"
    )

    print(
        f"Menor separação entre faixas: "
        f"{melhor['menor_separacao_mediana']:.2f}"
    )

    # -----------------------------------------------------
    # 7. SALVAR
    # -----------------------------------------------------

    RESULTADOS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        BASE_HEALTH_PATH,
        index=False
    )

    df_indicadores.to_csv(
        COMPARACAO_PATH,
        index=False
    )

    df_resumos.to_csv(
        RESUMO_PATH,
        index=False
    )

    config = {

        "dataset": "FD001",

        "rul_referencia": (
            RUL_REFERENCIA
        ),

        "rul_score": (
            "clip(RUL_linear / 125 * 100, 0, 100)"
        ),

        "saude_anomalia": (
            "100 - anomaly_score"
        ),

        "saude_trend": (
            "100 - trend_score"
        ),

        "formula": (
            "Health = "
            "peso_rul * rul_score + "
            "peso_anomalia * saude_anomalia + "
            "peso_trend * saude_trend"
        ),

        "configuracoes_testadas": (
            CONFIGURACOES
        ),

        "criterios_selecao": [
            "monotonia da mediana",
            "monotonia da media",
            "menor separacao mediana entre faixas",
            "queda mediana total"
        ],

        "observacao": (
            "A selecao automatica e apenas um criterio "
            "experimental. A configuracao definitiva deve "
            "ser analisada antes de ser congelada."
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
        f"\nBase completa:\n"
        f"{BASE_HEALTH_PATH}"
    )

    print(
        f"\nComparação:\n"
        f"{COMPARACAO_PATH}"
    )

    print(
        f"\nResumos:\n"
        f"{RESUMO_PATH}"
    )

    print(
        f"\nConfiguração:\n"
        f"{CONFIG_PATH}"
    )