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
    / "health_status_validacao.csv"
)

RESULTADOS_DIR = (
    BASE_DIR
    / "resultados"
    / "fd001_priority_score"
)

BASE_PRIORITY_PATH = (
    RESULTADOS_DIR
    / "base_priority_score.csv"
)

COMPARACAO_PATH = (
    RESULTADOS_DIR
    / "comparacao_priority_score.csv"
)

RESUMO_PATH = (
    RESULTADOS_DIR
    / "resumo_configuracoes_priority_score.csv"
)

CONFIG_PATH = (
    RESULTADOS_DIR
    / "config_experimento_priority_score.json"
)


# =========================================================
# CONFIGURAÇÕES
# =========================================================

RUL_REFERENCIA = 125.0


CONFIGURACOES = {

    "A_50_50": {
        "health_risk": 0.50,
        "rul_risk": 0.50
    },

    "B_60_40": {
        "health_risk": 0.60,
        "rul_risk": 0.40
    },

    "C_40_60": {
        "health_risk": 0.40,
        "rul_risk": 0.60
    },

    "D_70_30": {
        "health_risk": 0.70,
        "rul_risk": 0.30
    },

    "E_30_70": {
        "health_risk": 0.30,
        "rul_risk": 0.70
    }
}


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
        "health_score",
        "status"
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
# RUL SCORE / RISK
# =========================================================

def calcular_rul_score(rul):

    score = (
        rul
        / RUL_REFERENCIA
    ) * 100

    score = np.clip(
        score,
        0,
        100
    )

    return float(score)


# =========================================================
# PREPARAR RISCOS
# =========================================================

def preparar_riscos(df):

    df = df.copy()

    df["rul_score"] = (
        df["RUL_linear"]
        .apply(
            calcular_rul_score
        )
    )

    df["rul_risk"] = (
        100
        - df["rul_score"]
    )

    df["health_risk"] = (
        100
        - df["health_score"]
    )

    return df


# =========================================================
# PRIORITY SCORE
# =========================================================

def calcular_priority_score(
    df,
    nome_config,
    pesos
):

    coluna = (
        f"priority_{nome_config}"
    )

    df[coluna] = (

        pesos["health_risk"]
        * df["health_risk"]

        +

        pesos["rul_risk"]
        * df["rul_risk"]
    )

    df[coluna] = np.clip(
        df[coluna],
        0,
        100
    )

    return df


# =========================================================
# RESUMO POR FAIXA DE RUL
# =========================================================

def gerar_resumo_configuracao(
    df,
    nome_config
):

    coluna = (
        f"priority_{nome_config}"
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

            priority_medio=(
                coluna,
                "mean"
            ),

            priority_mediano=(
                coluna,
                "median"
            ),

            priority_p10=(
                coluna,
                lambda x:
                np.percentile(
                    x,
                    10
                )
            ),

            priority_p25=(
                coluna,
                lambda x:
                np.percentile(
                    x,
                    25
                )
            ),

            priority_p75=(
                coluna,
                lambda x:
                np.percentile(
                    x,
                    75
                )
            ),

            priority_p90=(
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
# INDICADORES
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
                    "priority_mediano"
                ]
            )
        )

        medias.append(
            float(
                tabela.loc[
                    faixa,
                    "priority_medio"
                ]
            )
        )

    # -----------------------------------------------------
    # Agora queremos o contrário do Health:
    #
    # >125      menor Priority
    # 76-125    maior
    # 31-75     maior
    # 0-30      maior
    # -----------------------------------------------------

    monotonia_mediana = all(
        medianas[i]
        <= medianas[i + 1]
        for i in range(
            len(medianas) - 1
        )
    )

    monotonia_media = all(
        medias[i]
        <= medias[i + 1]
        for i in range(
            len(medias) - 1
        )
    )

    crescimento_mediana_total = (
        medianas[-1]
        - medianas[0]
    )

    crescimento_media_total = (
        medias[-1]
        - medias[0]
    )

    separacao_1 = (
        medianas[1]
        - medianas[0]
    )

    separacao_2 = (
        medianas[2]
        - medianas[1]
    )

    separacao_3 = (
        medianas[3]
        - medianas[2]
    )

    menor_separacao_mediana = min(
        separacao_1,
        separacao_2,
        separacao_3
    )

    return {

        "configuracao": nome_config,

        "peso_health_risk": (
            pesos["health_risk"]
        ),

        "peso_rul_risk": (
            pesos["rul_risk"]
        ),

        "monotonia_mediana": (
            monotonia_mediana
        ),

        "monotonia_media": (
            monotonia_media
        ),

        "crescimento_mediana_total": (
            crescimento_mediana_total
        ),

        "crescimento_media_total": (
            crescimento_media_total
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
    print("EXPERIMENTO DE PRIORITY SCORE - NASA FD001")
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
    # 2. PREPARAR RISCOS
    # -----------------------------------------------------

    df = preparar_riscos(
        df
    )

    print("\n" + "=" * 78)
    print("COMPONENTES DO PRIORITY SCORE")
    print("=" * 78)

    print(
        "\nHealth Risk:"
        "\n100 - Health Score"
    )

    print(
        "\nRUL Risk:"
        "\n100 - RUL Score"
    )

    print(
        "\nPriority Score:"
        "\n0 = baixa urgência"
        "\n100 = urgência máxima"
    )

    # -----------------------------------------------------
    # 3. TESTAR CONFIGURAÇÕES
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

        df = calcular_priority_score(
            df,
            nome_config,
            pesos
        )

        resumo = gerar_resumo_configuracao(
            df,
            nome_config
        )

        indicador = gerar_indicadores(
            resumo,
            nome_config,
            pesos
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
            f"Health Risk="
            f"{pesos['health_risk']:.0%} | "
            f"RUL Risk="
            f"{pesos['rul_risk']:.0%}"
        )

        print("=" * 78)

        colunas_exibicao = [
            "faixa_RUL",
            "quantidade",
            "priority_medio",
            "priority_mediano",
            "priority_p10",
            "priority_p25",
            "priority_p75",
            "priority_p90"
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
    # 4. TABELAS FINAIS
    # -----------------------------------------------------

    df_resumos = pd.concat(
        resumos,
        ignore_index=True
    )

    df_indicadores = pd.DataFrame(
        indicadores
    )

    df_indicadores = (
        df_indicadores
        .sort_values(
            by=[
                "monotonia_mediana",
                "monotonia_media",
                "menor_separacao_mediana",
                "crescimento_mediana_total"
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
        "peso_health_risk",
        "peso_rul_risk",
        "monotonia_mediana",
        "monotonia_media",
        "crescimento_mediana_total",
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
    # 5. MELHOR CANDIDATO
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
        f"Peso Health Risk: "
        f"{melhor['peso_health_risk']:.0%}"
    )

    print(
        f"Peso RUL Risk: "
        f"{melhor['peso_rul_risk']:.0%}"
    )

    print(
        f"\nCrescimento mediano total: "
        f"{melhor['crescimento_mediana_total']:.2f}"
    )

    print(
        f"Menor separação entre faixas: "
        f"{melhor['menor_separacao_mediana']:.2f}"
    )

    # -----------------------------------------------------
    # 6. SALVAR
    # -----------------------------------------------------

    RESULTADOS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        BASE_PRIORITY_PATH,
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

        "health_risk": (
            "100 - health_score"
        ),

        "rul_risk": (
            "100 - clip(RUL_linear / 125 * 100, 0, 100)"
        ),

        "formula": (
            "Priority = "
            "peso_health_risk * health_risk + "
            "peso_rul_risk * rul_risk"
        ),

        "configuracoes_testadas": (
            CONFIGURACOES
        ),

        "criterios_selecao": [
            "monotonia da mediana",
            "monotonia da media",
            "menor separacao mediana entre faixas",
            "crescimento mediano total"
        ],

        "observacao": (
            "Priority Score representa urgencia operacional. "
            "0 significa baixa urgencia e 100 significa "
            "urgencia maxima."
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
        f"{BASE_PRIORITY_PATH}"
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