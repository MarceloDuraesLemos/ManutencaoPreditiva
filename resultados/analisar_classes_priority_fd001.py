from pathlib import Path

import json
import numpy as np
import pandas as pd


# =========================================================
# CAMINHOS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PRIORITY_PATH = (
    BASE_DIR
    / "resultados"
    / "fd001_priority_score"
    / "base_priority_score.csv"
)

RESULTADOS_DIR = (
    BASE_DIR
    / "resultados"
    / "fd001_priority_score"
)

RESUMO_PATH = (
    RESULTADOS_DIR
    / "comparacao_classes_priority.csv"
)

DETALHES_PATH = (
    RESULTADOS_DIR
    / "detalhes_classes_priority.csv"
)

CONFIG_PATH = (
    RESULTADOS_DIR
    / "config_classes_priority.json"
)


# =========================================================
# PRIORITY SCORE ESCOLHIDO
# =========================================================

COLUNA_PRIORITY = "priority_E_30_70"


# =========================================================
# CONJUNTOS DE LIMITES
#
# Formato:
#
# P4: score < limite_p3
# P3: limite_p3 <= score < limite_p2
# P2: limite_p2 <= score < limite_p1
# P1: score >= limite_p1
# =========================================================

CONFIGURACOES = {

    "A_20_50_80": {
        "p3": 20,
        "p2": 50,
        "p1": 80
    },

    "B_15_45_75": {
        "p3": 15,
        "p2": 45,
        "p1": 75
    },

    "C_20_45_75": {
        "p3": 20,
        "p2": 45,
        "p1": 75
    },

    "D_25_50_75": {
        "p3": 25,
        "p2": 50,
        "p1": 75
    },

    "E_20_50_75": {
        "p3": 20,
        "p2": 50,
        "p1": 75
    }
}


FAIXAS_RUL = [
    "RUL > 125",
    "76-125",
    "31-75",
    "0-30"
]


PRIORIDADES = [
    "P4_BAIXA",
    "P3_MEDIA",
    "P2_ALTA",
    "P1_IMEDIATA"
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

    if not PRIORITY_PATH.exists():

        raise FileNotFoundError(
            "Arquivo não encontrado:\n"
            f"{PRIORITY_PATH}"
        )

    df = pd.read_csv(
        PRIORITY_PATH
    )

    colunas = [
        "unit_number",
        "cycle",
        "RUL_linear",
        "faixa_RUL",
        "health_score",
        "status",
        COLUNA_PRIORITY
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
# CLASSIFICAÇÃO
# =========================================================

def classificar_priority(
    score,
    limites
):

    if score >= limites["p1"]:
        return "P1_IMEDIATA"

    if score >= limites["p2"]:
        return "P2_ALTA"

    if score >= limites["p3"]:
        return "P3_MEDIA"

    return "P4_BAIXA"


# =========================================================
# RESUMO POR RUL
# =========================================================

def resumo_por_rul(
    df,
    coluna_prioridade
):

    tabela = (
        df
        .groupby(
            [
                "faixa_RUL",
                coluna_prioridade
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
            tabela.sum(axis=1),
            axis=0
        )
        * 100
    )

    return percentuais


# =========================================================
# RESUMO POR STATUS
# =========================================================

def resumo_por_status(
    df,
    coluna_prioridade
):

    tabela = (
        df
        .groupby(
            [
                "status",
                coluna_prioridade
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
            tabela.sum(axis=1),
            axis=0
        )
        * 100
    )

    return percentuais


# =========================================================
# INDICADORES
# =========================================================

def calcular_indicadores(
    nome,
    limites,
    tabela_rul,
    tabela_status
):

    # -----------------------------------------------------
    # Desejos principais:
    #
    # RUL >125 -> P4
    # 76-125   -> P3/P4
    # 31-75    -> P2 principalmente
    # 0-30     -> P1 principalmente
    #
    # Sem forçar equivalência perfeita com Status.
    # -----------------------------------------------------

    p4_saudavel_rul = float(
        tabela_rul.loc[
            "RUL > 125",
            "P4_BAIXA"
        ]
    )

    p1_final_rul = float(
        tabela_rul.loc[
            "0-30",
            "P1_IMEDIATA"
        ]
    )

    p2_intermediario_rul = float(
        tabela_rul.loc[
            "31-75",
            "P2_ALTA"
        ]
    )

    p1_rul_saudavel = float(
        tabela_rul.loc[
            "RUL > 125",
            "P1_IMEDIATA"
        ]
    )

    p4_rul_final = float(
        tabela_rul.loc[
            "0-30",
            "P4_BAIXA"
        ]
    )

    # -----------------------------------------------------
    # Penalização de casos perigosos
    # -----------------------------------------------------

    penalizacao = (
        p1_rul_saudavel
        +
        p4_rul_final
    )

    # -----------------------------------------------------
    # Score experimental apenas para ajudar na comparação
    # -----------------------------------------------------

    score_criterio = (
        p4_saudavel_rul
        +
        p1_final_rul
        +
        p2_intermediario_rul
        -
        penalizacao
    )

    return {

        "configuracao": nome,

        "limite_P3": limites["p3"],
        "limite_P2": limites["p2"],
        "limite_P1": limites["p1"],

        "P4_em_RUL_maior_125": (
            p4_saudavel_rul
        ),

        "P2_em_RUL_31_75": (
            p2_intermediario_rul
        ),

        "P1_em_RUL_0_30": (
            p1_final_rul
        ),

        "P1_em_RUL_maior_125": (
            p1_rul_saudavel
        ),

        "P4_em_RUL_0_30": (
            p4_rul_final
        ),

        "score_criterio": (
            score_criterio
        )
    }


# =========================================================
# EXECUÇÃO
# =========================================================

if __name__ == "__main__":

    print("\n" + "=" * 78)
    print("VALIDAÇÃO DAS CLASSES DE PRIORIDADE - NASA FD001")
    print("=" * 78)

    # -----------------------------------------------------
    # 1. CARREGAR
    # -----------------------------------------------------

    df = carregar_dados()

    df["priority_score"] = (
        df[COLUNA_PRIORITY]
    )

    print(
        f"\nLinhas carregadas: "
        f"{len(df)}"
    )

    print(
        f"Motores: "
        f"{df['unit_number'].nunique()}"
    )

    print(
        "\nPriority Score utilizado:"
        "\n30% Health Risk + 70% RUL Risk"
    )

    # -----------------------------------------------------
    # 2. TESTAR CONFIGURAÇÕES
    # -----------------------------------------------------

    indicadores = []

    detalhes = []

    for nome, limites in CONFIGURACOES.items():

        coluna_prioridade = (
            f"classe_{nome}"
        )

        df[coluna_prioridade] = (
            df["priority_score"]
            .apply(
                lambda score:
                classificar_priority(
                    score,
                    limites
                )
            )
        )

        tabela_rul = resumo_por_rul(
            df,
            coluna_prioridade
        )

        tabela_status = resumo_por_status(
            df,
            coluna_prioridade
        )

        indicador = calcular_indicadores(
            nome,
            limites,
            tabela_rul,
            tabela_status
        )

        indicadores.append(
            indicador
        )

        # -------------------------------------------------
        # Salvar tabela detalhada em formato longo
        # -------------------------------------------------

        for faixa in FAIXAS_RUL:

            linha = {
                "configuracao": nome,
                "tipo": "RUL",
                "grupo": faixa
            }

            for prioridade in PRIORIDADES:

                linha[prioridade] = float(
                    tabela_rul.loc[
                        faixa,
                        prioridade
                    ]
                )

            detalhes.append(
                linha
            )

        for status in STATUS_ORDEM:

            linha = {
                "configuracao": nome,
                "tipo": "STATUS",
                "grupo": status
            }

            for prioridade in PRIORIDADES:

                linha[prioridade] = float(
                    tabela_status.loc[
                        status,
                        prioridade
                    ]
                )

            detalhes.append(
                linha
            )

        # -------------------------------------------------
        # EXIBIÇÃO
        # -------------------------------------------------

        print("\n" + "=" * 78)

        print(
            f"CONFIGURAÇÃO {nome}"
        )

        print(
            f"P4 < {limites['p3']}"
        )

        print(
            f"P3 >= {limites['p3']}"
            f" e < {limites['p2']}"
        )

        print(
            f"P2 >= {limites['p2']}"
            f" e < {limites['p1']}"
        )

        print(
            f"P1 >= {limites['p1']}"
        )

        print("=" * 78)

        print(
            "\nPOR FAIXA DE RUL (%)\n"
        )

        print(
            tabela_rul
            .round(2)
            .to_string()
        )

        print(
            "\nPOR STATUS (%)\n"
        )

        print(
            tabela_status
            .round(2)
            .to_string()
        )

    # -----------------------------------------------------
    # 3. COMPARAÇÃO FINAL
    # -----------------------------------------------------

    df_indicadores = pd.DataFrame(
        indicadores
    )

    df_indicadores = (
        df_indicadores
        .sort_values(
            by="score_criterio",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    print("\n" + "=" * 78)
    print("COMPARAÇÃO FINAL")
    print("=" * 78)

    colunas_exibir = [
        "configuracao",
        "limite_P3",
        "limite_P2",
        "limite_P1",
        "P4_em_RUL_maior_125",
        "P2_em_RUL_31_75",
        "P1_em_RUL_0_30",
        "score_criterio"
    ]

    print(
        "\n"
        + df_indicadores[
            colunas_exibir
        ]
        .round(2)
        .to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # 4. MELHOR AUTOMÁTICO
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
        f"P3 começa em: "
        f"{melhor['limite_P3']:.0f}"
    )

    print(
        f"P2 começa em: "
        f"{melhor['limite_P2']:.0f}"
    )

    print(
        f"P1 começa em: "
        f"{melhor['limite_P1']:.0f}"
    )

    # -----------------------------------------------------
    # 5. SALVAR
    # -----------------------------------------------------

    RESULTADOS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df_indicadores.to_csv(
        RESUMO_PATH,
        index=False
    )

    pd.DataFrame(
        detalhes
    ).to_csv(
        DETALHES_PATH,
        index=False
    )

    config = {

        "dataset": "FD001",

        "priority_score": {
            "formula": (
                "0.30 * health_risk + "
                "0.70 * rul_risk"
            ),

            "direcao": (
                "0 = baixa urgencia; "
                "100 = urgencia maxima"
            )
        },

        "classes_testadas": (
            CONFIGURACOES
        ),

        "observacao": (
            "Os limites de P1/P2/P3/P4 "
            "ainda nao estao congelados. "
            "O ranking automatico serve apenas "
            "como apoio para a decisao final."
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
        f"\nComparação:\n"
        f"{RESUMO_PATH}"
    )

    print(
        f"\nDetalhes:\n"
        f"{DETALHES_PATH}"
    )

    print(
        f"\nConfiguração:\n"
        f"{CONFIG_PATH}"
    )