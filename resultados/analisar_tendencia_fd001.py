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
    / "fd001_tendencia"
)

ARQUIVO_COMPARACAO = (
    RESULTADOS_DIR
    / "comparacao_horizontes.csv"
)

ARQUIVO_CONFIG = (
    RESULTADOS_DIR
    / "config_tendencia.json"
)


# =========================================================
# CONFIGURAÇÕES
# =========================================================

HORIZONTES = [
    5,
    10,
    20
]

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
            "Arquivo de anomalias não encontrado:\n"
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
            "As seguintes colunas não foram encontradas "
            f"no CSV:\n{faltantes}"
        )

    return df


# =========================================================
# SLOPE
# =========================================================

def calcular_slope(
    ciclos,
    erros
):

    ciclos = np.asarray(
        ciclos,
        dtype=float
    )

    erros = np.asarray(
        erros,
        dtype=float
    )

    if len(ciclos) < 2:
        return np.nan

    if np.all(
        ciclos == ciclos[0]
    ):
        return np.nan

    slope = np.polyfit(
        ciclos,
        erros,
        1
    )[0]

    return float(
        slope
    )


# =========================================================
# TENDÊNCIA MÓVEL
# =========================================================

def calcular_tendencia(
    df,
    horizonte
):

    resultados = []

    for unit_number in sorted(
        df["unit_number"].unique()
    ):

        dados_motor = (
            df[
                df["unit_number"]
                == unit_number
            ]
            .sort_values(
                "cycle"
            )
            .reset_index(drop=True)
        )

        # Só calculamos quando já existem
        # "horizonte" observações disponíveis.
        for indice in range(
            horizonte - 1,
            len(dados_motor)
        ):

            inicio = (
                indice
                - horizonte
                + 1
            )

            trecho = (
                dados_motor
                .iloc[
                    inicio:
                    indice + 1
                ]
            )

            slope = calcular_slope(
                trecho["cycle"].values,
                trecho[
                    "erro_reconstrucao"
                ].values
            )

            linha_atual = (
                dados_motor
                .iloc[indice]
            )

            resultados.append({
                "unit_number": int(
                    linha_atual[
                        "unit_number"
                    ]
                ),

                "cycle": int(
                    linha_atual[
                        "cycle"
                    ]
                ),

                "RUL_linear": float(
                    linha_atual[
                        "RUL_linear"
                    ]
                ),

                "erro_reconstrucao": float(
                    linha_atual[
                        "erro_reconstrucao"
                    ]
                ),

                "slope": float(
                    slope
                ),

                "horizonte": int(
                    horizonte
                )
            })

    return pd.DataFrame(
        resultados
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


def adicionar_faixa_rul(
    df
):

    df = df.copy()

    df["faixa_RUL"] = (
        df[
            "RUL_linear"
        ]
        .apply(
            definir_faixa_rul
        )
    )

    return df


# =========================================================
# RESUMO POR FAIXA
# =========================================================

def gerar_resumo(
    df,
    horizonte
):

    df = df.copy()

    df[
        "tendencia_positiva"
    ] = (
        df["slope"] > 0
    ).astype(int)

    resumo = (
        df
        .groupby(
            "faixa_RUL"
        )
        .agg(
            quantidade=(
                "slope",
                "count"
            ),

            slope_medio=(
                "slope",
                "mean"
            ),

            slope_mediano=(
                "slope",
                "median"
            ),

            slope_desvio_padrao=(
                "slope",
                "std"
            ),

            taxa_tendencia_positiva=(
                "tendencia_positiva",
                "mean"
            )
        )
        .reindex(
            FAIXAS_RUL
        )
        .reset_index()
    )

    resumo[
        "taxa_tendencia_positiva_percentual"
    ] = (
        resumo[
            "taxa_tendencia_positiva"
        ]
        * 100
    )

    resumo.insert(
        0,
        "horizonte",
        horizonte
    )

    return resumo


# =========================================================
# INDICADORES DE SEPARAÇÃO
# =========================================================

def calcular_indicadores(
    resumo,
    horizonte
):

    tabela = (
        resumo
        .set_index(
            "faixa_RUL"
        )
    )

    indicadores = {
        "horizonte": horizonte
    }

    for faixa in FAIXAS_RUL:

        if faixa not in tabela.index:
            continue

        indicadores[
            f"slope_mediano_{faixa}"
        ] = float(
            tabela.loc[
                faixa,
                "slope_mediano"
            ]
        )

        indicadores[
            f"taxa_positiva_{faixa}"
        ] = float(
            tabela.loc[
                faixa,
                "taxa_tendencia_positiva_percentual"
            ]
        )

    if (
        "RUL > 125" in tabela.index
        and "0-30" in tabela.index
    ):

        slope_inicio = float(
            tabela.loc[
                "RUL > 125",
                "slope_mediano"
            ]
        )

        slope_final = float(
            tabela.loc[
                "0-30",
                "slope_mediano"
            ]
        )

        taxa_inicio = float(
            tabela.loc[
                "RUL > 125",
                "taxa_tendencia_positiva_percentual"
            ]
        )

        taxa_final = float(
            tabela.loc[
                "0-30",
                "taxa_tendencia_positiva_percentual"
            ]
        )

        indicadores[
            "diferenca_slope_mediano_final_inicio"
        ] = (
            slope_final
            - slope_inicio
        )

        indicadores[
            "diferenca_taxa_positiva_final_inicio"
        ] = (
            taxa_final
            - taxa_inicio
        )

    return indicadores


# =========================================================
# EXECUÇÃO
# =========================================================

if __name__ == "__main__":

    print(
        "\n"
        + "=" * 70
    )

    print(
        "ANÁLISE DE TENDÊNCIA DE DEGRADAÇÃO - FD001"
    )

    print(
        "=" * 70
    )

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

    print(
        "\nHorizontes que serão testados:"
    )

    print(
        HORIZONTES
    )

    # -----------------------------------------------------
    # 2. CRIAR DIRETÓRIO
    # -----------------------------------------------------

    RESULTADOS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------------------------
    # 3. TESTAR HORIZONTES
    # -----------------------------------------------------

    resumos = []

    indicadores = []

    for horizonte in HORIZONTES:

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"HORIZONTE = "
            f"{horizonte} JANELAS"
        )

        print(
            "=" * 70
        )

        df_tendencia = (
            calcular_tendencia(
                df,
                horizonte
            )
        )

        df_tendencia = (
            adicionar_faixa_rul(
                df_tendencia
            )
        )

        resumo = gerar_resumo(
            df_tendencia,
            horizonte
        )

        indicador = (
            calcular_indicadores(
                resumo,
                horizonte
            )
        )

        resumos.append(
            resumo
        )

        indicadores.append(
            indicador
        )

        print(
            "\nResumo por faixa de RUL:\n"
        )

        colunas_exibicao = [
            "faixa_RUL",
            "quantidade",
            "slope_medio",
            "slope_mediano",
            "taxa_tendencia_positiva_percentual"
        ]

        print(
            resumo[
                colunas_exibicao
            ]
            .round(8)
            .to_string(
                index=False
            )
        )

        # ---------------------------------------------
        # Salvar resultados detalhados do horizonte
        # ---------------------------------------------

        caminho_detalhado = (
            RESULTADOS_DIR
            / (
                f"tendencia_horizonte_"
                f"{horizonte}.csv"
            )
        )

        df_tendencia.to_csv(
            caminho_detalhado,
            index=False
        )

        caminho_resumo = (
            RESULTADOS_DIR
            / (
                f"resumo_horizonte_"
                f"{horizonte}.csv"
            )
        )

        resumo.to_csv(
            caminho_resumo,
            index=False
        )

    # -----------------------------------------------------
    # 4. COMPARAÇÃO COMPLETA
    # -----------------------------------------------------

    df_resumos = pd.concat(
        resumos,
        ignore_index=True
    )

    df_resumos.to_csv(
        ARQUIVO_COMPARACAO,
        index=False
    )

    df_indicadores = pd.DataFrame(
        indicadores
    )

    caminho_indicadores = (
        RESULTADOS_DIR
        / "indicadores_horizontes.csv"
    )

    df_indicadores.to_csv(
        caminho_indicadores,
        index=False
    )

    # -----------------------------------------------------
    # 5. CONFIGURAÇÃO DO EXPERIMENTO
    # -----------------------------------------------------

    config = {
        "dataset": "FD001",

        "fonte": (
            "erro de reconstrucao "
            "do LSTM Autoencoder"
        ),

        "metodo": (
            "regressao linear "
            "erro_reconstrucao x ciclo"
        ),

        "horizontes_testados": (
            HORIZONTES
        ),

        "criterios_avaliados": [
            "slope_medio",
            "slope_mediano",
            "taxa_tendencia_positiva"
        ],

        "observacao": (
            "O horizonte definitivo ainda nao foi "
            "selecionado. A escolha sera feita apos "
            "analise dos resultados."
        )
    }

    with open(
        ARQUIVO_CONFIG,
        "w",
        encoding="utf-8"
    ) as arquivo:

        json.dump(
            config,
            arquivo,
            indent=4,
            ensure_ascii=False
        )

    # -----------------------------------------------------
    # 6. RESULTADO FINAL
    # -----------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "COMPARAÇÃO ENTRE HORIZONTES"
    )

    print(
        "=" * 70
    )

    colunas_comparacao = [
        "horizonte",
        "diferenca_slope_mediano_final_inicio",
        "diferenca_taxa_positiva_final_inicio"
    ]

    colunas_existentes = [
        coluna
        for coluna in colunas_comparacao
        if coluna in df_indicadores.columns
    ]

    print(
        "\n"
        + df_indicadores[
            colunas_existentes
        ]
        .round(8)
        .to_string(
            index=False
        )
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "ARQUIVOS SALVOS"
    )

    print(
        "=" * 70
    )

    print(
        f"\nComparação:\n"
        f"{ARQUIVO_COMPARACAO}"
    )

    print(
        f"\nIndicadores:\n"
        f"{caminho_indicadores}"
    )

    print(
        f"\nConfiguração:\n"
        f"{ARQUIVO_CONFIG}"
    )