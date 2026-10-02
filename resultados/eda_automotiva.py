from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# CAMINHOS
# ============================================================

ARQUIVO_DESENVOLVIMENTO = (
    Path("datasets")
    / "automotivo"
    / "desenvolvimento"
    / "telemetria_normal_desenvolvimento.csv"
)

ARQUIVO_TESTE = (
    Path("datasets")
    / "automotivo"
    / "teste"
    / "telemetria_teste_cenarios.csv"
)

PASTA_RESULTADOS = (
    Path("resultados")
    / "automotivo"
)

PASTA_RESULTADOS.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# VARIÁVEIS DE TELEMETRIA
# ============================================================

FEATURES = [
    "rpm",
    "speed_kmh",
    "coolant_temp_c",
    "engine_load_pct",
    "throttle_pct",
    "intake_temp_c",
    "map_kpa",
    "maf_g_s",
    "battery_voltage",
    "fuel_level_pct",
]


# ============================================================
# CARREGAMENTO
# ============================================================

def carregar_dados():

    desenvolvimento = pd.read_csv(
        ARQUIVO_DESENVOLVIMENTO
    )

    teste = pd.read_csv(
        ARQUIVO_TESTE
    )

    return desenvolvimento, teste


# ============================================================
# VERIFICAÇÃO BÁSICA
# ============================================================

def verificar_dados(
    desenvolvimento,
    teste,
):

    print()
    print("=" * 75)
    print("1. VERIFICAÇÃO DOS DADOS")
    print("=" * 75)

    print()
    print("DESENVOLVIMENTO")
    print(f"Registros: {len(desenvolvimento)}")
    print(
        "Sessões:",
        desenvolvimento["session_id"].nunique()
    )

    print()
    print("TESTE")
    print(f"Registros: {len(teste)}")
    print(
        "Sessões:",
        teste["session_id"].nunique()
    )

    print()
    print("VALORES AUSENTES - DESENVOLVIMENTO")
    print(
        desenvolvimento[FEATURES]
        .isna()
        .sum()
    )

    print()
    print("VALORES AUSENTES - TESTE")
    print(
        teste[FEATURES]
        .isna()
        .sum()
    )


# ============================================================
# ESTATÍSTICAS DO NORMAL
# ============================================================

def analisar_normal(
    desenvolvimento,
):

    print()
    print("=" * 75)
    print("2. COMPORTAMENTO NORMAL")
    print("=" * 75)

    resumo = (
        desenvolvimento[FEATURES]
        .describe()
        .T[
            [
                "mean",
                "std",
                "min",
                "25%",
                "50%",
                "75%",
                "max",
            ]
        ]
        .round(3)
    )

    print()
    print(resumo)

    caminho = (
        PASTA_RESULTADOS
        / "estatisticas_normal.csv"
    )

    resumo.to_csv(
        caminho,
        encoding="utf-8",
    )

    print()
    print(
        "Salvo em:",
        caminho,
    )


# ============================================================
# COMPARAÇÃO ENTRE CENÁRIOS
# ============================================================

def comparar_cenarios(
    teste,
):

    print()
    print("=" * 75)
    print("3. MÉDIAS POR CENÁRIO")
    print("=" * 75)

    medias = (
        teste
        .groupby("scenario")[FEATURES]
        .mean()
        .round(3)
    )

    print()
    print(medias)

    medias.to_csv(
        PASTA_RESULTADOS
        / "medias_por_cenario.csv",
        encoding="utf-8",
    )

    print()
    print("=" * 75)
    print("4. DESVIOS-PADRÃO POR CENÁRIO")
    print("=" * 75)

    desvios = (
        teste
        .groupby("scenario")[FEATURES]
        .std()
        .round(3)
    )

    print()
    print(desvios)

    desvios.to_csv(
        PASTA_RESULTADOS
        / "desvios_por_cenario.csv",
        encoding="utf-8",
    )


# ============================================================
# FASE FINAL DAS SESSÕES
# ============================================================

def analisar_fase_final(
    teste,
):

    print()
    print("=" * 75)
    print("5. FASE FINAL DAS SESSÕES")
    print("=" * 75)

    partes = []

    for session_id, sessao in teste.groupby(
        "session_id"
    ):

        sessao = sessao.reset_index(
            drop=True
        )

        inicio = int(
            len(sessao) * 0.75
        )

        final = sessao.iloc[
            inicio:
        ].copy()

        partes.append(final)

    fase_final = pd.concat(
        partes,
        ignore_index=True,
    )

    resumo = (
        fase_final
        .groupby("scenario")[FEATURES]
        .mean()
        .round(3)
    )

    print()
    print(resumo)

    resumo.to_csv(
        PASTA_RESULTADOS
        / "medias_fase_final_por_cenario.csv",
        encoding="utf-8",
    )


# ============================================================
# CORRELAÇÃO DO NORMAL
# ============================================================

def analisar_correlacoes(
    desenvolvimento,
):

    print()
    print("=" * 75)
    print("6. CORRELAÇÕES - OPERAÇÃO NORMAL")
    print("=" * 75)

    correlacao = (
        desenvolvimento[FEATURES]
        .corr()
        .round(3)
    )

    print()
    print(correlacao)

    correlacao.to_csv(
        PASTA_RESULTADOS
        / "correlacoes_normal.csv",
        encoding="utf-8",
    )


# ============================================================
# GRÁFICOS DE EXEMPLO
# ============================================================

def gerar_graficos(
    teste,
):

    print()
    print("=" * 75)
    print("7. GERANDO GRÁFICOS")
    print("=" * 75)

    variaveis_grafico = [
        "rpm",
        "coolant_temp_c",
        "engine_load_pct",
        "battery_voltage",
    ]

    for cenario in teste[
        "scenario"
    ].unique():

        dados_cenario = teste[
            teste["scenario"] == cenario
        ]

        primeira_sessao = (
            dados_cenario[
                "session_id"
            ].iloc[0]
        )

        sessao = dados_cenario[
            dados_cenario["session_id"]
            == primeira_sessao
        ].reset_index(drop=True)

        for feature in variaveis_grafico:

            plt.figure(
                figsize=(11, 4)
            )

            plt.plot(
                sessao.index,
                sessao[feature],
            )

            plt.title(
                f"{cenario} - {feature}"
            )

            plt.xlabel(
                "Amostra (1 segundo)"
            )

            plt.ylabel(feature)

            plt.tight_layout()

            nome = (
                f"{cenario.lower()}_"
                f"{feature}.png"
            )

            plt.savefig(
                PASTA_RESULTADOS / nome,
                dpi=140,
            )

            plt.close()

    print(
        "Gráficos salvos em:",
        PASTA_RESULTADOS,
    )


# ============================================================
# RESUMO DOS CENÁRIOS
# ============================================================

def resumo_interpretacao(
    teste,
):

    print()
    print("=" * 75)
    print("8. RESUMO AUTOMÁTICO DOS CENÁRIOS")
    print("=" * 75)

    for cenario, grupo in teste.groupby(
        "scenario"
    ):

        print()
        print(cenario)
        print("-" * 50)

        print(
            "RPM médio:",
            round(
                grupo["rpm"].mean(),
                2,
            )
        )

        print(
            "Temperatura média:",
            round(
                grupo[
                    "coolant_temp_c"
                ].mean(),
                2,
            )
        )

        print(
            "Temperatura máxima:",
            round(
                grupo[
                    "coolant_temp_c"
                ].max(),
                2,
            )
        )

        print(
            "Carga média:",
            round(
                grupo[
                    "engine_load_pct"
                ].mean(),
                2,
            )
        )

        print(
            "Tensão média:",
            round(
                grupo[
                    "battery_voltage"
                ].mean(),
                2,
            )
        )

        print(
            "Tensão mínima:",
            round(
                grupo[
                    "battery_voltage"
                ].min(),
                2,
            )
        )


# ============================================================
# EXECUÇÃO
# ============================================================

def main():

    desenvolvimento, teste = (
        carregar_dados()
    )

    verificar_dados(
        desenvolvimento,
        teste,
    )

    analisar_normal(
        desenvolvimento,
    )

    comparar_cenarios(
        teste,
    )

    analisar_fase_final(
        teste,
    )

    analisar_correlacoes(
        desenvolvimento,
    )

    gerar_graficos(
        teste,
    )

    resumo_interpretacao(
        teste,
    )

    print()
    print("=" * 75)
    print("EDA AUTOMOTIVA CONCLUÍDA")
    print("=" * 75)


if __name__ == "__main__":
    main()