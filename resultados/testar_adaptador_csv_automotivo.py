from pathlib import Path

import pandas as pd

from utils.adaptador_csv_automotivo import (
    ler_csv_automotivo,
    resumir_compatibilidade,
)


CAMINHO_VALIDO = Path(
    "datasets/automotivo/simulados/telemetria_normal.csv"
)

PASTA_TEMP = Path(
    "resultados/automotivo/testes_csv"
)

PASTA_TEMP.mkdir(
    parents=True,
    exist_ok=True,
)


def imprimir_resultado(
    nome_teste: str,
    resultado: dict,
):

    print()
    print("-" * 70)
    print(nome_teste)
    print("-" * 70)

    print(
        "Sucesso na leitura:",
        resultado["sucesso"],
    )

    print(
        "Resumo:",
        resumir_compatibilidade(
            resultado
        ),
    )

    if not resultado["sucesso"]:
        return

    print(
        "Linhas:",
        resultado["linhas"],
    )

    print(
        "Features ausentes:",
        resultado[
            "features_ia_ausentes"
        ],
    )

    print(
        "Não numéricas:",
        resultado[
            "colunas_nao_numericas"
        ],
    )

    print(
        "Valores ausentes:",
        resultado[
            "valores_ausentes"
        ],
    )

    print(
        "Amostras suficientes:",
        resultado[
            "amostras_suficientes"
        ],
    )

    print(
        "IA compatível:",
        resultado[
            "ia_compativel"
        ],
    )


def main():

    print()
    print("=" * 70)
    print("TESTES DE VALIDAÇÃO DO CSV AUTOMOTIVO")
    print("=" * 70)

    # ========================================================
    # BASE
    # ========================================================

    df_original = pd.read_csv(
        CAMINHO_VALIDO
    )

    # ========================================================
    # TESTE 1 — CSV VÁLIDO
    # ========================================================

    resultado = ler_csv_automotivo(
        CAMINHO_VALIDO
    )

    imprimir_resultado(
        "TESTE 1 — CSV VÁLIDO",
        resultado,
    )

    # ========================================================
    # TESTE 2 — FEATURE OBRIGATÓRIA AUSENTE
    # ========================================================

    df = df_original.drop(
        columns=["rpm"]
    )

    caminho = (
        PASTA_TEMP
        / "teste_sem_rpm.csv"
    )

    df.to_csv(
        caminho,
        index=False,
    )

    resultado = ler_csv_automotivo(
        caminho
    )

    imprimir_resultado(
        "TESTE 2 — SEM RPM",
        resultado,
    )

    # ========================================================
    # TESTE 3 — VALOR NÃO NUMÉRICO
    # ========================================================

    df = df_original.copy()

    # Converte propositalmente a coluna para object
    # para conseguirmos criar um CSV inválido de teste.
    df["coolant_temp_c"] = (
        df["coolant_temp_c"].astype(object)
    )

    df.loc[
        10,
        "coolant_temp_c",
    ] = "ERRO"

    caminho = (
        PASTA_TEMP
        / "teste_nao_numerico.csv"
    )

    df.to_csv(
        caminho,
        index=False,
    )

    resultado = ler_csv_automotivo(
        caminho
    )

    imprimir_resultado(
        "TESTE 3 — VALOR NÃO NUMÉRICO",
        resultado,
    )

    # ========================================================
    # TESTE 4 — VALOR AUSENTE
    # ========================================================

    df = df_original.copy()

    df.loc[
        20,
        "battery_voltage",
    ] = None

    caminho = (
        PASTA_TEMP
        / "teste_valor_ausente.csv"
    )

    df.to_csv(
        caminho,
        index=False,
    )

    resultado = ler_csv_automotivo(
        caminho
    )

    imprimir_resultado(
        "TESTE 4 — VALOR AUSENTE",
        resultado,
    )

    # ========================================================
    # TESTE 5 — MENOS DE 30 AMOSTRAS
    # ========================================================

    df = (
        df_original
        .head(20)
        .copy()
    )

    caminho = (
        PASTA_TEMP
        / "teste_poucas_amostras.csv"
    )

    df.to_csv(
        caminho,
        index=False,
    )

    resultado = ler_csv_automotivo(
        caminho
    )

    imprimir_resultado(
        "TESTE 5 — SOMENTE 20 AMOSTRAS",
        resultado,
    )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 70)
    print("TESTES FINALIZADOS")
    print("=" * 70)


if __name__ == "__main__":
    main()