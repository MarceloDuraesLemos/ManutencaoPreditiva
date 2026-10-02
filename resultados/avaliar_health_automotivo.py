from __future__ import annotations

import os
from pathlib import Path

# Compatibilidade com o ambiente atual do projeto.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import pandas as pd

from utils.pipeline_automotivo import PipelineAutomotivo


# ============================================================
# CAMINHOS
# ============================================================

CAMINHO_DATASET = Path(
    "datasets/automotivo/teste/telemetria_teste_cenarios.csv"
)

PASTA_SAIDA = Path(
    "resultados/automotivo"
)

CAMINHO_SESSOES = (
    PASTA_SAIDA
    / "avaliacao_health_por_sessao.csv"
)

CAMINHO_RESUMO = (
    PASTA_SAIDA
    / "resumo_health_por_cenario.csv"
)

CAMINHO_STATUS = (
    PASTA_SAIDA
    / "distribuicao_status_por_cenario.csv"
)

PASTA_SAIDA.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# FUNÇÃO AUXILIAR
# ============================================================

def calcular_resumo(
    df_resultados: pd.DataFrame,
) -> pd.DataFrame:

    resumo = (
        df_resultados
        .groupby("scenario")
        .agg(
            sessoes=(
                "session_id",
                "count",
            ),
            health_medio=(
                "automotive_health",
                "mean",
            ),
            health_mediano=(
                "automotive_health",
                "median",
            ),
            health_min=(
                "automotive_health",
                "min",
            ),
            health_max=(
                "automotive_health",
                "max",
            ),
            score_medio_recente_medio=(
                "anomaly_score_medio_recente",
                "mean",
            ),
            persistencia_media=(
                "taxa_anomalias_recente_pct",
                "mean",
            ),
            taxa_anomalias_total_media=(
                "taxa_anomalias_total_pct",
                "mean",
            ),
        )
        .reset_index()
    )

    return resumo


# ============================================================
# EXECUÇÃO
# ============================================================

def main():

    print()
    print("=" * 78)
    print("AVALIAÇÃO FINAL — AUTOMOTIVE HEALTH v1")
    print("=" * 78)

    if not CAMINHO_DATASET.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado: {CAMINHO_DATASET}"
        )

    dados = pd.read_csv(
        CAMINHO_DATASET
    )

    colunas_obrigatorias = [
        "session_id",
        "scenario",
    ]

    ausentes = [
        coluna
        for coluna in colunas_obrigatorias
        if coluna not in dados.columns
    ]

    if ausentes:
        raise ValueError(
            "Colunas obrigatórias ausentes no dataset: "
            + ", ".join(ausentes)
        )

    pipeline = PipelineAutomotivo()

    sessoes = (
        dados[
            [
                "session_id",
                "scenario",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "scenario",
                "session_id",
            ]
        )
        .reset_index(drop=True)
    )

    print()
    print(
        f"Sessões encontradas: {len(sessoes)}"
    )

    print()

    resultados = []

    for numero, linha in sessoes.iterrows():

        session_id = linha["session_id"]
        scenario = linha["scenario"]

        dados_sessao = (
            dados[
                dados["session_id"]
                == session_id
            ]
            .copy()
            .reset_index(drop=True)
        )

        resultado = pipeline.analisar(
            dados_sessao,
            origem_dados="SIMULADO",
        )

        status_pipeline = resultado[
            "status_pipeline"
        ]

        if status_pipeline != "OK":

            print(
                f"[{numero + 1:02d}/{len(sessoes)}] "
                f"{session_id} | "
                f"{scenario} | "
                f"ERRO: {status_pipeline}"
            )

            resultados.append(
                {
                    "session_id": session_id,
                    "scenario": scenario,
                    "status_pipeline": status_pipeline,
                    "automotive_health": np.nan,
                    "status": "INDISPONIVEL",
                    "anomaly_score_atual": np.nan,
                    "anomaly_score_medio_recente": np.nan,
                    "taxa_anomalias_recente_pct": np.nan,
                    "taxa_anomalias_total_pct": np.nan,
                }
            )

            continue

        resultados.append(
            {
                "session_id": session_id,
                "scenario": scenario,
                "status_pipeline": status_pipeline,
                "automotive_health": resultado[
                    "automotive_health"
                ],
                "status": resultado[
                    "status"
                ],
                "anomaly_score_atual": resultado[
                    "anomaly_score_atual"
                ],
                "anomaly_score_medio_recente": resultado[
                    "anomaly_score_medio_recente"
                ],
                "taxa_anomalias_recente_pct": resultado[
                    "taxa_anomalias_recente_pct"
                ],
                "taxa_anomalias_total_pct": resultado[
                    "taxa_anomalias_total_pct"
                ],
            }
        )

        print(
            f"[{numero + 1:02d}/{len(sessoes)}] "
            f"{session_id} | "
            f"{scenario} | "
            f"Health={resultado['automotive_health']:.2f} | "
            f"{resultado['status']}"
        )

    # ========================================================
    # DATAFRAME FINAL
    # ========================================================

    df_resultados = pd.DataFrame(
        resultados
    )

    df_resultados.to_csv(
        CAMINHO_SESSOES,
        index=False,
    )

    validos = (
        df_resultados[
            df_resultados[
                "status_pipeline"
            ]
            == "OK"
        ]
        .copy()
    )

    if validos.empty:
        raise RuntimeError(
            "Nenhuma sessão foi processada com sucesso."
        )

    # ========================================================
    # RESUMO POR CENÁRIO
    # ========================================================

    resumo = calcular_resumo(
        validos
    )

    resumo.to_csv(
        CAMINHO_RESUMO,
        index=False,
    )

    # ========================================================
    # DISTRIBUIÇÃO DOS STATUS
    # ========================================================

    distribuicao_status = (
        validos
        .groupby(
            [
                "scenario",
                "status",
            ]
        )
        .size()
        .reset_index(
            name="quantidade"
        )
    )

    distribuicao_status[
        "percentual"
    ] = (
        distribuicao_status[
            "quantidade"
        ]
        / distribuicao_status.groupby(
            "scenario"
        )[
            "quantidade"
        ].transform(
            "sum"
        )
        * 100
    )

    distribuicao_status.to_csv(
        CAMINHO_STATUS,
        index=False,
    )

    # ========================================================
    # IMPRESSÃO
    # ========================================================

    print()
    print("=" * 78)
    print("RESUMO POR CENÁRIO")
    print("=" * 78)
    print()

    print(
        resumo.to_string(
            index=False,
            formatters={
                "health_medio": lambda x: f"{x:.2f}",
                "health_mediano": lambda x: f"{x:.2f}",
                "health_min": lambda x: f"{x:.2f}",
                "health_max": lambda x: f"{x:.2f}",
                "score_medio_recente_medio": lambda x: f"{x:.2f}",
                "persistencia_media": lambda x: f"{x:.2f}",
                "taxa_anomalias_total_media": lambda x: f"{x:.2f}",
            },
        )
    )

    print()
    print("=" * 78)
    print("DISTRIBUIÇÃO DOS STATUS")
    print("=" * 78)
    print()

    print(
        distribuicao_status.to_string(
            index=False,
            formatters={
                "percentual": lambda x: f"{x:.2f}%",
            },
        )
    )

    print()
    print("=" * 78)
    print("IMPORTANTE")
    print("=" * 78)

    print(
        "Os cenários são simulados e não constituem "
        "rótulos de falha mecânica real."
    )

    print(
        "Esta avaliação mede o comportamento do índice "
        "demonstrativo nas sessões simuladas."
    )

    print(
        "Os resultados NÃO devem ser interpretados como "
        "acurácia de diagnóstico automotivo."
    )

    print()
    print("Arquivos gerados:")
    print(CAMINHO_SESSOES)
    print(CAMINHO_RESUMO)
    print(CAMINHO_STATUS)

    print()
    print("=" * 78)
    print("AVALIAÇÃO CONCLUÍDA")
    print("=" * 78)


if __name__ == "__main__":
    main()