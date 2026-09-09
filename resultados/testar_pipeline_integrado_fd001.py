from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from utils.pipeline_fd001 import (
    PipelineFD001,
    carregar_cmapss_fd001,
)


ROOT = Path(__file__).resolve().parents[1]

TEST_PATH = ROOT / "datasets" / "CMAPSSData" / "test_FD001.txt"
RUL_PATH = ROOT / "datasets" / "CMAPSSData" / "RUL_FD001.txt"

OUT_DIR = ROOT / "resultados" / "fd001_pipeline_integrado"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def carregar_rul_real(caminho):
    df = pd.read_csv(
        caminho,
        sep=r"\s+",
        header=None
    )

    df = df.iloc[:, [0]].copy()
    df.columns = ["rul_real"]

    df["unit_number"] = np.arange(
        1,
        len(df) + 1
    )

    return df[
        ["unit_number", "rul_real"]
    ]


def main():
    print("=" * 70)
    print("TESTE INTEGRADO DO PIPELINE FD001")
    print("=" * 70)

    print("\nCarregando dataset de teste...")
    df_test = carregar_cmapss_fd001(
        TEST_PATH
    )

    print("Carregando RUL real...")
    df_rul_real = carregar_rul_real(
        RUL_PATH
    )

    print("Carregando pipeline...")
    pipeline = PipelineFD001()

    motores = sorted(
        df_test["unit_number"].unique()
    )

    print(
        f"\nQuantidade de motores: {len(motores)}"
    )

    resultados = []

    for i, unit in enumerate(
        motores,
        start=1
    ):
        print(
            f"Processando motor {unit} "
            f"({i}/{len(motores)})..."
        )

        resultado = pipeline.analisar_motor(
            df_test,
            int(unit)
        )

        resultados.append(
            resultado
        )

    df_resultados = pd.DataFrame(
        resultados
    )

    df_resultados = df_resultados.merge(
        df_rul_real,
        on="unit_number",
        how="left"
    )

    # ==========================================================
    # ERRO RUL
    # ==========================================================

    df_resultados["erro_rul"] = (
        df_resultados["rul_estimado"]
        - df_resultados["rul_real"]
    )

    df_resultados["erro_abs_rul"] = (
        df_resultados["erro_rul"].abs()
    )

    # ==========================================================
    # MÉTRICAS RUL
    # ==========================================================

    validos_rul = df_resultados[
        df_resultados["rul_estimado"]
        .notna()
    ].copy()

    mae = mean_absolute_error(
        validos_rul["rul_real"],
        validos_rul["rul_estimado"]
    )

    rmse = np.sqrt(
        mean_squared_error(
            validos_rul["rul_real"],
            validos_rul["rul_estimado"]
        )
    )

    metricas_rul = {
        "dataset": "FD001",
        "quantidade_motores":
            int(len(validos_rul)),
        "mae":
            float(mae),
        "rmse":
            float(rmse),
        "erro_medio_assinado":
            float(
                validos_rul["erro_rul"].mean()
            ),
        "erro_mediano_absoluto":
            float(
                validos_rul[
                    "erro_abs_rul"
                ].median()
            ),
        "observacao":
            (
                "RUL previsto pelo pipeline integrado "
                "comparado ao RUL real oficial do "
                "test_FD001."
            )
    }

    # ==========================================================
    # RESUMO PIPELINE
    # ==========================================================

    resumo_pipeline = (
        df_resultados[
            "status_pipeline"
        ]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            "status_pipeline"
        )
        .reset_index(
            name="quantidade"
        )
    )

    resumo_pipeline[
        "percentual"
    ] = (
        resumo_pipeline[
            "quantidade"
        ]
        / len(df_resultados)
        * 100
    )

    # ==========================================================
    # RESUMO HEALTH STATUS
    # ==========================================================

    completos = df_resultados[
        df_resultados[
            "status_pipeline"
        ]
        == "ANALISE_COMPLETA"
    ].copy()

    if not completos.empty:
        resumo_status = (
            completos[
                "health_status"
            ]
            .value_counts(
                dropna=False
            )
            .rename_axis(
                "health_status"
            )
            .reset_index(
                name="quantidade"
            )
        )

        resumo_status[
            "percentual"
        ] = (
            resumo_status[
                "quantidade"
            ]
            / len(completos)
            * 100
        )

        resumo_prioridades = (
            completos[
                "priority_class"
            ]
            .value_counts(
                dropna=False
            )
            .rename_axis(
                "priority_class"
            )
            .reset_index(
                name="quantidade"
            )
        )

        resumo_prioridades[
            "percentual"
        ] = (
            resumo_prioridades[
                "quantidade"
            ]
            / len(completos)
            * 100
        )

    else:
        resumo_status = pd.DataFrame(
            columns=[
                "health_status",
                "quantidade",
                "percentual"
            ]
        )

        resumo_prioridades = pd.DataFrame(
            columns=[
                "priority_class",
                "quantidade",
                "percentual"
            ]
        )

    # ==========================================================
    # RANKING DE URGÊNCIA
    # ==========================================================

    ranking = completos.sort_values(
        by=[
            "priority_score",
            "health_score",
            "rul_estimado"
        ],
        ascending=[
            False,
            True,
            True
        ]
    ).reset_index(drop=True)

    ranking.insert(
        0,
        "ranking_prioridade",
        np.arange(
            1,
            len(ranking) + 1
        )
    )

    # ==========================================================
    # SALVAR
    # ==========================================================

    resultados_path = (
        OUT_DIR
        / "resultados_100_motores.csv"
    )

    metricas_path = (
        OUT_DIR
        / "metricas_rul_integrado.json"
    )

    resumo_pipeline_path = (
        OUT_DIR
        / "resumo_pipeline.csv"
    )

    resumo_status_path = (
        OUT_DIR
        / "resumo_status.csv"
    )

    resumo_prioridades_path = (
        OUT_DIR
        / "resumo_prioridades.csv"
    )

    ranking_path = (
        OUT_DIR
        / "ranking_prioridade.csv"
    )

    df_resultados.to_csv(
        resultados_path,
        index=False
    )

    resumo_pipeline.to_csv(
        resumo_pipeline_path,
        index=False
    )

    resumo_status.to_csv(
        resumo_status_path,
        index=False
    )

    resumo_prioridades.to_csv(
        resumo_prioridades_path,
        index=False
    )

    ranking.to_csv(
        ranking_path,
        index=False
    )

    with open(
        metricas_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metricas_rul,
            f,
            indent=4,
            ensure_ascii=False
        )

    # ==========================================================
    # SAÍDA TERMINAL
    # ==========================================================

    print("\n" + "=" * 70)
    print("MÉTRICAS RUL")
    print("=" * 70)

    print(
        f"MAE:  {mae:.4f}"
    )

    print(
        f"RMSE: {rmse:.4f}"
    )

    print(
        "Erro médio assinado: "
        f"{metricas_rul['erro_medio_assinado']:.4f}"
    )

    print("\n" + "=" * 70)
    print("STATUS DO PIPELINE")
    print("=" * 70)

    print(
        resumo_pipeline.to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("HEALTH STATUS")
    print("=" * 70)

    print(
        resumo_status.to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("PRIORIDADES")
    print("=" * 70)

    print(
        resumo_prioridades.to_string(
            index=False
        )
    )

    if not ranking.empty:
        print("\n" + "=" * 70)
        print("TOP 10 MOTORES MAIS URGENTES")
        print("=" * 70)

        colunas_top = [
            "ranking_prioridade",
            "unit_number",
            "rul_estimado",
            "rul_real",
            "anomaly_score",
            "trend_score",
            "health_score",
            "health_status",
            "priority_score",
            "priority_class"
        ]

        print(
            ranking[
                colunas_top
            ]
            .head(10)
            .to_string(
                index=False
            )
        )

    print("\nArquivos gerados em:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()