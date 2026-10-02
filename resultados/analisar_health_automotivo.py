from __future__ import annotations

import os

# Compatibilidade com o ambiente atual do projeto.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from pathlib import Path

import numpy as np
import pandas as pd

from utils.pipeline_automotivo import PipelineAutomotivo
from utils.simulador_automotivo import (
    ConfigSimulacao,
    gerar_telemetria,
)


# ============================================================
# CONFIGURAÇÃO
# ============================================================

PASTA_SAIDA = Path(
    "resultados/automotivo"
)

PASTA_SAIDA.mkdir(
    parents=True,
    exist_ok=True,
)

CAMINHO_SAIDA = (
    PASTA_SAIDA
    / "componentes_health_automotivo.csv"
)

JANELA_RECENTE = 60


# ============================================================
# CENÁRIOS
# ============================================================

CENARIOS = [
    "NORMAL",
    "AQUECIMENTO_PROGRESSIVO",
    "OPERACAO_IRREGULAR",
    "CONDICAO_SEVERA_SIMULADA",
]


# ============================================================
# FUNÇÃO DE ANÁLISE
# ============================================================

def analisar_cenario(
    pipeline: PipelineAutomotivo,
    cenario: str,
    seed: int,
) -> dict:

    config = ConfigSimulacao(
        vehicle_id="VEICULO_HEALTH_TESTE",
        duracao_minutos=20,
        intervalo_segundos=1,
        cenario=cenario,
        seed=seed,
    )

    dados = gerar_telemetria(
        config
    )

    resultado = pipeline.analisar(
        dados,
        origem_dados="SIMULADO",
    )

    if (
        resultado["status_pipeline"]
        != "OK"
    ):
        raise RuntimeError(
            f"Pipeline inválido para "
            f"{cenario}: {resultado}"
        )

    historico = resultado[
        "historico"
    ].copy()

    quantidade = min(
        JANELA_RECENTE,
        len(historico),
    )

    recente = historico.tail(
        quantidade
    )

    scores = (
        recente["anomaly_score"]
        .to_numpy(
            dtype=float
        )
    )

    anomalias = (
        recente["anomaly"]
        .astype(bool)
        .to_numpy()
    )

    erros = (
        recente[
            "reconstruction_error"
        ]
        .to_numpy(
            dtype=float
        )
    )

    return {
        "cenario": cenario,

        "janelas_total": int(
            len(historico)
        ),

        "janelas_recentes": int(
            quantidade
        ),

        "erro_atual": float(
            erros[-1]
        ),

        "score_atual": float(
            scores[-1]
        ),

        "score_medio_recente": float(
            np.mean(scores)
        ),

        "score_mediano_recente": float(
            np.median(scores)
        ),

        "score_p75_recente": float(
            np.percentile(
                scores,
                75,
            )
        ),

        "score_p90_recente": float(
            np.percentile(
                scores,
                90,
            )
        ),

        "score_p95_recente": float(
            np.percentile(
                scores,
                95,
            )
        ),

        "score_max_recente": float(
            np.max(scores)
        ),

        "taxa_anomalias_recente": float(
            np.mean(anomalias)
            * 100.0
        ),

        "health_atual_antigo": float(
            resultado[
                "automotive_health"
            ]
        ),
    }


# ============================================================
# EXECUÇÃO
# ============================================================

def main():

    print()
    print("=" * 78)
    print(
        "ANÁLISE DOS COMPONENTES "
        "DO AUTOMOTIVE HEALTH"
    )
    print("=" * 78)

    pipeline = PipelineAutomotivo()

    resultados = []

    for i, cenario in enumerate(
        CENARIOS
    ):

        # Mesmas seeds usadas pelo teste
        # direto do pipeline.
        seed = 20000 + i

        linha = analisar_cenario(
            pipeline,
            cenario,
            seed,
        )

        resultados.append(
            linha
        )

    df = pd.DataFrame(
        resultados
    )

    df.to_csv(
        CAMINHO_SAIDA,
        index=False,
    )

    print()
    print(
        df.to_string(
            index=False,
            formatters={
                "erro_atual": (
                    lambda x:
                    f"{x:.6f}"
                ),
                "score_atual": (
                    lambda x:
                    f"{x:.2f}"
                ),
                "score_medio_recente": (
                    lambda x:
                    f"{x:.2f}"
                ),
                "score_mediano_recente": (
                    lambda x:
                    f"{x:.2f}"
                ),
                "score_p75_recente": (
                    lambda x:
                    f"{x:.2f}"
                ),
                "score_p90_recente": (
                    lambda x:
                    f"{x:.2f}"
                ),
                "score_p95_recente": (
                    lambda x:
                    f"{x:.2f}"
                ),
                "score_max_recente": (
                    lambda x:
                    f"{x:.2f}"
                ),
                "taxa_anomalias_recente": (
                    lambda x:
                    f"{x:.2f}"
                ),
                "health_atual_antigo": (
                    lambda x:
                    f"{x:.2f}"
                ),
            },
        )
    )

    print()
    print("=" * 78)
    print("IMPORTANTE")
    print("=" * 78)

    print(
        "Este script NÃO altera o modelo, "
        "threshold, Anomaly Score ou pipeline."
    )

    print(
        "Ele apenas mede o comportamento "
        "recente para calibrarmos o Health."
    )

    print()
    print(
        "Arquivo salvo em:"
    )

    print(
        CAMINHO_SAIDA
    )

    print()
    print("=" * 78)
    print("ANÁLISE CONCLUÍDA")
    print("=" * 78)


if __name__ == "__main__":
    main()