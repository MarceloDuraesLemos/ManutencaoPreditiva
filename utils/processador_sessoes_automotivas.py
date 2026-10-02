from __future__ import annotations

from typing import Iterable

import pandas as pd

from utils.adaptador_csv_automotivo import (
    ler_csv_automotivo,
    resumir_compatibilidade,
)

from utils.pipeline_automotivo import (
    PipelineAutomotivo,
)


def processar_sessoes_csv(
    arquivos: Iterable,
    pipeline: PipelineAutomotivo,
) -> dict:
    """
    Processa múltiplos CSVs automotivos.

    Regra:
        1 arquivo = 1 sessão independente.

    Os arquivos NÃO são concatenados antes da inferência.
    """

    sessoes_validas = []
    sessoes_rejeitadas = []

    for indice, arquivo in enumerate(
        arquivos,
        start=1,
    ):

        nome_arquivo = getattr(
            arquivo,
            "name",
            f"sessao_{indice}.csv",
        )

        # O Streamlit reutiliza objetos UploadedFile.
        # Garantimos que a leitura comece do início.
        if hasattr(arquivo, "seek"):
            arquivo.seek(0)

        resultado_csv = ler_csv_automotivo(
            arquivo
        )

        # ----------------------------------------------------
        # Arquivo nem sequer pôde ser lido
        # ----------------------------------------------------

        if not resultado_csv.get(
            "sucesso",
            False,
        ):
            sessoes_rejeitadas.append(
                {
                    "arquivo": nome_arquivo,
                    "motivo": resumir_compatibilidade(
                        resultado_csv
                    ),
                }
            )

            continue

        # ----------------------------------------------------
        # CSV lido, mas incompatível com a IA
        # ----------------------------------------------------

        if not resultado_csv[
            "ia_compativel"
        ]:
            sessoes_rejeitadas.append(
                {
                    "arquivo": nome_arquivo,
                    "motivo": resumir_compatibilidade(
                        resultado_csv
                    ),
                }
            )

            continue

        dados = resultado_csv[
            "dados"
        ]

        # ----------------------------------------------------
        # Executar sessão individualmente
        # ----------------------------------------------------

        try:
            resultado_ia = pipeline.analisar(
                dados,
                origem_dados="CSV",
            )

        except Exception as erro:
            sessoes_rejeitadas.append(
                {
                    "arquivo": nome_arquivo,
                    "motivo": (
                        "Erro durante a execução "
                        f"da IA: {erro}"
                    ),
                }
            )

            continue

        if (
            resultado_ia.get(
                "status_pipeline"
            )
            != "OK"
        ):
            sessoes_rejeitadas.append(
                {
                    "arquivo": nome_arquivo,
                    "motivo": (
                        "O pipeline não conseguiu "
                        "concluir a análise."
                    ),
                }
            )

            continue

        # ----------------------------------------------------
        # Vehicle ID
        # ----------------------------------------------------

        vehicle_id = None

        if (
            "vehicle_id" in dados.columns
            and dados["vehicle_id"].notna().any()
        ):
            valores = (
                dados["vehicle_id"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            if len(valores) == 1:
                vehicle_id = valores[0]

            elif len(valores) > 1:
                vehicle_id = "MULTIPLOS_IDS"

        # ----------------------------------------------------
        # Guardar resultado completo
        # ----------------------------------------------------

        sessoes_validas.append(
            {
                "arquivo": nome_arquivo,
                "vehicle_id": vehicle_id,
                "dados": dados,
                "resultado": resultado_ia,
            }
        )

    # ========================================================
    # RESUMO
    # ========================================================

    linhas_resumo = []

    for sessao in sessoes_validas:

        resultado = sessao[
            "resultado"
        ]

        linhas_resumo.append(
            {
                "arquivo": sessao[
                    "arquivo"
                ],
                "vehicle_id": (
                    sessao[
                        "vehicle_id"
                    ]
                    or "NÃO INFORMADO"
                ),
                "amostras": resultado[
                    "amostras"
                ],
                "janelas": resultado[
                    "janelas"
                ],
                "automotive_health": round(
                    float(
                        resultado[
                            "automotive_health"
                        ]
                    ),
                    2,
                ),
                "status": resultado[
                    "status"
                ],
                "anomaly_score": round(
                    float(
                        resultado[
                            "anomaly_score_atual"
                        ]
                    ),
                    2,
                ),
                "persistencia_recente_pct": round(
                    float(
                        resultado[
                            "taxa_anomalias_recente_pct"
                        ]
                    ),
                    2,
                ),
            }
        )

    resumo = pd.DataFrame(
        linhas_resumo
    )

    return {
        "total_arquivos": (
            len(sessoes_validas)
            + len(sessoes_rejeitadas)
        ),
        "total_validos": len(
            sessoes_validas
        ),
        "total_rejeitados": len(
            sessoes_rejeitadas
        ),
        "sessoes_validas": sessoes_validas,
        "sessoes_rejeitadas": sessoes_rejeitadas,
        "resumo": resumo,
    }