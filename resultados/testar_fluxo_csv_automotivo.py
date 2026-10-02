from pathlib import Path

from utils.adaptador_csv_automotivo import (
    ler_csv_automotivo,
    resumir_compatibilidade,
)

from utils.pipeline_automotivo import (
    PipelineAutomotivo,
)


CAMINHO_VALIDO = Path(
    "datasets/automotivo/simulados/telemetria_normal.csv"
)

CAMINHO_INVALIDO = Path(
    "resultados/automotivo/testes_csv/teste_sem_rpm.csv"
)


def processar_csv(
    caminho: Path,
    pipeline: PipelineAutomotivo,
):

    print()
    print("=" * 72)
    print(f"ARQUIVO: {caminho}")
    print("=" * 72)

    # ========================================================
    # 1. ADAPTADOR CSV
    # ========================================================

    resultado_csv = ler_csv_automotivo(
        caminho
    )

    print()
    print(
        "Leitura:",
        resultado_csv["sucesso"],
    )

    print(
        "Compatibilidade:",
        resumir_compatibilidade(
            resultado_csv
        ),
    )

    # --------------------------------------------------------
    # Falha na leitura
    # --------------------------------------------------------

    if not resultado_csv["sucesso"]:

        print()
        print(
            "FLUXO INTERROMPIDO:"
            " não foi possível processar o arquivo."
        )

        return

    # --------------------------------------------------------
    # CSV lido, mas incompatível com a IA
    # --------------------------------------------------------

    if not resultado_csv["ia_compativel"]:

        print()
        print(
            "FLUXO INTERROMPIDO ANTES DA IA."
        )

        print(
            "Nenhuma inferência foi executada."
        )

        return

    # ========================================================
    # 2. DADOS PADRONIZADOS
    # ========================================================

    dados = resultado_csv[
        "dados"
    ]

    print()
    print(
        f"Dados liberados para IA: "
        f"{len(dados)} amostras."
    )

    # ========================================================
    # 3. PIPELINE AUTOMOTIVO
    # ========================================================

    resultado_ia = pipeline.analisar(
        dados,
        origem_dados="CSV",
    )

    print()
    print("-" * 72)
    print("RESULTADO DA IA")
    print("-" * 72)

    print(
        "Status pipeline:",
        resultado_ia[
            "status_pipeline"
        ],
    )

    # --------------------------------------------------------
    # Pipeline não concluiu
    # --------------------------------------------------------

    if (
        resultado_ia[
            "status_pipeline"
        ]
        != "OK"
    ):

        print()
        print(
            "Pipeline não conseguiu "
            "concluir a análise."
        )

        return

    # ========================================================
    # 4. RESULTADOS
    # ========================================================

    print(
        "Amostras:",
        resultado_ia[
            "amostras"
        ],
    )

    print(
        "Janelas analisadas:",
        resultado_ia[
            "janelas"
        ],
    )

    print(
        "Window size:",
        resultado_ia[
            "window_size"
        ],
    )

    print(
        "Erro reconstrução atual:",
        f"{resultado_ia['reconstruction_error_atual']:.6f}",
    )

    print(
        "Threshold:",
        f"{resultado_ia['threshold']:.6f}",
    )

    print(
        "Anomaly Score atual:",
        f"{resultado_ia['anomaly_score_atual']:.2f}",
    )

    print(
        "Anomaly Score médio recente:",
        f"{resultado_ia['anomaly_score_medio_recente']:.2f}",
    )

    print(
        "Anomalia atual:",
        resultado_ia[
            "anomalia_atual"
        ],
    )

    print(
        "Taxa anomalias total:",
        f"{resultado_ia['taxa_anomalias_total_pct']:.2f}%",
    )

    print(
        "Taxa anomalias recente:",
        f"{resultado_ia['taxa_anomalias_recente_pct']:.2f}%",
    )

    print(
        "Automotive Health:",
        f"{resultado_ia['automotive_health']:.2f}",
    )

    print(
        "Status:",
        resultado_ia[
            "status"
        ],
    )

    print(
        "Origem dos dados:",
        resultado_ia[
            "origem_dados"
        ],
    )

    print(
        "Modelo de referência:",
        resultado_ia[
            "modelo_referencia"
        ],
    )

    print()
    print(
        "Recomendação:"
    )

    print(
        resultado_ia[
            "recomendacao"
        ]
    )

    print()
    print(
        "FLUXO COMPLETO EXECUTADO COM SUCESSO."
    )


def main():

    print()
    print("#" * 72)
    print("TESTE PONTA A PONTA — CSV AUTOMOTIVO")
    print("#" * 72)

    # ========================================================
    # CARREGAR PIPELINE UMA ÚNICA VEZ
    # ========================================================

    pipeline = PipelineAutomotivo()

    # ========================================================
    # TESTE 1 — CSV VÁLIDO
    # ========================================================

    print()
    print(
        "TESTE 1 — CSV VÁLIDO"
    )

    processar_csv(
        CAMINHO_VALIDO,
        pipeline,
    )

    # ========================================================
    # TESTE 2 — CSV INCOMPATÍVEL
    # ========================================================

    print()
    print(
        "TESTE 2 — CSV INCOMPATÍVEL"
    )

    processar_csv(
        CAMINHO_INVALIDO,
        pipeline,
    )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("#" * 72)
    print(
        "TESTE PONTA A PONTA FINALIZADO"
    )
    print("#" * 72)


if __name__ == "__main__":
    main()