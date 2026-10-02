from __future__ import annotations

import json
import os
from pathlib import Path

# Precisa ser definido antes de importar TensorFlow.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


# ============================================================
# CONFIGURAÇÃO
# ============================================================

SEED = 42
VALIDATION_FRACTION = 0.20

CAMINHO_DADOS = Path(
    "datasets/automotivo/desenvolvimento/"
    "telemetria_normal_desenvolvimento.csv"
)

PASTA_MODELO = Path(
    "modelo/automotivo_anomalia"
)

CAMINHO_MODELO = (
    PASTA_MODELO
    / "modelo_autoencoder.keras"
)

CAMINHO_SCALER = (
    PASTA_MODELO
    / "scaler.pkl"
)

CAMINHO_CONFIG = (
    PASTA_MODELO
    / "config.json"
)

PASTA_RESULTADOS = Path(
    "resultados/automotivo"
)

CAMINHO_SAIDA = (
    PASTA_RESULTADOS
    / "calibracao_anomaly_score_normal.csv"
)

CAMINHO_RESUMO = (
    PASTA_RESULTADOS
    / "resumo_calibracao_anomaly_score.json"
)


# ============================================================
# FUNÇÕES
# ============================================================

def criar_janelas_por_sessao(
    df: pd.DataFrame,
    features: list[str],
    scaler,
    window_size: int,
):
    """
    Cria janelas sem misturar dados entre sessões.
    """

    janelas = []
    metadados = []

    for session_id, grupo in df.groupby(
        "session_id",
        sort=False,
    ):
        grupo = grupo.reset_index(drop=True)

        valores = scaler.transform(
            grupo[features]
        )

        quantidade = (
            len(grupo)
            - window_size
            + 1
        )

        if quantidade <= 0:
            continue

        for inicio in range(quantidade):
            fim = inicio + window_size

            janelas.append(
                valores[inicio:fim]
            )

            metadados.append(
                {
                    "session_id": session_id,
                    "indice_inicio": inicio,
                    "indice_final": fim - 1,
                }
            )

    X = np.asarray(
        janelas,
        dtype=np.float32,
    )

    meta = pd.DataFrame(
        metadados
    )

    return X, meta


def calcular_reconstruction_error(
    modelo,
    X: np.ndarray,
) -> np.ndarray:

    reconstruido = modelo.predict(
        X,
        batch_size=256,
        verbose=0,
    )

    erros = np.mean(
        np.square(
            X - reconstruido
        ),
        axis=(1, 2),
    )

    return erros


# ============================================================
# EXECUÇÃO
# ============================================================

def main():

    print()
    print("=" * 72)
    print("CALIBRAÇÃO DO ANOMALY SCORE AUTOMOTIVO")
    print("=" * 72)

    # --------------------------------------------------------
    # Arquivos
    # --------------------------------------------------------

    arquivos = [
        CAMINHO_DADOS,
        CAMINHO_MODELO,
        CAMINHO_SCALER,
        CAMINHO_CONFIG,
    ]

    for caminho in arquivos:
        if not caminho.exists():
            raise FileNotFoundError(
                f"Arquivo não encontrado: {caminho}"
            )

    PASTA_RESULTADOS.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Configuração original
    # --------------------------------------------------------

    with open(
        CAMINHO_CONFIG,
        "r",
        encoding="utf-8",
    ) as arquivo:
        config = json.load(arquivo)

    features = config["features"]
    window_size = int(
        config["window_size"]
    )

    threshold_original = float(
        config["threshold"]
    )

    mediana_original = float(
        config["median_normal_validation"]
    )

    p99_original = float(
        config["p99_normal_validation"]
    )

    print()
    print("Configuração carregada:")
    print(
        f"  Window size: {window_size}"
    )
    print(
        f"  Features: {len(features)}"
    )
    print(
        f"  Threshold original: "
        f"{threshold_original:.8f}"
    )

    # --------------------------------------------------------
    # Dados NORMAL
    # --------------------------------------------------------

    print()
    print("Carregando dados NORMAL...")

    df = pd.read_csv(
        CAMINHO_DADOS
    )

    colunas_necessarias = (
        ["session_id"]
        + features
    )

    ausentes = [
        coluna
        for coluna in colunas_necessarias
        if coluna not in df.columns
    ]

    if ausentes:
        raise ValueError(
            "Colunas ausentes no dataset: "
            + ", ".join(ausentes)
        )

    # --------------------------------------------------------
    # Reconstruir exatamente o split original
    # --------------------------------------------------------

    sessoes = np.array(
        sorted(
            df["session_id"]
            .unique()
        )
    )

    rng = np.random.default_rng(
        SEED
    )

    rng.shuffle(sessoes)

    quantidade_validacao = int(
        len(sessoes)
        * VALIDATION_FRACTION
    )

    sessoes_validacao = (
        sessoes[
            :quantidade_validacao
        ]
    )

    sessoes_treino = (
        sessoes[
            quantidade_validacao:
        ]
    )

    df_validacao = (
        df[
            df["session_id"].isin(
                sessoes_validacao
            )
        ]
        .copy()
    )

    print()
    print(
        f"Sessões totais: "
        f"{len(sessoes)}"
    )
    print(
        f"Sessões treino: "
        f"{len(sessoes_treino)}"
    )
    print(
        f"Sessões validação: "
        f"{len(sessoes_validacao)}"
    )
    print(
        f"Linhas validação: "
        f"{len(df_validacao)}"
    )

    # --------------------------------------------------------
    # Carregar scaler e modelo existentes
    # --------------------------------------------------------

    print()
    print(
        "Carregando scaler e Autoencoder "
        "já treinados..."
    )

    scaler = joblib.load(
        CAMINHO_SCALER
    )

    modelo = (
        tf.keras.models.load_model(
            CAMINHO_MODELO,
            compile=False,
        )
    )

    # --------------------------------------------------------
    # Janelas de validação
    # --------------------------------------------------------

    print()
    print(
        "Criando janelas da validação NORMAL..."
    )

    X_validacao, meta = (
        criar_janelas_por_sessao(
            df=df_validacao,
            features=features,
            scaler=scaler,
            window_size=window_size,
        )
    )

    print(
        f"Janelas NORMAL de validação: "
        f"{len(X_validacao)}"
    )

    # --------------------------------------------------------
    # Reconstruction error
    # --------------------------------------------------------

    print()
    print(
        "Calculando reconstruction errors..."
    )

    erros = (
        calcular_reconstruction_error(
            modelo,
            X_validacao,
        )
    )

    meta[
        "reconstruction_error"
    ] = erros

    meta[
        "acima_threshold"
    ] = (
        erros
        > threshold_original
    )

    # --------------------------------------------------------
    # Percentis
    # --------------------------------------------------------

    percentis_desejados = [
        50,
        75,
        90,
        95,
        97,
        98,
        99,
        99.5,
        99.9,
        100,
    ]

    percentis = {}

    for p in percentis_desejados:
        valor = float(
            np.percentile(
                erros,
                p,
            )
        )

        percentis[str(p)] = valor

    media = float(
        np.mean(erros)
    )

    desvio = float(
        np.std(erros)
    )

    minimo = float(
        np.min(erros)
    )

    maximo = float(
        np.max(erros)
    )

    taxa_threshold = float(
        np.mean(
            erros
            > threshold_original
        )
        * 100
    )

    # --------------------------------------------------------
    # Verificação contra treinamento original
    # --------------------------------------------------------

    mediana_recalculada = (
        percentis["50"]
    )

    p95_recalculado = (
        percentis["95"]
    )

    p99_recalculado = (
        percentis["99"]
    )

    diferenca_mediana = abs(
        mediana_recalculada
        - mediana_original
    )

    diferenca_p95 = abs(
        p95_recalculado
        - threshold_original
    )

    diferenca_p99 = abs(
        p99_recalculado
        - p99_original
    )

    # --------------------------------------------------------
    # Salvar erros individuais
    # --------------------------------------------------------

    meta.to_csv(
        CAMINHO_SAIDA,
        index=False,
    )

    resumo = {
        "seed": SEED,
        "validation_fraction": (
            VALIDATION_FRACTION
        ),
        "window_size": window_size,
        "features": features,
        "sessoes_totais": int(
            len(sessoes)
        ),
        "sessoes_treino": int(
            len(sessoes_treino)
        ),
        "sessoes_validacao": int(
            len(sessoes_validacao)
        ),
        "janelas_validacao": int(
            len(erros)
        ),
        "erro_minimo": minimo,
        "erro_medio": media,
        "erro_desvio_padrao": desvio,
        "percentis": percentis,
        "threshold_original": (
            threshold_original
        ),
        "taxa_acima_threshold_pct": (
            taxa_threshold
        ),
        "verificacao_reproducao": {
            "mediana_original": (
                mediana_original
            ),
            "mediana_recalculada": (
                mediana_recalculada
            ),
            "diferenca_mediana": (
                diferenca_mediana
            ),
            "p95_original": (
                threshold_original
            ),
            "p95_recalculado": (
                p95_recalculado
            ),
            "diferenca_p95": (
                diferenca_p95
            ),
            "p99_original": (
                p99_original
            ),
            "p99_recalculado": (
                p99_recalculado
            ),
            "diferenca_p99": (
                diferenca_p99
            ),
        },
    }

    with open(
        CAMINHO_RESUMO,
        "w",
        encoding="utf-8",
    ) as arquivo:
        json.dump(
            resumo,
            arquivo,
            indent=4,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Resultado
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("DISTRIBUIÇÃO DO ERRO — NORMAL / VALIDAÇÃO")
    print("=" * 72)

    print(
        f"Mínimo:  {minimo:.8f}"
    )
    print(
        f"Média:   {media:.8f}"
    )
    print(
        f"Desvio:  {desvio:.8f}"
    )

    print()

    for p in percentis_desejados:
        print(
            f"P{p:<5}: "
            f"{percentis[str(p)]:.8f}"
        )

    print()
    print(
        "Taxa acima do threshold: "
        f"{taxa_threshold:.4f}%"
    )

    print()
    print("=" * 72)
    print("VERIFICAÇÃO DA REPRODUÇÃO")
    print("=" * 72)

    print(
        "Mediana original:     "
        f"{mediana_original:.8f}"
    )
    print(
        "Mediana recalculada:  "
        f"{mediana_recalculada:.8f}"
    )
    print(
        "Diferença:            "
        f"{diferenca_mediana:.12f}"
    )

    print()

    print(
        "P95 original:         "
        f"{threshold_original:.8f}"
    )
    print(
        "P95 recalculado:      "
        f"{p95_recalculado:.8f}"
    )
    print(
        "Diferença:            "
        f"{diferenca_p95:.12f}"
    )

    print()

    print(
        "P99 original:         "
        f"{p99_original:.8f}"
    )
    print(
        "P99 recalculado:      "
        f"{p99_recalculado:.8f}"
    )
    print(
        "Diferença:            "
        f"{diferenca_p99:.12f}"
    )

    print()
    print("=" * 72)
    print("ARQUIVOS GERADOS")
    print("=" * 72)

    print(
        CAMINHO_SAIDA
    )
    print(
        CAMINHO_RESUMO
    )

    print()
    print(
        "IMPORTANTE: nenhum modelo foi "
        "retreinado ou alterado."
    )

    print()
    print("=" * 72)
    print("CALIBRAÇÃO CONCLUÍDA")
    print("=" * 72)


if __name__ == "__main__":
    main()