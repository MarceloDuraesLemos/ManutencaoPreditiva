from __future__ import annotations

import json
import os
import random
from pathlib import Path

# Mantém compatibilidade com o ambiente usado no projeto.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Dense,
    Input,
    LSTM,
    RepeatVector,
    TimeDistributed,
)
from tensorflow.keras.models import Model


# ============================================================
# CONFIGURAÇÃO
# ============================================================

SEED = 42
WINDOW_SIZE = 30
VALIDATION_FRACTION = 0.20

EPOCHS = 40
BATCH_SIZE = 128

FEATURES_IA = [
    "rpm",
    "speed_kmh",
    "coolant_temp_c",
    "engine_load_pct",
    "throttle_pct",
    "intake_temp_c",
    "map_kpa",
    "battery_voltage",
]

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

PASTA_MODELO = (
    Path("modelo")
    / "automotivo_anomalia"
)

PASTA_RESULTADOS = (
    Path("resultados")
    / "automotivo"
)

PASTA_MODELO.mkdir(
    parents=True,
    exist_ok=True,
)

PASTA_RESULTADOS.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# REPRODUTIBILIDADE
# ============================================================

def configurar_seeds():

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


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
# VALIDAÇÃO
# ============================================================

def validar_dataframe(
    df: pd.DataFrame,
    nome: str,
):

    obrigatorias = [
        "session_id",
        "scenario",
        *FEATURES_IA,
    ]

    ausentes = [
        coluna
        for coluna in obrigatorias
        if coluna not in df.columns
    ]

    if ausentes:
        raise ValueError(
            f"{nome}: colunas ausentes: "
            f"{ausentes}"
        )

    if df[FEATURES_IA].isna().any().any():
        raise ValueError(
            f"{nome}: existem valores ausentes "
            f"nas features da IA."
        )


# ============================================================
# SPLIT POR SESSÃO
# ============================================================

def separar_sessoes(
    desenvolvimento: pd.DataFrame,
):

    sessoes = np.array(
        sorted(
            desenvolvimento[
                "session_id"
            ].unique()
        )
    )

    rng = np.random.default_rng(SEED)
    rng.shuffle(sessoes)

    n_validacao = int(
        len(sessoes)
        * VALIDATION_FRACTION
    )

    sessoes_validacao = sessoes[
        :n_validacao
    ]

    sessoes_treino = sessoes[
        n_validacao:
    ]

    treino = desenvolvimento[
        desenvolvimento[
            "session_id"
        ].isin(sessoes_treino)
    ].copy()

    validacao = desenvolvimento[
        desenvolvimento[
            "session_id"
        ].isin(sessoes_validacao)
    ].copy()

    return (
        treino,
        validacao,
        sessoes_treino,
        sessoes_validacao,
    )


# ============================================================
# SCALER
# ============================================================

def ajustar_scaler(
    treino: pd.DataFrame,
):

    scaler = StandardScaler()

    scaler.fit(
        treino[FEATURES_IA]
    )

    return scaler


# ============================================================
# CRIAÇÃO DE JANELAS
# ============================================================

def criar_janelas(
    df: pd.DataFrame,
    scaler: StandardScaler,
):

    janelas = []
    metadados = []

    for session_id, sessao in df.groupby(
        "session_id",
        sort=False,
    ):

        sessao = sessao.reset_index(
            drop=True
        )

        if len(sessao) < WINDOW_SIZE:
            continue

        valores = scaler.transform(
            sessao[FEATURES_IA]
        )

        for inicio in range(
            0,
            len(sessao) - WINDOW_SIZE + 1,
        ):

            fim = inicio + WINDOW_SIZE

            janela = valores[
                inicio:fim
            ]

            janelas.append(janela)

            metadados.append(
                {
                    "session_id": session_id,
                    "scenario": (
                        sessao[
                            "scenario"
                        ].iloc[fim - 1]
                    ),
                    "window_start": inicio,
                    "window_end": fim - 1,
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


# ============================================================
# MODELO
# ============================================================

def construir_autoencoder(
    n_features: int,
):

    entrada = Input(
        shape=(
            WINDOW_SIZE,
            n_features,
        ),
        name="entrada",
    )

    x = LSTM(
        32,
        name="encoder_lstm",
    )(entrada)

    x = RepeatVector(
        WINDOW_SIZE,
        name="repeat_vector",
    )(x)

    x = LSTM(
        32,
        return_sequences=True,
        name="decoder_lstm",
    )(x)

    saida = TimeDistributed(
        Dense(n_features),
        name="reconstrucao",
    )(x)

    modelo = Model(
        entrada,
        saida,
        name="autoencoder_automotivo",
    )

    modelo.compile(
        optimizer="adam",
        loss="mse",
    )

    return modelo


# ============================================================
# ERRO DE RECONSTRUÇÃO
# ============================================================

def calcular_erros(
    modelo: Model,
    X: np.ndarray,
):

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
# AVALIAÇÃO POR CENÁRIO
# ============================================================

def avaliar_teste(
    modelo,
    X_teste,
    meta_teste,
    threshold,
):

    erros = calcular_erros(
        modelo,
        X_teste,
    )

    resultado = meta_teste.copy()

    resultado[
        "reconstruction_error"
    ] = erros

    resultado[
        "anomaly"
    ] = (
        resultado[
            "reconstruction_error"
        ]
        > threshold
    )

    resumo = (
        resultado
        .groupby("scenario")
        .agg(
            janelas=(
                "anomaly",
                "size",
            ),
            janelas_anomalas=(
                "anomaly",
                "sum",
            ),
            erro_medio=(
                "reconstruction_error",
                "mean",
            ),
            erro_mediano=(
                "reconstruction_error",
                "median",
            ),
            erro_maximo=(
                "reconstruction_error",
                "max",
            ),
        )
    )

    resumo[
        "taxa_janelas_sinalizadas_pct"
    ] = (
        resumo[
            "janelas_anomalas"
        ]
        / resumo[
            "janelas"
        ]
        * 100
    )

    return resultado, resumo


# ============================================================
# MAIN
# ============================================================

def main():

    configurar_seeds()

    print()
    print("=" * 75)
    print("AUTOENCODER AUTOMOTIVO")
    print("=" * 75)

    # --------------------------------------------------------
    # DADOS
    # --------------------------------------------------------

    desenvolvimento, teste = (
        carregar_dados()
    )

    validar_dataframe(
        desenvolvimento,
        "DESENVOLVIMENTO",
    )

    validar_dataframe(
        teste,
        "TESTE",
    )

    (
        treino,
        validacao,
        sessoes_treino,
        sessoes_validacao,
    ) = separar_sessoes(
        desenvolvimento
    )

    print()
    print("Sessões de desenvolvimento:")
    print(
        f"Treino: {len(sessoes_treino)}"
    )
    print(
        f"Validação: "
        f"{len(sessoes_validacao)}"
    )

    # --------------------------------------------------------
    # SCALER
    # --------------------------------------------------------

    scaler = ajustar_scaler(
        treino
    )

    # --------------------------------------------------------
    # JANELAS
    # --------------------------------------------------------

    print()
    print("Criando janelas temporais...")

    X_treino, meta_treino = (
        criar_janelas(
            treino,
            scaler,
        )
    )

    X_validacao, meta_validacao = (
        criar_janelas(
            validacao,
            scaler,
        )
    )

    X_teste, meta_teste = (
        criar_janelas(
            teste,
            scaler,
        )
    )

    print(
        f"Janelas treino: "
        f"{len(X_treino)}"
    )

    print(
        f"Janelas validação: "
        f"{len(X_validacao)}"
    )

    print(
        f"Janelas teste: "
        f"{len(X_teste)}"
    )

    print(
        "Formato da entrada:",
        X_treino.shape,
    )

    # --------------------------------------------------------
    # MODELO
    # --------------------------------------------------------

    modelo = construir_autoencoder(
        len(FEATURES_IA)
    )

    print()
    modelo.summary()

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    print()
    print("=" * 75)
    print("TREINAMENTO")
    print("=" * 75)

    historico = modelo.fit(
        X_treino,
        X_treino,
        validation_data=(
            X_validacao,
            X_validacao,
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            early_stopping,
        ],
        shuffle=True,
        verbose=1,
    )

    # --------------------------------------------------------
    # THRESHOLD
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("CALIBRAÇÃO DO THRESHOLD")
    print("=" * 75)

    erros_validacao = calcular_erros(
        modelo,
        X_validacao,
    )

    threshold = float(
        np.percentile(
            erros_validacao,
            95,
        )
    )

    mediana_normal = float(
        np.median(
            erros_validacao
        )
    )

    p99_normal = float(
        np.percentile(
            erros_validacao,
            99,
        )
    )

    taxa_validacao = float(
        np.mean(
            erros_validacao
            > threshold
        )
        * 100
    )

    print(
        f"Mediana normal: "
        f"{mediana_normal:.8f}"
    )

    print(
        f"P95 / threshold: "
        f"{threshold:.8f}"
    )

    print(
        f"P99 normal: "
        f"{p99_normal:.8f}"
    )

    print(
        f"Janelas normais de validação "
        f"acima do threshold: "
        f"{taxa_validacao:.2f}%"
    )

    # --------------------------------------------------------
    # TESTE RESERVADO
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("TESTE NOS CENÁRIOS RESERVADOS")
    print("=" * 75)

    (
        resultado_teste,
        resumo_teste,
    ) = avaliar_teste(
        modelo,
        X_teste,
        meta_teste,
        threshold,
    )

    pd.set_option(
        "display.max_columns",
        None,
    )

    print()
    print(
        resumo_teste.round(6)
    )

    # --------------------------------------------------------
    # SALVAR MODELO
    # --------------------------------------------------------

    caminho_modelo = (
        PASTA_MODELO
        / "modelo_autoencoder.keras"
    )

    modelo.save(
        caminho_modelo
    )

    caminho_scaler = (
        PASTA_MODELO
        / "scaler.pkl"
    )

    joblib.dump(
        scaler,
        caminho_scaler,
    )

    # --------------------------------------------------------
    # CONFIG
    # --------------------------------------------------------

    config = {
        "modulo": "automotivo",
        "tipo_modelo": (
            "LSTM Autoencoder"
        ),
        "objetivo": (
            "deteccao_de_anomalias"
        ),
        "origem_treino": (
            "telemetria_simulada_normal"
        ),
        "window_size": WINDOW_SIZE,
        "sampling_interval_seconds": 1,
        "features": FEATURES_IA,
        "seed": SEED,
        "validation_fraction": (
            VALIDATION_FRACTION
        ),
        "threshold_method": (
            "P95_reconstruction_error_"
            "normal_validation"
        ),
        "threshold": threshold,
        "median_normal_validation": (
            mediana_normal
        ),
        "p99_normal_validation": (
            p99_normal
        ),
        "observacao": (
            "Modelo demonstrativo treinado "
            "com telemetria simulada. "
            "Nao representa diagnostico "
            "mecanico validado em veiculos reais."
        ),
    }

    with open(
        PASTA_MODELO / "config.json",
        "w",
        encoding="utf-8",
    ) as arquivo:
        json.dump(
            config,
            arquivo,
            indent=4,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # MÉTRICAS
    # --------------------------------------------------------

    metricas = {
        "best_train_loss": float(
            min(
                historico.history[
                    "loss"
                ]
            )
        ),
        "best_validation_loss": float(
            min(
                historico.history[
                    "val_loss"
                ]
            )
        ),
        "epochs_executed": len(
            historico.history[
                "loss"
            ]
        ),
        "threshold": threshold,
        "validation_normal_above_threshold_pct": (
            taxa_validacao
        ),
    }

    with open(
        PASTA_MODELO / "metricas.json",
        "w",
        encoding="utf-8",
    ) as arquivo:
        json.dump(
            metricas,
            arquivo,
            indent=4,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # RESULTADOS
    # --------------------------------------------------------

    resultado_teste.to_csv(
        PASTA_RESULTADOS
        / "janelas_teste_autoencoder.csv",
        index=False,
        encoding="utf-8",
    )

    resumo_teste.to_csv(
        PASTA_RESULTADOS
        / "desempenho_autoencoder_por_cenario.csv",
        encoding="utf-8",
    )

    historico_df = pd.DataFrame(
        historico.history
    )

    historico_df.index = (
        historico_df.index + 1
    )

    historico_df.index.name = "epoch"

    historico_df.to_csv(
        PASTA_RESULTADOS
        / "historico_treino_autoencoder.csv",
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("TREINAMENTO CONCLUÍDO")
    print("=" * 75)

    print()
    print(
        "Modelo:",
        caminho_modelo,
    )

    print(
        "Scaler:",
        caminho_scaler,
    )

    print(
        "Config:",
        PASTA_MODELO
        / "config.json",
    )

    print(
        "Métricas:",
        PASTA_MODELO
        / "metricas.json",
    )

    print()
    print(
        "IMPORTANTE: a taxa de janelas "
        "sinalizadas nos cenários artificiais "
        "não representa acurácia em veículos reais."
    )


if __name__ == "__main__":
    main()