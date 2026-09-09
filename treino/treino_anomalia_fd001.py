from pathlib import Path

import json
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    LSTM,
    RepeatVector,
    TimeDistributed,
    Dense
)
from tensorflow.keras.callbacks import EarlyStopping

import joblib


# =========================================================
# REPRODUTIBILIDADE
# =========================================================

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# =========================================================
# CAMINHOS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_PATH = (
    BASE_DIR
    / "datasets"
    / "CMAPSSData"
    / "train_FD001.txt"
)

MODEL_DIR = (
    BASE_DIR
    / "modelo"
    / "fd001_anomalia"
)

RESULTADOS_DIR = (
    BASE_DIR
    / "resultados"
    / "fd001_anomalia"
)


# =========================================================
# CONFIGURAÇÕES
# =========================================================

WINDOW_SIZE = 30

RUL_SAUDAVEL_MIN = 125

PERCENTIL_THRESHOLD = 95

SENSORES_SELECIONADOS = [
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_17",
    "sensor_20",
    "sensor_21",
]


# =========================================================
# COLUNAS C-MAPSS
# =========================================================

COLUMN_NAMES = [
    "unit_number",
    "time_in_cycles",
    "operational_setting_1",
    "operational_setting_2",
    "operational_setting_3",
] + [
    f"sensor_{i}"
    for i in range(1, 22)
]


# =========================================================
# CARREGAMENTO
# =========================================================

def carregar_dados():

    df = pd.read_csv(
        TRAIN_PATH,
        sep=r"\s+",
        header=None
    )

    df.columns = COLUMN_NAMES

    return df


# =========================================================
# RUL LINEAR
# =========================================================

def adicionar_rul_linear(df):

    df = df.copy()

    ciclo_maximo = (
        df.groupby("unit_number")["time_in_cycles"]
        .transform("max")
    )

    df["RUL_linear"] = (
        ciclo_maximo
        - df["time_in_cycles"]
    )

    return df


# =========================================================
# DIVISÃO POR MOTOR
# =========================================================

def dividir_treino_validacao(
    df,
    test_size=0.20
):

    motores = np.array(
        sorted(
            df["unit_number"].unique()
        )
    )

    motores_treino, motores_validacao = (
        train_test_split(
            motores,
            test_size=test_size,
            random_state=RANDOM_SEED
        )
    )

    df_treino = (
        df[
            df["unit_number"].isin(
                motores_treino
            )
        ]
        .copy()
    )

    df_validacao = (
        df[
            df["unit_number"].isin(
                motores_validacao
            )
        ]
        .copy()
    )

    return (
        df_treino,
        df_validacao
    )


# =========================================================
# NORMALIZAÇÃO
# =========================================================

def normalizar_dados(
    df_treino,
    df_validacao
):

    df_treino = df_treino.copy()
    df_validacao = df_validacao.copy()

    # -----------------------------------------------------
    # O scaler aprende somente com a região considerada
    # saudável dos motores de TREINO.
    # -----------------------------------------------------

    df_treino_saudavel = df_treino[
        df_treino["RUL_linear"]
        > RUL_SAUDAVEL_MIN
    ]

    if len(df_treino_saudavel) == 0:

        raise ValueError(
            "Nenhuma amostra saudável encontrada "
            "para ajustar o scaler."
        )

    scaler = MinMaxScaler()

    scaler.fit(
        df_treino_saudavel[
            SENSORES_SELECIONADOS
        ]
    )

    df_treino[
        SENSORES_SELECIONADOS
    ] = scaler.transform(
        df_treino[
            SENSORES_SELECIONADOS
        ]
    )

    df_validacao[
        SENSORES_SELECIONADOS
    ] = scaler.transform(
        df_validacao[
            SENSORES_SELECIONADOS
        ]
    )

    return (
        df_treino,
        df_validacao,
        scaler
    )


# =========================================================
# CRIAÇÃO DAS JANELAS
# =========================================================

def criar_janelas(df):

    X = []

    metadados = []

    for unit_number in sorted(
        df["unit_number"].unique()
    ):

        dados_motor = (
            df[
                df["unit_number"]
                == unit_number
            ]
            .sort_values(
                "time_in_cycles"
            )
            .reset_index(drop=True)
        )

        sensores = (
            dados_motor[
                SENSORES_SELECIONADOS
            ]
            .values
        )

        ciclos = (
            dados_motor[
                "time_in_cycles"
            ]
            .values
        )

        rul = (
            dados_motor[
                "RUL_linear"
            ]
            .values
        )

        for inicio in range(
            len(dados_motor)
            - WINDOW_SIZE
            + 1
        ):

            fim = (
                inicio
                + WINDOW_SIZE
            )

            janela = sensores[
                inicio:fim
            ]

            indice_final = (
                fim - 1
            )

            X.append(
                janela
            )

            metadados.append({
                "unit_number": int(
                    unit_number
                ),
                "cycle": int(
                    ciclos[indice_final]
                ),
                "RUL_linear": float(
                    rul[indice_final]
                )
            })

    X = np.array(
        X,
        dtype=np.float32
    )

    metadados = pd.DataFrame(
        metadados
    )

    return (
        X,
        metadados
    )


# =========================================================
# SELEÇÃO DAS JANELAS SAUDÁVEIS
# =========================================================

def selecionar_janelas_saudaveis(
    X,
    metadados
):

    mascara = (
        metadados[
            "RUL_linear"
        ]
        > RUL_SAUDAVEL_MIN
    ).values

    X_saudavel = (
        X[
            mascara
        ]
    )

    metadados_saudavel = (
        metadados[
            mascara
        ]
        .reset_index(drop=True)
    )

    return (
        X_saudavel,
        metadados_saudavel
    )


# =========================================================
# MODELO LSTM AUTOENCODER
# =========================================================

def criar_autoencoder():

    quantidade_sensores = len(
        SENSORES_SELECIONADOS
    )

    model = Sequential([
        Input(
            shape=(
                WINDOW_SIZE,
                quantidade_sensores
            )
        ),

        LSTM(
            64
        ),

        RepeatVector(
            WINDOW_SIZE
        ),

        LSTM(
            64,
            return_sequences=True
        ),

        TimeDistributed(
            Dense(
                quantidade_sensores
            )
        )
    ])

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model


# =========================================================
# TREINAMENTO
# =========================================================

def treinar_autoencoder(
    model,
    X_treino_saudavel,
    X_validacao_saudavel
):

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    historico = model.fit(
        X_treino_saudavel,
        X_treino_saudavel,

        validation_data=(
            X_validacao_saudavel,
            X_validacao_saudavel
        ),

        epochs=50,
        batch_size=64,

        callbacks=[
            early_stopping
        ],

        verbose=1
    )

    return historico


# =========================================================
# ERRO DE RECONSTRUÇÃO
# =========================================================

def calcular_erro_reconstrucao(
    model,
    X
):

    reconstrucoes = (
        model.predict(
            X,
            verbose=0
        )
    )

    erros = np.mean(
        np.square(
            X
            - reconstrucoes
        ),
        axis=(1, 2)
    )

    return erros


# =========================================================
# THRESHOLD
# =========================================================

def calcular_threshold(
    erros_validacao_saudavel
):

    threshold = np.percentile(
        erros_validacao_saudavel,
        PERCENTIL_THRESHOLD
    )

    return float(
        threshold
    )


# =========================================================
# CLASSIFICAÇÃO
# =========================================================

def classificar_anomalias(
    metadados,
    erros,
    threshold
):

    resultados = (
        metadados.copy()
    )

    resultados[
        "erro_reconstrucao"
    ] = erros

    resultados[
        "anomalia"
    ] = (
        resultados[
            "erro_reconstrucao"
        ]
        > threshold
    ).astype(int)

    resultados[
        "status"
    ] = np.where(
        resultados[
            "anomalia"
        ] == 1,
        "ANOMALO",
        "NORMAL"
    )

    return resultados


# =========================================================
# FAIXAS DE RUL PARA ANÁLISE
# =========================================================

def adicionar_faixa_rul(resultados):

    resultados = (
        resultados.copy()
    )

    def definir_faixa(rul):

        if rul > 125:
            return "RUL > 125"

        if rul > 75:
            return "76-125"

        if rul > 30:
            return "31-75"

        return "0-30"

    resultados[
        "faixa_RUL"
    ] = (
        resultados[
            "RUL_linear"
        ]
        .apply(
            definir_faixa
        )
    )

    return resultados


# =========================================================
# RESUMO POR FAIXA
# =========================================================

def gerar_resumo_por_faixa(
    resultados
):

    ordem = [
        "RUL > 125",
        "76-125",
        "31-75",
        "0-30"
    ]

    resumo = (
        resultados
        .groupby(
            "faixa_RUL",
            observed=False
        )
        .agg(
            quantidade=(
                "erro_reconstrucao",
                "count"
            ),

            erro_medio=(
                "erro_reconstrucao",
                "mean"
            ),

            erro_mediano=(
                "erro_reconstrucao",
                "median"
            ),

            taxa_anomalia=(
                "anomalia",
                "mean"
            )
        )
        .reindex(
            ordem
        )
        .reset_index()
    )

    resumo[
        "taxa_anomalia_percentual"
    ] = (
        resumo[
            "taxa_anomalia"
        ]
        * 100
    )

    return resumo


# =========================================================
# SALVAMENTO
# =========================================================

def salvar_resultados(
    model,
    scaler,
    threshold,
    resultados,
    resumo,
    erros_treino_saudavel,
    erros_validacao_saudavel,
    historico
):

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTADOS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------------------------
    # MODELO
    # -----------------------------------------------------

    caminho_modelo = (
        MODEL_DIR
        / "modelo_autoencoder.keras"
    )

    model.save(
        caminho_modelo
    )

    # -----------------------------------------------------
    # SCALER
    # -----------------------------------------------------

    caminho_scaler = (
        MODEL_DIR
        / "scaler.pkl"
    )

    joblib.dump(
        scaler,
        caminho_scaler
    )

    # -----------------------------------------------------
    # CONFIG
    # -----------------------------------------------------

    config = {
        "dataset": "FD001",
        "tipo_modelo": "LSTM Autoencoder",
        "objetivo": "deteccao_anomalia",
        "window_size": WINDOW_SIZE,

        "sensores": (
            SENSORES_SELECIONADOS
        ),

        "quantidade_sensores": len(
            SENSORES_SELECIONADOS
        ),

        "random_seed": RANDOM_SEED,

        "proxy_saudavel": (
            "RUL_linear > 125"
        ),

        "rul_saudavel_min": (
            RUL_SAUDAVEL_MIN
        ),

        "percentil_threshold": (
            PERCENTIL_THRESHOLD
        ),

        "threshold": float(
            threshold
        ),

        "observacao": (
            "RUL foi utilizado apenas para definir "
            "a regiao proxy de funcionamento saudavel "
            "durante o desenvolvimento. "
            "O modelo de anomalia recebe apenas sensores."
        )
    }

    caminho_config = (
        MODEL_DIR
        / "config.json"
    )

    with open(
        caminho_config,
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
    # MÉTRICAS
    # -----------------------------------------------------

    metricas = {
        "threshold": float(
            threshold
        ),

        "erro_medio_treino_saudavel": float(
            np.mean(
                erros_treino_saudavel
            )
        ),

        "erro_mediano_treino_saudavel": float(
            np.median(
                erros_treino_saudavel
            )
        ),

        "erro_medio_validacao_saudavel": float(
            np.mean(
                erros_validacao_saudavel
            )
        ),

        "erro_mediano_validacao_saudavel": float(
            np.median(
                erros_validacao_saudavel
            )
        ),

        "ultima_loss_treino": float(
            historico.history[
                "loss"
            ][-1]
        ),

        "ultima_loss_validacao": float(
            historico.history[
                "val_loss"
            ][-1]
        )
    }

    caminho_metricas = (
        MODEL_DIR
        / "metricas.json"
    )

    with open(
        caminho_metricas,
        "w",
        encoding="utf-8"
    ) as arquivo:

        json.dump(
            metricas,
            arquivo,
            indent=4,
            ensure_ascii=False
        )

    # -----------------------------------------------------
    # RESULTADOS DE TODA A VALIDAÇÃO
    # -----------------------------------------------------

    caminho_resultados = (
        RESULTADOS_DIR
        / "anomalias_validacao.csv"
    )

    resultados.to_csv(
        caminho_resultados,
        index=False
    )

    # -----------------------------------------------------
    # RESUMO POR FAIXA DE RUL
    # -----------------------------------------------------

    caminho_resumo = (
        RESULTADOS_DIR
        / "resumo_por_faixa_rul.csv"
    )

    resumo.to_csv(
        caminho_resumo,
        index=False
    )

    # -----------------------------------------------------
    # HISTÓRICO
    # -----------------------------------------------------

    df_historico = pd.DataFrame(
        historico.history
    )

    caminho_historico = (
        RESULTADOS_DIR
        / "historico_treinamento.csv"
    )

    df_historico.to_csv(
        caminho_historico,
        index=False
    )

    print("\n" + "=" * 60)
    print("ARQUIVOS SALVOS")
    print("=" * 60)

    print(
        f"\nModelo:\n"
        f"{caminho_modelo}"
    )

    print(
        f"\nScaler:\n"
        f"{caminho_scaler}"
    )

    print(
        f"\nConfig:\n"
        f"{caminho_config}"
    )

    print(
        f"\nMétricas:\n"
        f"{caminho_metricas}"
    )

    print(
        f"\nResultados:\n"
        f"{caminho_resultados}"
    )

    print(
        f"\nResumo:\n"
        f"{caminho_resumo}"
    )

    print(
        f"\nHistórico:\n"
        f"{caminho_historico}"
    )


# =========================================================
# EXECUÇÃO PRINCIPAL
# =========================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("DETECÇÃO DE ANOMALIAS - NASA FD001")
    print("=" * 60)

    # -----------------------------------------------------
    # 1. CARREGAR DADOS
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

    # -----------------------------------------------------
    # 2. RUL LINEAR
    # -----------------------------------------------------

    df = adicionar_rul_linear(
        df
    )

    # -----------------------------------------------------
    # 3. SPLIT POR MOTOR
    # -----------------------------------------------------

    (
        df_treino,
        df_validacao
    ) = dividir_treino_validacao(
        df
    )

    motores_treino = set(
        df_treino[
            "unit_number"
        ].unique()
    )

    motores_validacao = set(
        df_validacao[
            "unit_number"
        ].unique()
    )

    print(
        f"\nMotores treino: "
        f"{len(motores_treino)}"
    )

    print(
        f"Motores validação: "
        f"{len(motores_validacao)}"
    )

    print(
        f"Motores em comum: "
        f"{len(motores_treino.intersection(motores_validacao))}"
    )

    print(
        "\nIDs validação:"
    )

    print(
        sorted(
            motores_validacao
        )
    )

    # -----------------------------------------------------
    # 4. NORMALIZAÇÃO
    # -----------------------------------------------------

    (
        df_treino,
        df_validacao,
        scaler
    ) = normalizar_dados(
        df_treino,
        df_validacao
    )

    # -----------------------------------------------------
    # 5. TODAS AS JANELAS
    # -----------------------------------------------------

    (
        X_treino_todas,
        meta_treino
    ) = criar_janelas(
        df_treino
    )

    (
        X_validacao_todas,
        meta_validacao
    ) = criar_janelas(
        df_validacao
    )

    print("\nJanelas totais:")

    print(
        f"Treino: "
        f"{X_treino_todas.shape}"
    )

    print(
        f"Validação: "
        f"{X_validacao_todas.shape}"
    )

    # -----------------------------------------------------
    # 6. JANELAS SAUDÁVEIS
    # -----------------------------------------------------

    (
        X_treino_saudavel,
        meta_treino_saudavel
    ) = selecionar_janelas_saudaveis(
        X_treino_todas,
        meta_treino
    )

    (
        X_validacao_saudavel,
        meta_validacao_saudavel
    ) = selecionar_janelas_saudaveis(
        X_validacao_todas,
        meta_validacao
    )

    print("\nJanelas proxy saudáveis:")

    print(
        f"Treino: "
        f"{X_treino_saudavel.shape}"
    )

    print(
        f"Validação: "
        f"{X_validacao_saudavel.shape}"
    )

    if len(
        X_treino_saudavel
    ) == 0:

        raise ValueError(
            "Nenhuma janela saudável de treino."
        )

    if len(
        X_validacao_saudavel
    ) == 0:

        raise ValueError(
            "Nenhuma janela saudável de validação."
        )

    # -----------------------------------------------------
    # 7. MODELO
    # -----------------------------------------------------

    model = criar_autoencoder()

    print("\n" + "=" * 60)
    print("ARQUITETURA DO AUTOENCODER")
    print("=" * 60)

    model.summary()

    # -----------------------------------------------------
    # 8. TREINAMENTO
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("INÍCIO DO TREINAMENTO")
    print("=" * 60)

    historico = treinar_autoencoder(
        model,
        X_treino_saudavel,
        X_validacao_saudavel
    )

    # -----------------------------------------------------
    # 9. ERRO DAS JANELAS SAUDÁVEIS
    # -----------------------------------------------------

    erros_treino_saudavel = (
        calcular_erro_reconstrucao(
            model,
            X_treino_saudavel
        )
    )

    erros_validacao_saudavel = (
        calcular_erro_reconstrucao(
            model,
            X_validacao_saudavel
        )
    )

    # -----------------------------------------------------
    # 10. THRESHOLD
    # -----------------------------------------------------

    threshold = calcular_threshold(
        erros_validacao_saudavel
    )

    print("\n" + "=" * 60)
    print("THRESHOLD DE ANOMALIA")
    print("=" * 60)

    print(
        f"\nPercentil utilizado: "
        f"{PERCENTIL_THRESHOLD}"
    )

    print(
        f"Threshold: "
        f"{threshold:.8f}"
    )

    # -----------------------------------------------------
    # 11. AVALIAR TODAS AS JANELAS DE VALIDAÇÃO
    # -----------------------------------------------------

    erros_validacao_todas = (
        calcular_erro_reconstrucao(
            model,
            X_validacao_todas
        )
    )

    resultados = (
        classificar_anomalias(
            meta_validacao,
            erros_validacao_todas,
            threshold
        )
    )

    resultados = (
        adicionar_faixa_rul(
            resultados
        )
    )

    # -----------------------------------------------------
    # 12. RESUMO
    # -----------------------------------------------------

    resumo = (
        gerar_resumo_por_faixa(
            resultados
        )
    )

    print("\n" + "=" * 60)
    print("COMPORTAMENTO POR FAIXA DE RUL")
    print("=" * 60)

    print(
        resumo
        .round(6)
        .to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # 13. SALVAR
    # -----------------------------------------------------

    salvar_resultados(
        model,
        scaler,
        threshold,
        resultados,
        resumo,
        erros_treino_saudavel,
        erros_validacao_saudavel,
        historico
    )