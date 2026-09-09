from pathlib import Path

import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping


# =========================================================
# REPRODUTIBILIDADE
# =========================================================

np.random.seed(42)
tf.random.set_seed(42)


# =========================================================
# CAMINHOS DO PROJETO
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_PATH = (
    BASE_DIR
    / "datasets"
    / "CMAPSSData"
    / "train_FD001.txt"
)

TEST_PATH = (
    BASE_DIR
    / "datasets"
    / "CMAPSSData"
    / "test_FD001.txt"
)

RUL_PATH = (
    BASE_DIR
    / "datasets"
    / "CMAPSSData"
    / "RUL_FD001.txt"
)

MODEL_DIR = (
    BASE_DIR
    / "modelo"
    / "fd001"
)

RESULTADOS_DIR = (
    BASE_DIR
    / "resultados"
)


# =========================================================
# NOMES DAS COLUNAS DO C-MAPSS
# =========================================================

COLUMN_NAMES = [
    "unit_number",
    "time_in_cycles",
    "operational_setting_1",
    "operational_setting_2",
    "operational_setting_3",
] + [f"sensor_{i}" for i in range(1, 22)]


# =========================================================
# CONFIGURAÇÕES
# =========================================================

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

WINDOW_SIZE = 30


# =========================================================
# CARREGAMENTO DOS DADOS
# =========================================================

def carregar_dados(caminho):

    df = pd.read_csv(
        caminho,
        sep=r"\s+",
        header=None
    )

    df.columns = COLUMN_NAMES

    return df


# =========================================================
# VALIDAÇÃO INICIAL
# =========================================================

def validar_dados(df):

    print("\n" + "=" * 60)
    print("VALIDAÇÃO DO DATASET FD001")
    print("=" * 60)

    print(f"\nQuantidade de linhas: {len(df)}")
    print(f"Quantidade de colunas: {df.shape[1]}")

    print(
        f"Quantidade de motores: "
        f"{df['unit_number'].nunique()}"
    )

    print("\nPrimeiras linhas:")
    print(df.head())

    print("\nColunas:")
    print(df.columns.tolist())

    print("\nValores ausentes:")
    print(df.isnull().sum().sum())

    print("\nCiclos por motor:")

    ciclos_por_motor = (
        df.groupby("unit_number")["time_in_cycles"]
        .max()
    )

    print(
        f"Menor vida útil: "
        f"{ciclos_por_motor.min()} ciclos"
    )

    print(
        f"Maior vida útil: "
        f"{ciclos_por_motor.max()} ciclos"
    )

    print(
        f"Vida útil média: "
        f"{ciclos_por_motor.mean():.2f} ciclos"
    )


# =========================================================
# GERAÇÃO DO RUL
# =========================================================

def adicionar_rul(df):

    df = df.copy()

    ciclo_maximo = (
        df.groupby("unit_number")["time_in_cycles"]
        .transform("max")
    )

    df["RUL"] = (
        ciclo_maximo
        - df["time_in_cycles"]
    )

    return df


# =========================================================
# VALIDAÇÃO DO RUL E DOS SENSORES
# =========================================================

def validar_rul_e_sensores(df):

    print("\n" + "=" * 60)
    print("VALIDAÇÃO DO RUL E DOS SENSORES")
    print("=" * 60)

    print("\nSensores selecionados:")

    for sensor in SENSORES_SELECIONADOS:
        print(f"- {sensor}")

    print(
        f"\nQuantidade de sensores selecionados: "
        f"{len(SENSORES_SELECIONADOS)}"
    )

    print("\nExemplo do Motor 1:")

    print(
        df[
            df["unit_number"] == 1
        ][
            [
                "unit_number",
                "time_in_cycles",
                "RUL"
            ]
        ].head(10)
    )

    print("\nÚltimos ciclos do Motor 1:")

    print(
        df[
            df["unit_number"] == 1
        ][
            [
                "unit_number",
                "time_in_cycles",
                "RUL"
            ]
        ].tail(10)
    )

    print(
        "\nRUL mínimo:",
        df["RUL"].min()
    )

    print(
        "RUL máximo:",
        df["RUL"].max()
    )


# =========================================================
# DIVISÃO TREINO / VALIDAÇÃO
# =========================================================

def dividir_treino_validacao(
    df,
    test_size=0.20,
    random_state=42
):

    motores = (
        df["unit_number"]
        .unique()
    )

    motores_treino, motores_validacao = (
        train_test_split(
            motores,
            test_size=test_size,
            random_state=random_state
        )
    )

    df_treino = df[
        df["unit_number"].isin(
            motores_treino
        )
    ].copy()

    df_validacao = df[
        df["unit_number"].isin(
            motores_validacao
        )
    ].copy()

    return (
        df_treino,
        df_validacao
    )


def validar_divisao(
    df_treino,
    df_validacao
):

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

    motores_em_comum = (
        motores_treino.intersection(
            motores_validacao
        )
    )

    print("\n" + "=" * 60)
    print("DIVISÃO TREINO / VALIDAÇÃO")
    print("=" * 60)

    print(
        f"\nMotores no treino: "
        f"{len(motores_treino)}"
    )

    print(
        f"Motores na validação: "
        f"{len(motores_validacao)}"
    )

    print(
        f"\nLinhas de treino: "
        f"{len(df_treino)}"
    )

    print(
        f"Linhas de validação: "
        f"{len(df_validacao)}"
    )

    print(
        f"\nMotores presentes nos dois conjuntos: "
        f"{len(motores_em_comum)}"
    )

    print("\nIDs dos motores de validação:")

    print(
        sorted(motores_validacao)
    )


# =========================================================
# NORMALIZAÇÃO
# =========================================================

def normalizar_dados(
    df_treino,
    df_validacao
):

    scaler = MinMaxScaler()

    df_treino = df_treino.copy()
    df_validacao = df_validacao.copy()

    # O scaler aprende apenas no conjunto de treino
    scaler.fit(
        df_treino[
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


def validar_normalizacao(
    df_treino,
    df_validacao
):

    print("\n" + "=" * 60)
    print("VALIDAÇÃO DA NORMALIZAÇÃO")
    print("=" * 60)

    print("\nTreino:")

    print(
        df_treino[
            SENSORES_SELECIONADOS
        ]
        .agg(["min", "max"])
        .round(3)
    )

    print("\nValidação:")

    print(
        df_validacao[
            SENSORES_SELECIONADOS
        ]
        .agg(["min", "max"])
        .round(3)
    )


# =========================================================
# JANELAS TEMPORAIS
# =========================================================

def criar_janelas(
    df,
    window_size=WINDOW_SIZE
):

    X = []
    y = []

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
        )

        sensores = (
            dados_motor[
                SENSORES_SELECIONADOS
            ]
            .values
        )

        rul = (
            dados_motor[
                "RUL"
            ]
            .values
        )

        for inicio in range(
            len(dados_motor)
            - window_size
            + 1
        ):

            fim = (
                inicio
                + window_size
            )

            janela = sensores[
                inicio:fim
            ]

            alvo_rul = rul[
                fim - 1
            ]

            X.append(
                janela
            )

            y.append(
                alvo_rul
            )

    X = np.array(
        X,
        dtype=np.float32
    )

    y = np.array(
        y,
        dtype=np.float32
    )

    return X, y


def validar_janelas(
    X_treino,
    y_treino,
    X_validacao,
    y_validacao
):

    print("\n" + "=" * 60)
    print("VALIDAÇÃO DAS JANELAS TEMPORAIS")
    print("=" * 60)

    print("\nTREINO")

    print(
        f"X treino: "
        f"{X_treino.shape}"
    )

    print(
        f"y treino: "
        f"{y_treino.shape}"
    )

    print("\nVALIDAÇÃO")

    print(
        f"X validação: "
        f"{X_validacao.shape}"
    )

    print(
        f"y validação: "
        f"{y_validacao.shape}"
    )

    print("\nFormato de uma janela:")
    print(
        X_treino[0].shape
    )

    print("\nRUL da primeira janela:")
    print(
        y_treino[0]
    )

    print("\nRUL da última janela:")
    print(
        y_treino[-1]
    )


# =========================================================
# MODELO LSTM
# =========================================================

def criar_modelo_lstm():

    model = Sequential([
        Input(
            shape=(
                WINDOW_SIZE,
                len(
                    SENSORES_SELECIONADOS
                )
            )
        ),

        LSTM(
            64
        ),

        Dense(
            32,
            activation="relu"
        ),

        Dense(
            1
        )
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(
                name="mae"
            ),
            tf.keras.metrics.RootMeanSquaredError(
                name="rmse"
            )
        ]
    )

    return model


# =========================================================
# TREINAMENTO
# =========================================================

def treinar_modelo(
    model,
    X_treino,
    y_treino,
    X_validacao,
    y_validacao
):

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    historico = model.fit(
        X_treino,
        y_treino,
        validation_data=(
            X_validacao,
            y_validacao
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
# AVALIAÇÃO DA VALIDAÇÃO
# =========================================================

def avaliar_validacao(
    model,
    X_validacao,
    y_validacao
):

    previsoes = (
        model.predict(
            X_validacao,
            verbose=0
        )
        .reshape(-1)
    )

    mae = mean_absolute_error(
        y_validacao,
        previsoes
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_validacao,
            previsoes
        )
    )

    print("\n" + "=" * 60)
    print("RESULTADO DA VALIDAÇÃO - LINEAR")
    print("=" * 60)

    print(
        f"\nMAE validação: "
        f"{mae:.2f} ciclos"
    )

    print(
        f"RMSE validação: "
        f"{rmse:.2f} ciclos"
    )

    return (
        mae,
        rmse
    )


# =========================================================
# PREPARAÇÃO DO TESTE OFICIAL
# =========================================================

def preparar_teste(
    df_test,
    scaler
):

    df_test = df_test.copy()

    df_test[
        SENSORES_SELECIONADOS
    ] = scaler.transform(
        df_test[
            SENSORES_SELECIONADOS
        ]
    )

    return df_test


# =========================================================
# CRIAÇÃO DAS JANELAS DE TESTE
# =========================================================

def criar_janelas_teste(
    df_test,
    window_size=WINDOW_SIZE
):

    X_test = []
    unidades = []

    for unit_number in sorted(
        df_test["unit_number"].unique()
    ):

        dados_motor = (
            df_test[
                df_test["unit_number"]
                == unit_number
            ]
            .sort_values(
                "time_in_cycles"
            )
        )

        if len(
            dados_motor
        ) < window_size:

            raise ValueError(
                f"Motor {unit_number} possui "
                f"apenas {len(dados_motor)} ciclos."
            )

        janela = (
            dados_motor[
                SENSORES_SELECIONADOS
            ]
            .values[
                -window_size:
            ]
        )

        X_test.append(
            janela
        )

        unidades.append(
            unit_number
        )

    X_test = np.array(
        X_test,
        dtype=np.float32
    )

    return (
        X_test,
        unidades
    )


# =========================================================
# AVALIAÇÃO DO TESTE OFICIAL
# =========================================================

def avaliar_teste(
    model,
    X_test,
    unidades,
    rul_real
):

    previsoes = (
        model.predict(
            X_test,
            verbose=0
        )
        .reshape(-1)
    )

    valores_reais = (
        rul_real[
            "RUL"
        ]
        .values
    )

    mae = mean_absolute_error(
        valores_reais,
        previsoes
    )

    rmse = np.sqrt(
        mean_squared_error(
            valores_reais,
            previsoes
        )
    )

    resultados = pd.DataFrame({
        "unit_number": unidades,
        "RUL_real": valores_reais,
        "RUL_previsto": previsoes
    })

    resultados[
        "erro"
    ] = (
        resultados[
            "RUL_previsto"
        ]
        - resultados[
            "RUL_real"
        ]
    )

    resultados[
        "erro_absoluto"
    ] = (
        resultados[
            "erro"
        ].abs()
    )

    print("\n" + "=" * 60)
    print("AVALIAÇÃO OFICIAL - FD001 LINEAR")
    print("=" * 60)

    print(
        f"\nMAE teste: "
        f"{mae:.2f} ciclos"
    )

    print(
        f"RMSE teste: "
        f"{rmse:.2f} ciclos"
    )

    print("\nPrimeiras previsões:")

    print(
        resultados
        .head(10)
        .round(2)
    )

    return (
        resultados,
        mae,
        rmse
    )


# =========================================================
# SALVAMENTO DO MODELO E DOS RESULTADOS
# =========================================================

def salvar_modelo_e_resultados(
    model,
    scaler,
    resultados,
    mae_validacao,
    rmse_validacao,
    mae_teste,
    rmse_teste
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
    # 1. MODELO
    # -----------------------------------------------------

    caminho_modelo = (
        MODEL_DIR
        / "modelo_rul.keras"
    )

    model.save(
        caminho_modelo
    )

    # -----------------------------------------------------
    # 2. SCALER
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
    # 3. CONFIGURAÇÕES
    # -----------------------------------------------------

    config = {
        "dataset": "FD001",
        "tipo_modelo": "LSTM",
        "rul_target": "linear",
        "window_size": WINDOW_SIZE,
        "sensores": SENSORES_SELECIONADOS,
        "quantidade_sensores": len(
            SENSORES_SELECIONADOS
        ),
        "random_seed": 42,
        "split_validacao": 0.20
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
    # 4. MÉTRICAS
    # -----------------------------------------------------

    metricas = {
        "dataset": "FD001",
        "rul_target": "linear",
        "mae_validacao": float(
            mae_validacao
        ),
        "rmse_validacao": float(
            rmse_validacao
        ),
        "mae_teste": float(
            mae_teste
        ),
        "rmse_teste": float(
            rmse_teste
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
    # 5. PREVISÕES DOS 100 MOTORES
    # -----------------------------------------------------

    caminho_previsoes = (
        RESULTADOS_DIR
        / "fd001_linear_previsoes.csv"
    )

    resultados.to_csv(
        caminho_previsoes,
        index=False
    )

    # -----------------------------------------------------
    # CONFIRMAÇÃO
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("MODELO LINEAR E RESULTADOS SALVOS")
    print("=" * 60)

    print(
        f"\nModelo: "
        f"{caminho_modelo}"
    )

    print(
        f"Scaler: "
        f"{caminho_scaler}"
    )

    print(
        f"Configuração: "
        f"{caminho_config}"
    )

    print(
        f"Métricas: "
        f"{caminho_metricas}"
    )

    print(
        f"Previsões: "
        f"{caminho_previsoes}"
    )


# =========================================================
# EXECUÇÃO PRINCIPAL
# =========================================================

if __name__ == "__main__":

    # -----------------------------------------------------
    # 1. CARREGAR TREINO
    # -----------------------------------------------------

    df_train = carregar_dados(
        TRAIN_PATH
    )

    validar_dados(
        df_train
    )

    # -----------------------------------------------------
    # 2. CRIAR RUL LINEAR
    # -----------------------------------------------------

    df_train = adicionar_rul(
        df_train
    )

    validar_rul_e_sensores(
        df_train
    )

    # -----------------------------------------------------
    # 3. DIVIDIR TREINO / VALIDAÇÃO
    # -----------------------------------------------------

    (
        df_treino,
        df_validacao
    ) = dividir_treino_validacao(
        df_train
    )

    validar_divisao(
        df_treino,
        df_validacao
    )

    # -----------------------------------------------------
    # 4. NORMALIZAR
    # -----------------------------------------------------

    (
        df_treino,
        df_validacao,
        scaler
    ) = normalizar_dados(
        df_treino,
        df_validacao
    )

    validar_normalizacao(
        df_treino,
        df_validacao
    )

    # -----------------------------------------------------
    # 5. CRIAR JANELAS TEMPORAIS
    # -----------------------------------------------------

    X_treino, y_treino = criar_janelas(
        df_treino
    )

    X_validacao, y_validacao = criar_janelas(
        df_validacao
    )

    validar_janelas(
        X_treino,
        y_treino,
        X_validacao,
        y_validacao
    )

    # -----------------------------------------------------
    # 6. CRIAR MODELO
    # -----------------------------------------------------

    model = criar_modelo_lstm()

    print("\n" + "=" * 60)
    print("ARQUITETURA DO MODELO - LINEAR")
    print("=" * 60)

    model.summary()

    # -----------------------------------------------------
    # 7. TREINAR
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("INÍCIO DO TREINAMENTO - LINEAR")
    print("=" * 60)

    historico = treinar_modelo(
        model,
        X_treino,
        y_treino,
        X_validacao,
        y_validacao
    )

    # -----------------------------------------------------
    # 8. AVALIAR VALIDAÇÃO
    # -----------------------------------------------------

    (
        mae_validacao,
        rmse_validacao
    ) = avaliar_validacao(
        model,
        X_validacao,
        y_validacao
    )

    # -----------------------------------------------------
    # 9. CARREGAR TESTE OFICIAL
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("PREPARAÇÃO DO TESTE OFICIAL NASA")
    print("=" * 60)

    df_test = carregar_dados(
        TEST_PATH
    )

    rul_real = pd.read_csv(
        RUL_PATH,
        sep=r"\s+",
        header=None,
        names=["RUL"]
    )

    # -----------------------------------------------------
    # 10. NORMALIZAR TESTE
    # -----------------------------------------------------

    df_test = preparar_teste(
        df_test,
        scaler
    )

    # -----------------------------------------------------
    # 11. CRIAR ÚLTIMA JANELA DE CADA MOTOR
    # -----------------------------------------------------

    (
        X_test,
        unidades_teste
    ) = criar_janelas_teste(
        df_test
    )

    print("\nFormato do teste oficial:")

    print(
        f"X teste: "
        f"{X_test.shape}"
    )

    print(
        f"Quantidade de RULs reais: "
        f"{len(rul_real)}"
    )

    # -----------------------------------------------------
    # 12. AVALIAÇÃO OFICIAL
    # -----------------------------------------------------

    (
        resultados,
        mae_teste,
        rmse_teste
    ) = avaliar_teste(
        model,
        X_test,
        unidades_teste,
        rul_real
    )

    # -----------------------------------------------------
    # 13. SALVAR MODELO E RESULTADOS
    # -----------------------------------------------------

    salvar_modelo_e_resultados(
        model,
        scaler,
        resultados,
        mae_validacao,
        rmse_validacao,
        mae_teste,
        rmse_teste
    )