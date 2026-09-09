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
    / "fd001_capped125"
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

# Limite utilizado para o RUL piecewise/capped
RUL_CAP = 125


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

    print("\nValores ausentes:")
    print(df.isnull().sum().sum())

    ciclos_por_motor = (
        df.groupby("unit_number")["time_in_cycles"]
        .max()
    )

    print("\nCiclos por motor:")

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
# GERAÇÃO DO RUL CAPPED
# =========================================================

def adicionar_rul_capped(
    df,
    rul_cap=RUL_CAP
):

    df = df.copy()

    ciclo_maximo = (
        df.groupby("unit_number")["time_in_cycles"]
        .transform("max")
    )

    # RUL linear original
    df["RUL_linear"] = (
        ciclo_maximo
        - df["time_in_cycles"]
    )

    # RUL usado como alvo do modelo
    df["RUL"] = (
        df["RUL_linear"]
        .clip(upper=rul_cap)
    )

    return df


# =========================================================
# VALIDAÇÃO DO RUL CAPPED
# =========================================================

def validar_rul_capped(df):

    print("\n" + "=" * 60)
    print("VALIDAÇÃO DO RUL CAPPED")
    print("=" * 60)

    print(
        f"\nLimite de RUL utilizado: "
        f"{RUL_CAP} ciclos"
    )

    print(
        f"RUL linear máximo: "
        f"{df['RUL_linear'].max()}"
    )

    print(
        f"RUL capped máximo: "
        f"{df['RUL'].max()}"
    )

    print(
        f"RUL mínimo: "
        f"{df['RUL'].min()}"
    )

    print("\nInício da vida do Motor 1:")

    print(
        df[
            df["unit_number"] == 1
        ][
            [
                "unit_number",
                "time_in_cycles",
                "RUL_linear",
                "RUL"
            ]
        ].head(10)
    )

    print("\nFinal da vida do Motor 1:")

    print(
        df[
            df["unit_number"] == 1
        ][
            [
                "unit_number",
                "time_in_cycles",
                "RUL_linear",
                "RUL"
            ]
        ].tail(10)
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
        f"\nMotores de treino: "
        f"{len(motores_treino)}"
    )

    print(
        f"Motores de validação: "
        f"{len(motores_validacao)}"
    )

    print(
        f"Motores em comum: "
        f"{len(motores_em_comum)}"
    )

    print(
        f"\nLinhas de treino: "
        f"{len(df_treino)}"
    )

    print(
        f"Linhas de validação: "
        f"{len(df_validacao)}"
    )

    print("\nIDs da validação:")
    print(sorted(motores_validacao))


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

    # O scaler aprende SOMENTE no conjunto de treino
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

            # RUL referente ao último ciclo da janela
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
    print("VALIDAÇÃO DAS JANELAS")
    print("=" * 60)

    print(
        f"\nX treino: "
        f"{X_treino.shape}"
    )

    print(
        f"y treino: "
        f"{y_treino.shape}"
    )

    print(
        f"\nX validação: "
        f"{X_validacao.shape}"
    )

    print(
        f"y validação: "
        f"{y_validacao.shape}"
    )

    print(
        f"\nFormato de uma janela: "
        f"{X_treino[0].shape}"
    )

    print(
        f"Maior RUL no y treino: "
        f"{y_treino.max()}"
    )

    print(
        f"Menor RUL no y treino: "
        f"{y_treino.min()}"
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
    print("RESULTADO DA VALIDAÇÃO - CAPPED 125")
    print("=" * 60)

    print(
        f"\nMAE validação: "
        f"{mae:.2f} ciclos"
    )

    print(
        f"RMSE validação: "
        f"{rmse:.2f} ciclos"
    )

    return mae, rmse


# =========================================================
# PREPARAÇÃO DO TESTE OFICIAL
# =========================================================

def preparar_teste(
    df_test,
    scaler
):

    df_test = df_test.copy()

    # IMPORTANTE:
    # não criamos RUL capped para o teste.
    # Apenas normalizamos os sensores.
    df_test[
        SENSORES_SELECIONADOS
    ] = scaler.transform(
        df_test[
            SENSORES_SELECIONADOS
        ]
    )

    return df_test


# =========================================================
# ÚLTIMA JANELA DE CADA MOTOR DE TESTE
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
# AVALIAÇÃO NO TESTE OFICIAL
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

    # RUL REAL OFICIAL DA NASA
    # NÃO É LIMITADO EM 125
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
    print("AVALIAÇÃO OFICIAL - FD001 CAPPED 125")
    print("=" * 60)

    print(
        f"\nMAE teste: "
        f"{mae:.2f} ciclos"
    )

    print(
        f"RMSE teste: "
        f"{rmse:.2f} ciclos"
    )

    print(
        "\nPrimeiras previsões:"
    )

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
# SALVAMENTO
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
    # MODELO
    # -----------------------------------------------------

    caminho_modelo = (
        MODEL_DIR
        / "modelo_rul.keras"
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
        "tipo_modelo": "LSTM",
        "rul_target": "capped",
        "rul_cap": RUL_CAP,
        "window_size": WINDOW_SIZE,
        "sensores": SENSORES_SELECIONADOS,
        "quantidade_sensores": len(
            SENSORES_SELECIONADOS
        ),
        "random_seed": 42,
        "observacao": (
            "RUL limitado a 125 ciclos "
            "somente durante treino e validacao"
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
        "dataset": "FD001",
        "rul_target": "capped",
        "rul_cap": RUL_CAP,
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
    # PREVISÕES DOS 100 MOTORES
    # -----------------------------------------------------

    caminho_resultados = (
        RESULTADOS_DIR
        / "fd001_capped125_previsoes.csv"
    )

    resultados.to_csv(
        caminho_resultados,
        index=False
    )

    # -----------------------------------------------------
    # CONFIRMAÇÃO
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("MODELO CAPPED 125 SALVO")
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
        f"{caminho_resultados}"
    )


# =========================================================
# EXECUÇÃO PRINCIPAL
# =========================================================

if __name__ == "__main__":

    # -----------------------------------------------------
    # 1. CARREGAR TREINO
    # -----------------------------------------------------

    df_train = carregar_dados(TRAIN_PATH)

    validar_dados(df_train)

    # -----------------------------------------------------
    # 2. CRIAR RUL CAPPED
    # -----------------------------------------------------

    df_train = adicionar_rul_capped(df_train)

    validar_rul_capped(df_train)

    # -----------------------------------------------------
    # 3. DIVIDIR TREINO / VALIDAÇÃO
    # -----------------------------------------------------

    df_treino, df_validacao = dividir_treino_validacao(
        df_train
    )

    validar_divisao(
        df_treino,
        df_validacao
    )

    # -----------------------------------------------------
    # 4. NORMALIZAR
    # -----------------------------------------------------

    df_treino, df_validacao, scaler = normalizar_dados(
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
    print("ARQUITETURA DO MODELO - CAPPED 125")
    print("=" * 60)

    model.summary()

    # -----------------------------------------------------
    # 7. TREINAR
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("INÍCIO DO TREINAMENTO - CAPPED 125")
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

    mae_validacao, rmse_validacao = avaliar_validacao(
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

    X_test, unidades_teste = criar_janelas_teste(
        df_test
    )

    print("\nFormato do teste oficial:")
    print(f"X teste: {X_test.shape}")
    print(
        f"Quantidade de RULs reais: "
        f"{len(rul_real)}"
    )

    # -----------------------------------------------------
    # 12. AVALIAÇÃO OFICIAL
    # -----------------------------------------------------

    resultados, mae_teste, rmse_teste = avaliar_teste(
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