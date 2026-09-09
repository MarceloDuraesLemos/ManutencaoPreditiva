from pathlib import Path
import os
import pickle

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


ROOT = Path(__file__).resolve().parents[1]

TEST_PATH = ROOT / "datasets" / "CMAPSSData" / "test_FD001.txt"

SENSORES = [
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


def carregar_scaler(path):
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


def carregar_cmapss(path):
    colunas = (
        ["unit_number", "time_in_cycles"]
        + [f"operational_setting_{i}" for i in range(1, 4)]
        + [f"sensor_{i}" for i in range(1, 22)]
    )

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None
    )

    df.columns = colunas

    return df


def criar_janelas(array, window_size):
    janelas = []

    for i in range(
        len(array) - window_size + 1
    ):
        janelas.append(
            array[i:i + window_size]
        )

    return np.asarray(janelas)


def main():
    print("=" * 70)
    print("COMPARACAO MODELOS H5 x KERAS")
    print("=" * 70)

    df = carregar_cmapss(TEST_PATH)

    # Vamos usar alguns motores diferentes para o teste
    motores_teste = [2, 34, 76, 100]

    df = df[
        df["unit_number"].isin(motores_teste)
    ].copy()

    # ==========================================================
    # RUL
    # ==========================================================

    scaler_rul = carregar_scaler(
        ROOT
        / "modelo"
        / "fd001_capped125"
        / "scaler.pkl"
    )

    modelo_rul_h5 = load_model(
        ROOT
        / "modelo"
        / "fd001_capped125"
        / "modelo_rul.h5",
        compile=False
    )

    modelo_rul_keras = load_model(
        ROOT
        / "modelo"
        / "fd001_capped125"
        / "modelo_rul_convertido.keras",
        compile=False
    )

    diferencas_rul = []

    print("\nTESTE RUL")

    for unit in motores_teste:
        dados_motor = (
            df[
                df["unit_number"] == unit
            ]
            .sort_values("time_in_cycles")
            .copy()
        )

        sensores = dados_motor[SENSORES]

        normalizado = scaler_rul.transform(
            sensores
        )

        janelas = criar_janelas(
            normalizado,
            WINDOW_SIZE
        )

        pred_h5 = modelo_rul_h5.predict(
            janelas,
            verbose=0
        ).reshape(-1)

        pred_keras = modelo_rul_keras.predict(
            janelas,
            verbose=0
        ).reshape(-1)

        diff = np.abs(
            pred_h5 - pred_keras
        )

        diferencas_rul.extend(
            diff.tolist()
        )

        print(
            f"Motor {unit}: "
            f"max diff = {diff.max():.12f}"
        )

    diferencas_rul = np.asarray(
        diferencas_rul
    )

    # ==========================================================
    # ANOMALIA
    # ==========================================================

    scaler_anomalia = carregar_scaler(
        ROOT
        / "modelo"
        / "fd001_anomalia"
        / "scaler.pkl"
    )

    modelo_anomalia_h5 = load_model(
        ROOT
        / "modelo"
        / "fd001_anomalia"
        / "modelo_autoencoder.h5",
        compile=False
    )

    modelo_anomalia_keras = load_model(
        ROOT
        / "modelo"
        / "fd001_anomalia"
        / "modelo_autoencoder_convertido.keras",
        compile=False
    )

    diferencas_anomalia = []

    print("\nTESTE AUTOENCODER")

    for unit in motores_teste:
        dados_motor = (
            df[
                df["unit_number"] == unit
            ]
            .sort_values("time_in_cycles")
            .copy()
        )

        sensores = dados_motor[SENSORES]

        normalizado = scaler_anomalia.transform(
            sensores
        )

        janelas = criar_janelas(
            normalizado,
            WINDOW_SIZE
        )

        rec_h5 = modelo_anomalia_h5.predict(
            janelas,
            verbose=0
        )

        rec_keras = modelo_anomalia_keras.predict(
            janelas,
            verbose=0
        )

        diff = np.abs(
            rec_h5 - rec_keras
        )

        diferencas_anomalia.extend(
            diff.reshape(-1).tolist()
        )

        print(
            f"Motor {unit}: "
            f"max diff = {diff.max():.12f}"
        )

    diferencas_anomalia = np.asarray(
        diferencas_anomalia
    )

    # ==========================================================
    # RESUMO
    # ==========================================================

    print("\n" + "=" * 70)
    print("RESUMO")
    print("=" * 70)

    print(
        "\nRUL:"
    )

    print(
        "Maior diferenca absoluta:",
        f"{diferencas_rul.max():.15f}"
    )

    print(
        "Diferenca media:",
        f"{diferencas_rul.mean():.15f}"
    )

    print(
        "\nAUTOENCODER:"
    )

    print(
        "Maior diferenca absoluta:",
        f"{diferencas_anomalia.max():.15f}"
    )

    print(
        "Diferenca media:",
        f"{diferencas_anomalia.mean():.15f}"
    )

    tolerancia = 1e-6

    rul_ok = (
        diferencas_rul.max()
        <= tolerancia
    )

    anomalia_ok = (
        diferencas_anomalia.max()
        <= tolerancia
    )

    print("\n" + "=" * 70)
    print("VALIDACAO")
    print("=" * 70)

    print(
        "RUL:",
        "OK" if rul_ok else "DIFERENTE"
    )

    print(
        "AUTOENCODER:",
        "OK" if anomalia_ok else "DIFERENTE"
    )

    if rul_ok and anomalia_ok:
        print(
            "\nMODELOS CONVERTIDOS EQUIVALENTES "
            "AOS MODELOS H5."
        )
    else:
        print(
            "\nATENCAO: existem diferencas acima "
            "da tolerancia definida."
        )


if __name__ == "__main__":
    main()