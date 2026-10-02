from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURAÇÃO
# ============================================================

CENARIOS_DISPONIVEIS = (
    "NORMAL",
    "AQUECIMENTO_PROGRESSIVO",
    "OPERACAO_IRREGULAR",
    "CONDICAO_SEVERA_SIMULADA",
)


@dataclass
class ConfigSimulacao:
    vehicle_id: str = "VEICULO_DEMO_01"
    duracao_minutos: int = 20
    intervalo_segundos: int = 1
    cenario: str = "NORMAL"
    seed: int = 42


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def _limitar(valor, minimo, maximo):
    return np.clip(valor, minimo, maximo)


def _suavizar(serie, janela=7):
    return (
        pd.Series(serie)
        .rolling(window=janela, min_periods=1, center=True)
        .mean()
        .to_numpy()
    )


def _gerar_perfil_velocidade(n: int, rng: np.random.Generator):
    """
    Gera um perfil simples de condução urbana/mista.

    O objetivo não é reproduzir um veículo específico, mas gerar
    telemetria coerente para demonstração do pipeline.
    """

    velocidade = np.zeros(n, dtype=float)

    velocidade_atual = 0.0
    velocidade_alvo = 0.0
    tempo_ate_novo_alvo = 0

    for i in range(n):
        if tempo_ate_novo_alvo <= 0:
            tipo = rng.choice(
                ["parado", "urbano", "via_rapida"],
                p=[0.18, 0.62, 0.20],
            )

            if tipo == "parado":
                velocidade_alvo = 0.0
                tempo_ate_novo_alvo = int(rng.integers(5, 35))

            elif tipo == "urbano":
                velocidade_alvo = float(rng.uniform(15, 60))
                tempo_ate_novo_alvo = int(rng.integers(15, 60))

            else:
                velocidade_alvo = float(rng.uniform(60, 100))
                tempo_ate_novo_alvo = int(rng.integers(20, 90))

        diferenca = velocidade_alvo - velocidade_atual

        if diferenca > 0:
            velocidade_atual += min(
                diferenca,
                float(rng.uniform(0.3, 1.3)),
            )

        elif diferenca < 0:
            velocidade_atual -= min(
                abs(diferenca),
                float(rng.uniform(0.4, 1.8)),
            )

        velocidade_atual += float(rng.normal(0, 0.15))
        velocidade_atual = max(0.0, velocidade_atual)

        if velocidade_atual < 0.8:
            velocidade_atual = 0.0

        velocidade[i] = velocidade_atual
        tempo_ate_novo_alvo -= 1

    return _suavizar(velocidade, janela=5)


# ============================================================
# SIMULAÇÃO PRINCIPAL
# ============================================================

def gerar_telemetria(config: ConfigSimulacao) -> pd.DataFrame:

    if config.cenario not in CENARIOS_DISPONIVEIS:
        raise ValueError(
            f"Cenário inválido: {config.cenario}. "
            f"Opções: {CENARIOS_DISPONIVEIS}"
        )

    if config.duracao_minutos <= 0:
        raise ValueError("A duração deve ser maior que zero.")

    if config.intervalo_segundos <= 0:
        raise ValueError("O intervalo deve ser maior que zero.")

    rng = np.random.default_rng(config.seed)

    total_segundos = config.duracao_minutos * 60

    n = max(
        1,
        total_segundos // config.intervalo_segundos,
    )

    tempo_s = np.arange(n) * config.intervalo_segundos

    inicio = pd.Timestamp(datetime.now()).floor("s")

    timestamps = inicio + pd.to_timedelta(
        tempo_s,
        unit="s",
    )

    # --------------------------------------------------------
    # VELOCIDADE
    # --------------------------------------------------------

    velocidade = _gerar_perfil_velocidade(n, rng)

    aceleracao = np.gradient(
        velocidade,
        config.intervalo_segundos,
    )

    # --------------------------------------------------------
    # THROTTLE
    # --------------------------------------------------------

    throttle = (
        10
        + velocidade * 0.16
        + np.maximum(aceleracao, 0) * 11
        + rng.normal(0, 2.0, n)
    )

    parado = velocidade < 1

    throttle[parado] = (
        12
        + rng.normal(0, 1.0, parado.sum())
    )

    throttle = _limitar(throttle, 5, 90)

    # --------------------------------------------------------
    # RPM
    # --------------------------------------------------------

    rpm = (
        750
        + velocidade * 23
        + np.maximum(aceleracao, 0) * 180
        + throttle * 7
        + rng.normal(0, 65, n)
    )

    rpm[parado] = (
        780
        + rng.normal(0, 35, parado.sum())
    )

    rpm = _limitar(rpm, 650, 5500)

    # --------------------------------------------------------
    # CARGA DO MOTOR
    # --------------------------------------------------------

    engine_load = (
        15
        + throttle * 0.72
        + np.maximum(aceleracao, 0) * 9
        + rng.normal(0, 3, n)
    )

    engine_load = _limitar(engine_load, 8, 100)

    # --------------------------------------------------------
    # TEMPERATURA DO MOTOR
    # --------------------------------------------------------

    progresso_aquecimento = 1 - np.exp(
        -tempo_s / 240
    )

    coolant_temp = (
        28
        + 62 * progresso_aquecimento
        + engine_load * 0.025
        + rng.normal(0, 0.6, n)
    )

    # --------------------------------------------------------
    # TEMPERATURA DO AR DE ADMISSÃO
    # --------------------------------------------------------

    intake_temp = (
        27
        + 0.035 * coolant_temp
        + rng.normal(0, 1.0, n)
    )

    # --------------------------------------------------------
    # MAP
    # --------------------------------------------------------

    map_kpa = (
        28
        + throttle * 0.70
        + engine_load * 0.16
        + rng.normal(0, 2.0, n)
    )

    map_kpa = _limitar(map_kpa, 20, 105)

    # --------------------------------------------------------
    # MAF
    # --------------------------------------------------------

    maf = (
        1.7
        + rpm * 0.0024
        + engine_load * 0.055
        + rng.normal(0, 0.4, n)
    )

    maf = _limitar(maf, 1.0, 60)

    # --------------------------------------------------------
    # BATERIA / SISTEMA DE CARGA
    # --------------------------------------------------------

    battery_voltage = (
        14.15
        + rng.normal(0, 0.08, n)
    )

    # --------------------------------------------------------
    # COMBUSTÍVEL
    # --------------------------------------------------------

    fuel_inicial = float(rng.uniform(55, 90))

    consumo_relativo = np.cumsum(
        (
            0.0007
            + engine_load * 0.000008
            + rpm * 0.00000025
        )
        * config.intervalo_segundos
    )

    fuel_level = fuel_inicial - consumo_relativo
    fuel_level = _limitar(fuel_level, 0, 100)

    # ========================================================
    # CENÁRIOS ARTIFICIAIS
    # ========================================================

    inicio_evento = int(n * 0.55)

    progresso_evento = np.zeros(n)

    if inicio_evento < n:
        progresso_evento[inicio_evento:] = np.linspace(
            0,
            1,
            n - inicio_evento,
        )

    # --------------------------------------------------------
    # AQUECIMENTO PROGRESSIVO
    # --------------------------------------------------------

    if config.cenario == "AQUECIMENTO_PROGRESSIVO":

        coolant_temp += (
            progresso_evento * 18
        )

        intake_temp += (
            progresso_evento * 5
        )

    # --------------------------------------------------------
    # OPERAÇÃO IRREGULAR
    # --------------------------------------------------------

    elif config.cenario == "OPERACAO_IRREGULAR":

        mascara = np.arange(n) >= inicio_evento

        oscilacao = (
            np.sin(np.arange(n) * 0.35)
            * 350
            * progresso_evento
        )

        rpm += oscilacao

        rpm[mascara] += rng.normal(
            0,
            150,
            mascara.sum(),
        )

        engine_load[mascara] += rng.normal(
            8,
            7,
            mascara.sum(),
        )

        throttle[mascara] += rng.normal(
            4,
            5,
            mascara.sum(),
        )

    # --------------------------------------------------------
    # CONDIÇÃO SEVERA SIMULADA
    # --------------------------------------------------------

    elif config.cenario == "CONDICAO_SEVERA_SIMULADA":

        coolant_temp += (
            progresso_evento * 28
        )

        intake_temp += (
            progresso_evento * 8
        )

        engine_load += (
            progresso_evento * 15
        )

        battery_voltage -= (
            progresso_evento * 1.4
        )

        rpm += (
            np.sin(np.arange(n) * 0.30)
            * 220
            * progresso_evento
        )

    # --------------------------------------------------------
    # LIMITES FINAIS DE SEGURANÇA DA SIMULAÇÃO
    # --------------------------------------------------------

    rpm = _limitar(rpm, 600, 6500)
    velocidade = _limitar(velocidade, 0, 160)
    coolant_temp = _limitar(coolant_temp, 20, 130)
    engine_load = _limitar(engine_load, 0, 100)
    throttle = _limitar(throttle, 0, 100)
    intake_temp = _limitar(intake_temp, -10, 80)
    map_kpa = _limitar(map_kpa, 10, 120)
    maf = _limitar(maf, 0, 100)
    battery_voltage = _limitar(
        battery_voltage,
        10,
        15.5,
    )

    # ========================================================
    # DATAFRAME FINAL
    # ========================================================

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "vehicle_id": config.vehicle_id,
            "rpm": np.round(rpm, 1),
            "speed_kmh": np.round(velocidade, 1),
            "coolant_temp_c": np.round(
                coolant_temp,
                1,
            ),
            "engine_load_pct": np.round(
                engine_load,
                1,
            ),
            "throttle_pct": np.round(
                throttle,
                1,
            ),
            "intake_temp_c": np.round(
                intake_temp,
                1,
            ),
            "map_kpa": np.round(
                map_kpa,
                1,
            ),
            "maf_g_s": np.round(
                maf,
                2,
            ),
            "battery_voltage": np.round(
                battery_voltage,
                2,
            ),
            "fuel_level_pct": np.round(
                fuel_level,
                2,
            ),
            "scenario": config.cenario,
            "data_origin": "SIMULADO",
        }
    )

    return df


# ============================================================
# EXPORTAÇÃO
# ============================================================

def salvar_csv(
    df: pd.DataFrame,
    caminho: str | Path,
):
    caminho = Path(caminho)

    caminho.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        caminho,
        index=False,
        encoding="utf-8",
    )


# ============================================================
# TESTE DIRETO
# ============================================================

if __name__ == "__main__":

    config = ConfigSimulacao(
        vehicle_id="VEICULO_DEMO_01",
        duracao_minutos=20,
        intervalo_segundos=1,
        cenario="NORMAL",
        seed=42,
    )

    dados = gerar_telemetria(config)

    caminho_saida = (
        Path("datasets")
        / "automotivo"
        / "simulados"
        / "telemetria_normal.csv"
    )

    salvar_csv(
        dados,
        caminho_saida,
    )

    print()
    print("=" * 60)
    print("SIMULADOR AUTOMOTIVO")
    print("=" * 60)

    print(f"Veículo: {config.vehicle_id}")
    print(f"Cenário: {config.cenario}")
    print(f"Registros: {len(dados)}")
    print(
        f"Duração: "
        f"{config.duracao_minutos} minutos"
    )

    print()
    print("Resumo:")
    print(
        dados[
            [
                "rpm",
                "speed_kmh",
                "coolant_temp_c",
                "engine_load_pct",
                "throttle_pct",
                "battery_voltage",
            ]
        ].describe().round(2)
    )

    print()
    print(
        f"Arquivo salvo em: "
        f"{caminho_saida}"
    )