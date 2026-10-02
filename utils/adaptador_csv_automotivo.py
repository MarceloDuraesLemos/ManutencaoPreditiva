from __future__ import annotations

from pathlib import Path
from typing import IO

import pandas as pd


# ============================================================
# SCHEMA AUTOMOTIVO PADRÃO
# ============================================================

COLUNAS_PADRAO = [
    "timestamp",
    "vehicle_id",
    "rpm",
    "speed_kmh",
    "coolant_temp_c",
    "engine_load_pct",
    "throttle_pct",
    "intake_temp_c",
    "map_kpa",
    "maf_g_s",
    "battery_voltage",
    "fuel_level_pct",
]


# Variáveis exigidas pelo Autoencoder v1.
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


# ============================================================
# ALIASES
# ============================================================

# Permite reconhecer nomes comuns sem alterar o schema interno.
ALIASES = {
    # RPM
    "engine_rpm": "rpm",
    "rpm_engine": "rpm",

    # Velocidade
    "speed": "speed_kmh",
    "vehicle_speed": "speed_kmh",
    "vehicle_speed_kmh": "speed_kmh",

    # Temperatura do líquido de arrefecimento
    "coolant_temp": "coolant_temp_c",
    "coolant_temperature": "coolant_temp_c",
    "engine_coolant_temp": "coolant_temp_c",
    "engine_coolant_temperature": "coolant_temp_c",

    # Carga
    "engine_load": "engine_load_pct",
    "load": "engine_load_pct",

    # Borboleta / throttle
    "throttle": "throttle_pct",
    "throttle_position": "throttle_pct",

    # Temperatura de admissão
    "intake_temp": "intake_temp_c",
    "intake_temperature": "intake_temp_c",
    "intake_air_temp": "intake_temp_c",
    "intake_air_temperature": "intake_temp_c",

    # MAP
    "map": "map_kpa",
    "manifold_pressure": "map_kpa",
    "intake_manifold_pressure": "map_kpa",

    # MAF
    "maf": "maf_g_s",
    "mass_air_flow": "maf_g_s",

    # Bateria
    "voltage": "battery_voltage",
    "battery_v": "battery_voltage",

    # Combustível
    "fuel_level": "fuel_level_pct",
    "fuel": "fuel_level_pct",

    # Tempo
    "time": "timestamp",
    "datetime": "timestamp",

    # Veículo
    "vehicle": "vehicle_id",
    "car_id": "vehicle_id",
}


# ============================================================
# NORMALIZAÇÃO DE NOMES
# ============================================================

def normalizar_nome_coluna(
    nome: str,
) -> str:

    nome = str(nome)

    nome = (
        nome
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("%", "pct")
        .replace("°", "")
    )

    while "__" in nome:
        nome = nome.replace(
            "__",
            "_",
        )

    return nome


# ============================================================
# LEITURA
# ============================================================

def ler_csv_automotivo(
    origem: str | Path | IO,
) -> dict:
    """
    Lê e prepara um CSV para o módulo automotivo.

    A função NÃO executa a IA.

    Responsabilidades:
    - ler CSV;
    - normalizar nomes;
    - aplicar aliases conhecidos;
    - converter features da IA para numérico;
    - interpretar timestamp quando disponível;
    - informar compatibilidade com a IA.
    """

    try:
        df = pd.read_csv(
            origem
        )

    except Exception as erro:
        return {
            "sucesso": False,
            "motivo": (
                "Não foi possível ler o arquivo CSV."
            ),
            "erro": str(erro),
            "dados": None,
        }

    if df.empty:
        return {
            "sucesso": False,
            "motivo": (
                "O arquivo CSV está vazio."
            ),
            "dados": None,
        }

    # --------------------------------------------------------
    # Normalizar nomes
    # --------------------------------------------------------

    nomes_normalizados = {
        coluna: normalizar_nome_coluna(
            coluna
        )
        for coluna in df.columns
    }

    df = df.rename(
        columns=nomes_normalizados
    )

    # --------------------------------------------------------
    # Aplicar aliases
    # --------------------------------------------------------

    renomear_aliases = {}

    for coluna in df.columns:

        if coluna in ALIASES:
            destino = ALIASES[
                coluna
            ]

            # Evita sobrescrever uma coluna padrão
            # que já exista no arquivo.
            if destino not in df.columns:
                renomear_aliases[
                    coluna
                ] = destino

    df = df.rename(
        columns=renomear_aliases
    )

    # --------------------------------------------------------
    # Verificar duplicidades depois da padronização
    # --------------------------------------------------------

    duplicadas = (
        df.columns[
            df.columns.duplicated()
        ]
        .tolist()
    )

    if duplicadas:
        return {
            "sucesso": False,
            "motivo": (
                "Após a padronização existem "
                "colunas duplicadas."
            ),
            "colunas_duplicadas": duplicadas,
            "dados": None,
        }

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    timestamp_valido = False

    if "timestamp" in df.columns:

        convertido = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
        )

        if convertido.notna().all():
            df["timestamp"] = convertido

            df = (
                df
                .sort_values(
                    "timestamp"
                )
                .reset_index(
                    drop=True
                )
            )

            timestamp_valido = True

    # --------------------------------------------------------
    # Features da IA
    # --------------------------------------------------------

    features_presentes = [
        feature
        for feature in FEATURES_IA
        if feature in df.columns
    ]

    features_ausentes = [
        feature
        for feature in FEATURES_IA
        if feature not in df.columns
    ]

    colunas_nao_numericas = []

    valores_ausentes = {}

    for feature in features_presentes:

        original = df[
            feature
        ]

        convertido = pd.to_numeric(
            original,
            errors="coerce",
        )

        problema_conversao = (
            original.notna()
            & convertido.isna()
        )

        if problema_conversao.any():
            colunas_nao_numericas.append(
                feature
            )

        quantidade_nan = int(
            convertido
            .isna()
            .sum()
        )

        if quantidade_nan > 0:
            valores_ausentes[
                feature
            ] = quantidade_nan

        df[feature] = convertido

    # --------------------------------------------------------
    # Compatibilidade
    # --------------------------------------------------------

    amostras_suficientes = (
        len(df) >= 30
    )

    ia_compativel = (
        len(features_ausentes) == 0
        and len(colunas_nao_numericas) == 0
        and len(valores_ausentes) == 0
        and amostras_suficientes
    )

    # --------------------------------------------------------
    # Resultado
    # --------------------------------------------------------

    return {
        "sucesso": True,

        "dados": df,

        "linhas": int(
            len(df)
        ),

        "colunas": int(
            len(df.columns)
        ),

        "timestamp_presente": (
            "timestamp"
            in df.columns
        ),

        "timestamp_valido": (
            timestamp_valido
        ),

        "features_ia_presentes": (
            features_presentes
        ),

        "features_ia_ausentes": (
            features_ausentes
        ),

        "colunas_nao_numericas": (
            colunas_nao_numericas
        ),

        "valores_ausentes": (
            valores_ausentes
        ),

        "amostras_suficientes": (
            amostras_suficientes
        ),

        "ia_compativel": (
            ia_compativel
        ),

        "colunas_recebidas": (
            df.columns.tolist()
        ),
    }


# ============================================================
# RESUMO PARA INTERFACE
# ============================================================

def resumir_compatibilidade(
    resultado: dict,
) -> str:

    if not resultado.get(
        "sucesso",
        False,
    ):
        return resultado.get(
            "motivo",
            "Falha na leitura do CSV.",
        )

    if resultado[
        "ia_compativel"
    ]:
        return (
            "CSV compatível com o "
            "modelo automotivo atual."
        )

    problemas = []

    if resultado[
        "features_ia_ausentes"
    ]:
        problemas.append(
            "faltam: "
            + ", ".join(
                resultado[
                    "features_ia_ausentes"
                ]
            )
        )

    if resultado[
        "colunas_nao_numericas"
    ]:
        problemas.append(
            "valores não numéricos em: "
            + ", ".join(
                resultado[
                    "colunas_nao_numericas"
                ]
            )
        )

    if resultado[
        "valores_ausentes"
    ]:
        problemas.append(
            "existem valores ausentes"
        )

    if not resultado[
        "amostras_suficientes"
    ]:
        problemas.append(
            "menos de 30 amostras"
        )

    return (
        "CSV carregado, mas a IA não pode "
        "ser executada: "
        + "; ".join(problemas)
        + "."
    )