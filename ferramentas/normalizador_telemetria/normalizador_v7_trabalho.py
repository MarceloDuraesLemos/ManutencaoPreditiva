from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ============================================================
# CAMINHOS E CONTRATO DO MODELO
# ============================================================

PASTA_ATUAL = Path(__file__).resolve().parent
ARQUIVO_CATALOGO = PASTA_ATUAL / "sensores.json"
ARQUIVO_PERFIS = PASTA_ATUAL / "perfis_scanner.json"

FEATURES_UPX = [
    "rpm",
    "speed_kmh",
    "coolant_temp_c",
    "engine_load_pct",
    "throttle_pct",
    "intake_temp_c",
    "map_kpa",
    "battery_voltage",
]

WINDOW_SIZE = 30
SAMPLING_INTERVAL_SECONDS = 1.0

MAX_CANDIDATOS = 3
LIMITE_CANDIDATO = 0.40

# Tolerância usada somente para DIAGNÓSTICO.
# Ainda não existe resampling/interpolação na V5.
TOLERANCIA_INTERVALO_SEGUNDOS = 0.20
PERCENTUAL_MIN_INTERVALOS_COMPATIVEIS = 0.90


# ============================================================
# ESTRUTURAS
# ============================================================

@dataclass
class Mapeamento:
    feature_upx: str
    coluna_origem: Optional[str]
    confianca: float
    metodo: str
    observacao: str = ""


@dataclass
class Candidato:
    feature_upx: str
    coluna_origem: str
    score: float
    score_textual: float
    positivos: list[str]
    negativos: list[str]
    unidade_detectada: Optional[str]
    unidade_compativel: Optional[bool]
    motivo: str


# ============================================================
# CATÁLOGO
# ============================================================

def carregar_catalogo() -> dict:
    if not ARQUIVO_CATALOGO.exists():
        raise FileNotFoundError(
            f"Catálogo não encontrado:\n{ARQUIVO_CATALOGO}"
        )

    with open(ARQUIVO_CATALOGO, "r", encoding="utf-8") as arquivo:
        catalogo = json.load(arquivo)

    if "features" not in catalogo:
        raise ValueError(
            "sensores.json não possui a chave 'features'."
        )

    faltantes = [
        feature
        for feature in FEATURES_UPX
        if feature not in catalogo["features"]
    ]

    if faltantes:
        raise ValueError(
            "Features ausentes no catálogo: "
            + ", ".join(faltantes)
        )

    return catalogo


# ============================================================
# NORMALIZAÇÃO DE TEXTO
# ============================================================

def normalizar_texto(texto: str) -> str:
    texto = str(texto).strip().lower()

    substituicoes = {
        "á": "a", "à": "a", "ã": "a", "â": "a", "ä": "a",
        "é": "e", "è": "e", "ê": "e", "ë": "e",
        "í": "i", "ì": "i", "î": "i", "ï": "i",
        "ó": "o", "ò": "o", "õ": "o", "ô": "o", "ö": "o",
        "ú": "u", "ù": "u", "û": "u", "ü": "u",
        "ç": "c",
        "°": "",
    }

    for antigo, novo in substituicoes.items():
        texto = texto.replace(antigo, novo)

    texto = texto.replace("_", " ")
    texto = texto.replace("-", " ")
    texto = re.sub(r"[\(\)\[\]\{\}]", " ", texto)
    texto = re.sub(r"\s+", " ", texto)

    return texto.strip()


def normalizar_para_comparacao(texto: str) -> str:
    texto = normalizar_texto(texto)
    texto = texto.replace("%", " percent ")
    texto = texto.replace("/", " ")
    texto = texto.replace("\\", " ")
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto)

    return texto.strip()


# ============================================================
# ENCODING / DELIMITADOR / LOCALIZAÇÃO
# ============================================================

def detectar_encoding(caminho: Path) -> str:
    for encoding in [
        "utf-8-sig",
        "utf-8",
        "cp1252",
        "latin1",
    ]:
        try:
            with open(
                caminho,
                "r",
                encoding=encoding,
            ) as arquivo:
                arquivo.read(4096)

            return encoding

        except UnicodeDecodeError:
            continue

    return "latin1"


def detectar_delimitador(
    caminho: Path,
    encoding: str,
) -> str:
    with open(
        caminho,
        "r",
        encoding=encoding,
        errors="replace",
        newline="",
    ) as arquivo:
        amostra = arquivo.read(8192)

    if not amostra.strip():
        raise ValueError("O arquivo está vazio.")

    try:
        dialect = csv.Sniffer().sniff(
            amostra,
            delimiters=",;\t|",
        )

        return dialect.delimiter

    except csv.Error:
        primeira_linha = amostra.splitlines()[0]
        candidatos = [",", ";", "\t", "|"]

        return max(
            candidatos,
            key=primeira_linha.count,
        )


def localizar_csvs(
    caminho: str | Path,
) -> list[Path]:
    caminho = Path(caminho)

    if not caminho.exists():
        raise FileNotFoundError(
            f"Caminho não encontrado: {caminho}"
        )

    if caminho.is_file():
        if caminho.suffix.lower() != ".csv":
            raise ValueError(
                "O arquivo informado não é CSV."
            )

        return [caminho]

    if caminho.is_dir():
        arquivos = sorted(
            arquivo
            for arquivo in caminho.rglob("*")
            if arquivo.is_file()
            and arquivo.suffix.lower() == ".csv"
            and "_normalizado_upx" not in arquivo.stem.lower()
            and "_regularizado_1hz_upx" not in arquivo.stem.lower()
        )

        if not arquivos:
            raise ValueError(
                "Nenhum CSV foi encontrado nessa pasta."
            )

        return arquivos

    raise ValueError("Caminho inválido.")


# ============================================================
# LEITURA
# ============================================================

def ler_csv(
    caminho: Path,
) -> tuple[pd.DataFrame, dict]:
    encoding = detectar_encoding(caminho)
    delimitador = detectar_delimitador(
        caminho,
        encoding,
    )

    df = pd.read_csv(
        caminho,
        sep=delimitador,
        encoding=encoding,
        dtype=str,
        low_memory=False,
    )

    # Corrige nomes como " Device Time".
    df.columns = [
        str(coluna).strip()
        for coluna in df.columns
    ]

    metadados = {
        "arquivo": caminho.name,
        "caminho": str(caminho),
        "linhas_brutas": len(df),
        "colunas": len(df.columns),
        "encoding": encoding,
        "delimitador": repr(delimitador),
    }

    return df, metadados


# ============================================================
# LIMPEZA ESTRUTURAL - NOVO NA V5
# ============================================================

def detectar_cabecalhos_repetidos(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Detecta linhas que são, na prática, uma repetição do cabeçalho.

    Não depende de um scanner específico.

    Para cada linha, compara os valores com os nomes das respectivas
    colunas. Uma linha é considerada cabeçalho repetido quando existe
    forte correspondência entre os dois.

    Exemplo real:
        GPS Time | Device Time | Longitude | ...
        GPS Time | Device Time | Longitude | ...
    """

    if df.empty:
        return pd.Series(
            False,
            index=df.index,
            dtype=bool,
        )

    colunas_normalizadas = {
        coluna: normalizar_para_comparacao(coluna)
        for coluna in df.columns
    }

    def linha_e_cabecalho(linha: pd.Series) -> bool:
        comparaveis = 0
        iguais = 0

        for coluna in df.columns:
            valor = linha[coluna]

            if pd.isna(valor):
                continue

            valor_norm = normalizar_para_comparacao(
                valor
            )

            coluna_norm = colunas_normalizadas[
                coluna
            ]

            if not valor_norm:
                continue

            comparaveis += 1

            if valor_norm == coluna_norm:
                iguais += 1

        # Exigimos pelo menos duas correspondências.
        # Isso evita classificar uma linha como cabeçalho
        # apenas porque um valor coincidiu por acaso.
        if comparaveis < 2:
            return False

        proporcao = iguais / comparaveis

        return iguais >= 2 and proporcao >= 0.50

    return df.apply(
        linha_e_cabecalho,
        axis=1,
    )


def limpar_estrutura(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    linhas_originais = len(df)

    # Linhas completamente vazias.
    mascara_vazias = df.apply(
        lambda linha: all(
            pd.isna(valor)
            or str(valor).strip() == ""
            for valor in linha
        ),
        axis=1,
    )

    quantidade_vazias = int(
        mascara_vazias.sum()
    )

    df_sem_vazias = df.loc[
        ~mascara_vazias
    ].copy()

    # Cabeçalhos repetidos.
    mascara_cabecalho = (
        detectar_cabecalhos_repetidos(
            df_sem_vazias
        )
    )

    quantidade_cabecalhos = int(
        mascara_cabecalho.sum()
    )

    indices_cabecalhos = [
        int(indice)
        if isinstance(indice, (int, np.integer))
        else str(indice)
        for indice
        in df_sem_vazias.index[
            mascara_cabecalho
        ].tolist()
    ]

    df_limpo = df_sem_vazias.loc[
        ~mascara_cabecalho
    ].copy()

    # O índice original deixa de ter significado após
    # a limpeza estrutural.
    df_limpo.reset_index(
        drop=True,
        inplace=True,
    )

    relatorio = {
        "linhas_originais": linhas_originais,
        "linhas_vazias_removidas": quantidade_vazias,
        "cabecalhos_repetidos_removidos": quantidade_cabecalhos,
        "indices_cabecalhos_repetidos": indices_cabecalhos,
        "linhas_restantes": len(df_limpo),
    }

    return df_limpo, relatorio


# ============================================================
# LIMPEZA DOS VALORES
# ============================================================

VALORES_AUSENTES = {
    "",
    "-",
    "--",
    "---",
    "n/a",
    "na",
    "nan",
    "null",
    "none",
    "∞",
    "+∞",
    "-∞",
    "inf",
    "+inf",
    "-inf",
}


def limpar_valor(valor):
    if pd.isna(valor):
        return np.nan

    texto = str(valor).strip()

    if texto.lower() in VALORES_AUSENTES:
        return np.nan

    return texto


def limpar_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:
    resultado = df.copy()

    for coluna in resultado.columns:
        resultado[coluna] = (
            resultado[coluna].map(
                limpar_valor
            )
        )

    return resultado


def converter_colunas_numericas(
    df: pd.DataFrame,
) -> pd.DataFrame:
    resultado = df.copy()

    for coluna in resultado.columns:
        serie_original = resultado[coluna]

        tentativa_ponto = pd.to_numeric(
            serie_original,
            errors="coerce",
        )

        tentativa_virgula = pd.to_numeric(
            serie_original.astype(str)
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False),
            errors="coerce",
        )

        validos_ponto = (
            tentativa_ponto.notna().sum()
        )

        validos_virgula = (
            tentativa_virgula.notna().sum()
        )

        melhor = (
            tentativa_virgula
            if validos_virgula > validos_ponto
            else tentativa_ponto
        )

        total_nao_nulo = (
            serie_original.notna().sum()
        )

        if total_nao_nulo == 0:
            continue

        if (
            melhor.notna().sum()
            / total_nao_nulo
            >= 0.80
        ):
            resultado[coluna] = melhor

    return resultado



# ============================================================
# INVENTÁRIO INTELIGENTE DE SENSORES - NOVO NA V7
# ============================================================

def classificar_grandeza_coluna(
    coluna: str,
) -> str:
    """
    Classificação descritiva e conservadora da coluna.

    Esta função NÃO mapeia a coluna para uma feature do modelo.
    Ela apenas descreve a natureza aparente do dado para o
    inventário da V7.
    """
    nome = normalizar_para_comparacao(coluna)
    tokens = set(nome.split())

    regras = [
        (
            "tempo",
            [
                "time", "timestamp", "datetime",
                "date time", "data hora",
            ],
        ),
        (
            "localizacao",
            [
                "latitude", "longitude", "altitude",
                "bearing", "satellite", "satellites",
                "gps accuracy",
            ],
        ),
        (
            "rotacao",
            ["rpm", "rev min", "engine speed"],
        ),
        (
            "velocidade",
            [
                "speed", "vehicle speed",
                "gps speed", "velocity",
            ],
        ),
        (
            "temperatura",
            [
                "temperature", "temp", "coolant",
                "intake air", "charge air cooler",
                "cact",
            ],
        ),
        (
            "pressao",
            [
                "pressure", "map", "boost",
                "manifold absolute",
            ],
        ),
        (
            "tensao",
            [
                "voltage", "volt", "battery",
            ],
        ),
        (
            "carga",
            [
                "engine load", "absolute load",
                "load value",
            ],
        ),
        (
            "posicao_acelerador",
            [
                "throttle", "accelerator pedal",
                "pedal position",
            ],
        ),
        (
            "aceleracao",
            [
                "acceleration sensor",
                "acceleration", "g x", "g y", "g z",
            ],
        ),
        (
            "combustivel",
            [
                "fuel", "fuel trim", "consumption",
            ],
        ),
        (
            "torque",
            ["torque"],
        ),
        (
            "potencia",
            ["power", "horsepower", " hp ", "kw"],
        ),
    ]

    nome_com_espacos = f" {nome} "

    for grandeza, termos in regras:
        for termo in termos:
            termo_norm = normalizar_para_comparacao(termo)

            if not termo_norm:
                continue

            if " " in termo_norm:
                if termo_norm in nome:
                    return grandeza
            elif (
                termo_norm in tokens
                or f" {termo_norm} " in nome_com_espacos
            ):
                return grandeza

    return "nao_classificada"


def avaliar_equivalencia_fisica_modelo(
    coluna: str,
    grandeza: str,
    unidade: Optional[str],
) -> dict:
    """Avalia somente compatibilidade física potencial com o contrato UPX.

    V7.2 / Bloco 2A: esta função NÃO seleciona fonte, NÃO altera o
    mapeamento confirmado e NÃO envia nenhuma coluna nova ao modelo.
    Ela apenas registra equivalências físicas conservadoras e conversões
    conhecidas para uso futuro pelo seletor de fontes.
    """
    nome = normalizar_para_comparacao(coluna)

    def resultado(feature: str, conversao: str) -> dict:
        return {
            "compativel": True,
            "features": [feature],
            "conversao": conversao,
        }

    # Velocidade é uma grandeza sem ambiguidade dentro do contrato atual.
    if grandeza == "velocidade":
        if unidade == "km/h":
            return resultado("speed_kmh", "km/h -> km/h (sem conversão)")
        if unidade == "mph":
            return resultado("speed_kmh", "mph x 1.609344 -> km/h")
        if unidade == "m/s":
            return resultado("speed_kmh", "m/s x 3.6 -> km/h")

    # Rotação: somente unidades próprias de rotação.
    if grandeza == "rotacao" and unidade in {"rpm", "rev/min"}:
        return resultado("rpm", f"{unidade} -> rpm")

    # Posição do acelerador: exige grandeza já identificada e percentual.
    if grandeza == "posicao_acelerador" and unidade == "%":
        return resultado("throttle_pct", "% -> % (sem conversão)")

    # Carga do motor: evita promover o PID 'absolute engine load', que é
    # semanticamente distinto do Engine Load usado pelo contrato atual.
    if (
        grandeza == "carga"
        and unidade == "%"
        and "engine load" in nome
        and "absolute" not in nome
    ):
        return resultado("engine_load_pct", "% -> % (sem conversão)")

    # Temperaturas exigem localização física explícita. CACT/catalisador
    # não podem virar coolant ou intake por compartilharem °C/°F.
    if grandeza == "temperatura" and unidade in {"C", "F"}:
        conversao = (
            "°C -> °C (sem conversão)"
            if unidade == "C"
            else "(°F - 32) x 5/9 -> °C"
        )
        if "coolant" in nome:
            return resultado("coolant_temp_c", conversao)
        if (
            "intake air" in nome
            or "intake temperature" in nome
            or "iat" in nome.split()
        ):
            return resultado("intake_temp_c", conversao)

    # MAP exige referência explícita ao manifold absolute pressure/MAP.
    if grandeza == "pressao" and unidade in {"kPa", "Pa", "bar", "psi"}:
        if "manifold absolute" in nome or "map" in nome.split():
            conversoes = {
                "kPa": "kPa -> kPa (sem conversão)",
                "Pa": "Pa / 1000 -> kPa",
                "bar": "bar x 100 -> kPa",
                "psi": "psi x 6.894757293 -> kPa",
            }
            return resultado("map_kpa", conversoes[unidade])

    # Tensão: somente uma origem explicitamente ligada ao veículo/OBD.
    # Ex.: 'Android device Battery Level(%)' jamais é promovida.
    if grandeza == "tensao" and unidade in {"V", "mV"}:
        if any(termo in nome for termo in ["obd", "adapter", "vehicle", "ecu"]):
            conversao = (
                "V -> V (sem conversão)"
                if unidade == "V"
                else "mV / 1000 -> V"
            )
            return resultado("battery_voltage", conversao)

    return {
        "compativel": False,
        "features": [],
        "conversao": "-",
    }


def construir_inventario_sensores(
    df: pd.DataFrame,
    mapeamentos: list[Mapeamento],
    candidatos: dict[str, list[Candidato]],
) -> pd.DataFrame:
    """
    Gera o inventário descritivo de TODAS as colunas do CSV.

    O inventário não altera o mapeamento da V6/V7 e não escolhe
    candidatos. Ele apenas registra:
    - disponibilidade real de dados;
    - completude;
    - unidade detectada;
    - grandeza aparente;
    - relação atual com o contrato do modelo.

    Status possíveis neste primeiro bloco:
    CONFIRMADO
    CANDIDATO
    FORA_DO_CONTRATO
    SEM_DADOS
    DESCONHECIDO

    AMBÍGUO será usado nos próximos blocos da V7 quando a seleção
    inteligente de fontes for implementada.
    """
    mapa_confirmados = {
        item.coluna_origem: item.feature_upx
        for item in mapeamentos
        if item.coluna_origem is not None
    }

    mapa_candidatos: dict[str, list[str]] = {}

    for feature, lista in candidatos.items():
        for candidato in lista:
            mapa_candidatos.setdefault(
                candidato.coluna_origem,
                [],
            ).append(feature)

    registros = []
    total_linhas = len(df)

    for coluna in df.columns:
        serie = df[coluna]
        dados_validos = int(
            serie.notna().sum()
        )

        completude = (
            dados_validos / total_linhas * 100
            if total_linhas
            else 0.0
        )

        unidade = detectar_unidade(coluna)
        grandeza = classificar_grandeza_coluna(
            coluna
        )

        equivalencia_fisica = avaliar_equivalencia_fisica_modelo(
            coluna=coluna,
            grandeza=grandeza,
            unidade=unidade,
        )

        feature_confirmada = (
            mapa_confirmados.get(coluna)
        )

        features_candidatas = (
            mapa_candidatos.get(
                coluna,
                [],
            )
        )

        if dados_validos == 0:
            status = "SEM_DADOS"
            relacao = (
                feature_confirmada
                or ", ".join(features_candidatas)
                or "-"
            )

        elif feature_confirmada is not None:
            status = "CONFIRMADO"
            relacao = feature_confirmada

        elif features_candidatas:
            status = "CANDIDATO"
            relacao = ", ".join(
                features_candidatas
            )

        elif grandeza != "nao_classificada":
            status = "FORA_DO_CONTRATO"
            relacao = "-"

        else:
            status = "DESCONHECIDO"
            relacao = "-"

        registros.append(
            {
                "coluna_original": coluna,
                "grandeza": grandeza,
                "unidade": unidade or "-",
                "dados_validos": dados_validos,
                "total_linhas": total_linhas,
                "completude_pct": round(
                    completude,
                    2,
                ),
                "status": status,
                "feature_upx_relacionada": relacao,
                "compatibilidade_fisica": (
                    "SIM" if equivalencia_fisica["compativel"] else "NÃO"
                ),
                "features_fisicamente_compativeis": (
                    ", ".join(equivalencia_fisica["features"])
                    if equivalencia_fisica["features"]
                    else "-"
                ),
                "conversao_fisica_disponivel": equivalencia_fisica["conversao"],
            }
        )

    return pd.DataFrame(registros)


def resumir_inventario(
    inventario: pd.DataFrame,
) -> dict:
    if inventario.empty:
        return {
            "total_colunas": 0,
            "com_dados": 0,
            "sem_dados": 0,
            "confirmadas": 0,
            "candidatas": 0,
            "fora_do_contrato": 0,
            "desconhecidas": 0,
        }

    status = inventario["status"]

    return {
        "total_colunas": len(inventario),
        "com_dados": int(
            (inventario["dados_validos"] > 0).sum()
        ),
        "sem_dados": int(
            (status == "SEM_DADOS").sum()
        ),
        "confirmadas": int(
            (status == "CONFIRMADO").sum()
        ),
        "candidatas": int(
            (status == "CANDIDATO").sum()
        ),
        "fora_do_contrato": int(
            (status == "FORA_DO_CONTRATO").sum()
        ),
        "desconhecidas": int(
            (status == "DESCONHECIDO").sum()
        ),
    }


# ============================================================
# UNIDADES
# ============================================================

def detectar_unidade(
    nome_coluna: str,
) -> Optional[str]:
    texto = str(nome_coluna).strip().lower()

    padroes = [
        ("rev/min", ["rev/min", "rev min"]),
        ("rpm", ["rpm"]),
        ("km/h", ["km/h", "kmh", "kph"]),
        ("mph", ["mph"]),
        ("m/s", ["m/s", "mps", "meters/second", "meter/second", "meters per second", "meter per second"]),
        ("kPa", ["kpa"]),
        ("psi", ["psi"]),
        ("bar", ["bar"]),
        ("mV", ["(mv)", "[mv]"]),
        ("V", ["(v)", "[v]"]),
        ("%", ["%", "percent"]),
        ("F", ["°f", "(f)", "[f]"]),
        ("C", ["°c", "(c)", "[c]"]),
        ("Pa", ["(pa)", "[pa]"]),
    ]

    for unidade, termos in padroes:
        for termo in termos:
            if termo in texto:
                return unidade

    return None


def avaliar_unidade(
    coluna: str,
    configuracao: dict,
) -> tuple[Optional[str], Optional[bool]]:
    unidade = detectar_unidade(coluna)

    if unidade is None:
        return None, None

    aceitas = configuracao.get(
        "unidades_aceitas",
        [],
    )

    return unidade, unidade in aceitas


# ============================================================
# ALIAS EXATO
# ============================================================

def coluna_corresponde_alias(
    coluna: str,
    aliases: list[str],
) -> bool:
    coluna_original = str(coluna).strip()

    coluna_completa = (
        normalizar_para_comparacao(
            coluna_original
        )
    )

    for alias in aliases:
        if (
            coluna_completa
            == normalizar_para_comparacao(alias)
        ):
            return True

    padrao_unidade_final = re.compile(
        r"""
        \s*
        [\(\[]
        \s*
        (
            rpm
            |
            rev\s*/\s*min
            |
            r\s*/\s*min
            |
            km\s*/\s*h
            |
            kmh
            |
            kph
            |
            mph
            |
            m\s*/\s*s
            |
            kpa
            |
            pa
            |
            psi
            |
            bar
            |
            mv
            |
            v
            |
            %
            |
            °?\s*c
            |
            °?\s*f
        )
        \s*
        [\)\]]
        \s*$
        """,
        flags=re.IGNORECASE | re.VERBOSE,
    )

    coluna_sem_unidade = re.sub(
        padrao_unidade_final,
        "",
        coluna_original,
    ).strip()

    coluna_sem_unidade_norm = (
        normalizar_para_comparacao(
            coluna_sem_unidade
        )
    )

    for alias in aliases:
        if (
            coluna_sem_unidade_norm
            == normalizar_para_comparacao(alias)
        ):
            return True

    return False


# ============================================================
# SEMÂNTICA
# ============================================================

def similaridade_textual(
    texto_a: str,
    texto_b: str,
) -> float:
    a = normalizar_para_comparacao(texto_a)
    b = normalizar_para_comparacao(texto_b)

    if not a or not b:
        return 0.0

    score_sequencia = SequenceMatcher(
        None,
        a,
        b,
    ).ratio()

    tokens_a = set(a.split())
    tokens_b = set(b.split())

    score_tokens = (
        len(tokens_a & tokens_b)
        / len(tokens_a | tokens_b)
        if tokens_a and tokens_b
        else 0.0
    )

    return (
        score_sequencia * 0.55
        + score_tokens * 0.45
    )


def contem_termo(
    texto: str,
    termo: str,
) -> bool:
    texto_norm = (
        normalizar_para_comparacao(texto)
    )

    termo_norm = (
        normalizar_para_comparacao(termo)
    )

    if not termo_norm:
        return False

    if " " in termo_norm:
        return termo_norm in texto_norm

    return termo_norm in set(
        texto_norm.split()
    )


def analisar_candidato(
    feature: str,
    coluna: str,
    configuracao: dict,
) -> Candidato:
    aliases = configuracao.get(
        "aliases",
        [],
    )

    positivos_config = configuracao.get(
        "termos_positivos",
        [],
    )

    negativos_config = configuracao.get(
        "termos_negativos",
        [],
    )

    melhor_textual = max(
        (
            similaridade_textual(
                coluna,
                alias,
            )
            for alias in aliases
        ),
        default=0.0,
    )

    positivos = [
        termo
        for termo in positivos_config
        if contem_termo(
            coluna,
            termo,
        )
    ]

    negativos = [
        termo
        for termo in negativos_config
        if contem_termo(
            coluna,
            termo,
        )
    ]

    unidade, unidade_compativel = (
        avaliar_unidade(
            coluna,
            configuracao,
        )
    )

    score = melhor_textual * 0.55

    if positivos_config:
        proporcao = min(
            len(positivos)
            / max(
                min(
                    len(positivos_config),
                    3,
                ),
                1,
            ),
            1.0,
        )

        score += proporcao * 0.25

    if unidade_compativel is True:
        score += 0.20

    elif unidade_compativel is False:
        score -= 0.35

    if negativos:
        score -= min(
            0.55,
            0.30 * len(negativos),
        )

    score = max(
        0.0,
        min(score, 1.0),
    )

    motivos = []

    if positivos:
        motivos.append(
            "termos positivos: "
            + ", ".join(positivos)
        )

    if negativos:
        motivos.append(
            "termos conflitantes: "
            + ", ".join(negativos)
        )

    if unidade is None:
        motivos.append(
            "unidade não identificada"
        )

    elif unidade_compativel:
        motivos.append(
            f"unidade compatível: {unidade}"
        )

    else:
        motivos.append(
            f"unidade incompatível: {unidade}"
        )

    return Candidato(
        feature_upx=feature,
        coluna_origem=coluna,
        score=score,
        score_textual=melhor_textual,
        positivos=positivos,
        negativos=negativos,
        unidade_detectada=unidade,
        unidade_compativel=unidade_compativel,
        motivo="; ".join(motivos),
    )


def encontrar_candidatos(
    df: pd.DataFrame,
    feature: str,
    configuracao: dict,
    colunas_ja_usadas: set[str],
) -> list[Candidato]:
    candidatos = []

    for coluna in df.columns:
        if coluna in colunas_ja_usadas:
            continue

        candidato = analisar_candidato(
            feature,
            coluna,
            configuracao,
        )

        if (
            candidato.unidade_compativel
            is False
        ):
            continue

        if candidato.negativos:
            continue

        if (
            candidato.score
            >= LIMITE_CANDIDATO
        ):
            candidatos.append(
                candidato
            )

    candidatos.sort(
        key=lambda item: item.score,
        reverse=True,
    )

    return candidatos[:MAX_CANDIDATOS]


# ============================================================
# MAPEAMENTO
# ============================================================

def mapear_features(
    df: pd.DataFrame,
    catalogo: dict,
) -> tuple[
    list[Mapeamento],
    dict[str, list[Candidato]],
]:
    resultados = []
    candidatos_por_feature = {}
    colunas_ja_usadas = set()

    configuracoes = catalogo["features"]

    for feature in FEATURES_UPX:
        configuracao = configuracoes[
            feature
        ]

        aliases = configuracao.get(
            "aliases",
            [],
        )

        encontrados = [
            coluna
            for coluna in df.columns
            if coluna_corresponde_alias(
                coluna,
                aliases,
            )
        ]

        if len(encontrados) == 1:
            coluna = encontrados[0]

            unidade, compativel = (
                avaliar_unidade(
                    coluna,
                    configuracao,
                )
            )

            if compativel is False:
                resultados.append(
                    Mapeamento(
                        feature,
                        None,
                        0.0,
                        "unidade_incompativel",
                        (
                            f"Alias reconhecido em "
                            f"'{coluna}', mas unidade "
                            f"'{unidade}' é incompatível."
                        ),
                    )
                )

            else:
                colunas_ja_usadas.add(
                    coluna
                )

                resultados.append(
                    Mapeamento(
                        feature,
                        coluna,
                        1.0,
                        "alias_exato",
                        (
                            "Correspondência conhecida "
                            "pelo catálogo."
                        ),
                    )
                )

        elif len(encontrados) > 1:
            resultados.append(
                Mapeamento(
                    feature,
                    None,
                    0.0,
                    "ambiguo",
                    (
                        "Mais de uma correspondência "
                        "exata: "
                        + ", ".join(encontrados)
                    ),
                )
            )

        else:
            resultados.append(
                Mapeamento(
                    feature,
                    None,
                    0.0,
                    "nao_encontrado",
                    (
                        "Nenhuma correspondência segura "
                        "foi encontrada."
                    ),
                )
            )

    for item in resultados:
        if item.coluna_origem is not None:
            continue

        candidatos_por_feature[
            item.feature_upx
        ] = encontrar_candidatos(
            df,
            item.feature_upx,
            configuracoes[
                item.feature_upx
            ],
            colunas_ja_usadas,
        )

    return (
        resultados,
        candidatos_por_feature,
    )


# ============================================================
# CONVERSÃO DE UNIDADES
# ============================================================

def aplicar_conversao(
    serie: pd.Series,
    unidade_origem: Optional[str],
    configuracao: dict,
) -> tuple[pd.Series, bool, str]:
    serie = pd.to_numeric(
        serie,
        errors="coerce",
    )

    unidade_destino = configuracao.get(
        "unidade_padrao"
    )

    if unidade_origem is None:
        return (
            serie,
            False,
            (
                "Unidade não identificada; "
                "valores preservados."
            ),
        )

    if unidade_origem == unidade_destino:
        return (
            serie,
            False,
            (
                f"{unidade_origem} -> "
                f"{unidade_destino} "
                "(sem conversão)"
            ),
        )

    conversoes = configuracao.get(
        "conversoes",
        {},
    )

    regra = conversoes.get(
        unidade_origem
    )

    if regra is None:
        return (
            serie,
            False,
            (
                "Nenhuma regra de conversão "
                f"cadastrada para "
                f"{unidade_origem} -> "
                f"{unidade_destino}."
            ),
        )

    operacao = regra.get(
        "operacao"
    )

    if operacao == "multiplicar":
        valor = float(
            regra["valor"]
        )

        convertido = serie * valor

        return (
            convertido,
            True,
            (
                f"{unidade_origem} -> "
                f"{unidade_destino}; "
                f"multiplicar por {valor}"
            ),
        )

    if operacao == "dividir":
        valor = float(
            regra["valor"]
        )

        convertido = serie / valor

        return (
            convertido,
            True,
            (
                f"{unidade_origem} -> "
                f"{unidade_destino}; "
                f"dividir por {valor}"
            ),
        )

    if (
        operacao
        == "fahrenheit_para_celsius"
    ):
        convertido = (
            (serie - 32.0)
            * (5.0 / 9.0)
        )

        return (
            convertido,
            True,
            "F -> C; (valor - 32) * 5/9",
        )

    raise ValueError(
        "Operação de conversão "
        f"desconhecida: {operacao}"
    )


# ============================================================
# PLAUSIBILIDADE
# ============================================================

def analisar_plausibilidade(
    serie: pd.Series,
    configuracao: dict,
) -> dict:
    serie = pd.to_numeric(
        serie,
        errors="coerce",
    ).dropna()

    if serie.empty:
        return {
            "avaliavel": False,
            "min_observado": None,
            "max_observado": None,
            "percentual_plausivel": None,
        }

    faixa = configuracao.get(
        "faixa_plausivel"
    )

    if not faixa:
        return {
            "avaliavel": True,
            "min_observado": float(
                serie.min()
            ),
            "max_observado": float(
                serie.max()
            ),
            "percentual_plausivel": None,
        }

    dentro = serie.between(
        faixa["min"],
        faixa["max"],
        inclusive="both",
    )

    return {
        "avaliavel": True,
        "min_observado": float(
            serie.min()
        ),
        "max_observado": float(
            serie.max()
        ),
        "percentual_plausivel": round(
            float(
                dentro.mean() * 100
            ),
            2,
        ),
    }


# ============================================================
# DATAFRAME PADRÃO UPX
# ============================================================

def construir_dataframe_upx(
    df: pd.DataFrame,
    mapeamentos: list[Mapeamento],
    catalogo: dict,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    df_upx = pd.DataFrame(
        index=df.index
    )

    transformacoes = []

    for item in mapeamentos:
        if item.coluna_origem is None:
            continue

        feature = item.feature_upx
        origem = item.coluna_origem

        configuracao = (
            catalogo["features"][
                feature
            ]
        )

        unidade_origem = (
            detectar_unidade(origem)
        )

        unidade_destino = (
            configuracao.get(
                "unidade_padrao"
            )
        )

        (
            serie_convertida,
            converteu,
            descricao,
        ) = aplicar_conversao(
            df[origem],
            unidade_origem,
            configuracao,
        )

        df_upx[feature] = (
            serie_convertida
        )

        plausibilidade = (
            analisar_plausibilidade(
                serie_convertida,
                configuracao,
            )
        )

        completude = (
            float(
                serie_convertida
                .notna()
                .mean()
                * 100
            )
            if len(serie_convertida)
            else 0.0
        )

        transformacoes.append(
            {
                "feature_upx": feature,
                "coluna_origem": origem,
                "unidade_origem": (
                    unidade_origem
                ),
                "unidade_destino": (
                    unidade_destino
                ),
                "conversao_aplicada": (
                    "SIM"
                    if converteu
                    else "NÃO"
                ),
                "transformacao": descricao,
                "completude_pct": round(
                    completude,
                    2,
                ),
                "plausibilidade_pct": (
                    plausibilidade[
                        "percentual_plausivel"
                    ]
                ),
                "min_observado": (
                    plausibilidade[
                        "min_observado"
                    ]
                ),
                "max_observado": (
                    plausibilidade[
                        "max_observado"
                    ]
                ),
            }
        )

    return (
        df_upx,
        pd.DataFrame(
            transformacoes
        ),
    )


# ============================================================
# TEMPO - NOVO NA V5
# ============================================================

MESES_PT = {
    "jan.": "01",
    "fev.": "02",
    "mar.": "03",
    "abr.": "04",
    "mai.": "05",
    "jun.": "06",
    "jul.": "07",
    "ago.": "08",
    "set.": "09",
    "out.": "10",
    "nov.": "11",
    "dez.": "12",
}


def localizar_colunas_temporais(
    df: pd.DataFrame,
) -> list[str]:
    """
    Localiza possíveis colunas de timestamp.

    A prioridade é determinada separadamente.
    """

    candidatos = []

    nomes_exatos = {
        "device time",
        "gps time",
        "timestamp",
        "datetime",
        "date time",
        "data hora",
        "data e hora",
    }

    for coluna in df.columns:
        normalizada = (
            normalizar_para_comparacao(
                coluna
            )
        )

        if normalizada in nomes_exatos:
            candidatos.append(
                coluna
            )
            continue

        tokens = set(
            normalizada.split()
        )

        if "timestamp" in tokens:
            candidatos.append(
                coluna
            )
            continue

        if (
            "time" in tokens
            and (
                "device" in tokens
                or "gps" in tokens
            )
        ):
            candidatos.append(
                coluna
            )

    return candidatos


def prioridade_coluna_temporal(
    coluna: str,
) -> int:
    nome = normalizar_para_comparacao(
        coluna
    )

    prioridades = {
        "device time": 100,
        "timestamp": 90,
        "datetime": 85,
        "date time": 80,
        "data hora": 80,
        "data e hora": 80,
        "gps time": 70,
    }

    return prioridades.get(
        nome,
        50,
    )


def substituir_meses_portugueses(
    serie: pd.Series,
) -> pd.Series:
    resultado = (
        serie.astype("string")
        .str.strip()
        .str.lower()
    )

    for mes, numero in MESES_PT.items():
        resultado = (
            resultado.str.replace(
                mes,
                numero,
                regex=False,
            )
        )

    return resultado


def tentar_parse_device_time(
    serie: pd.Series,
) -> pd.Series:
    """
    Parser explícito para formatos como:

    02-out.-2026 07:02:30.380

    Depois da substituição:
    02-10-2026 07:02:30.380
    """

    texto = (
        substituir_meses_portugueses(
            serie
        )
    )

    formatos = [
        "%d-%m-%Y %H:%M:%S.%f",
        "%d-%m-%Y %H:%M:%S",
    ]

    melhor = pd.Series(
        pd.NaT,
        index=serie.index,
        dtype="datetime64[ns]",
    )

    melhor_validos = -1

    for formato in formatos:
        tentativa = pd.to_datetime(
            texto,
            format=formato,
            errors="coerce",
        )

        validos = int(
            tentativa.notna().sum()
        )

        if validos > melhor_validos:
            melhor = tentativa
            melhor_validos = validos

    return melhor


def tentar_parse_gps_time(
    serie: pd.Series,
) -> pd.Series:
    """
    Parser para formato semelhante a:

    Fri Oct 02 07:02:32 GMT-03:00 2026

    Não usamos o GPS Time como primeira escolha quando
    Device Time válido está disponível.
    """

    texto = (
        serie.astype("string")
        .str.strip()
    )

    # Remove dia da semana.
    texto = texto.str.replace(
        r"^[A-Za-z]{3}\s+",
        "",
        regex=True,
    )

    # Remove o prefixo GMT, preservando o offset.
    texto = texto.str.replace(
        "GMT",
        "",
        regex=False,
    )

    # Mês inglês numérico para evitar dependência de locale.
    meses_en = {
        "Jan": "01",
        "Feb": "02",
        "Mar": "03",
        "Apr": "04",
        "May": "05",
        "Jun": "06",
        "Jul": "07",
        "Aug": "08",
        "Sep": "09",
        "Oct": "10",
        "Nov": "11",
        "Dec": "12",
    }

    for mes, numero in meses_en.items():
        texto = texto.str.replace(
            mes,
            numero,
            regex=False,
        )

    # Após transformação:
    # 10 02 07:02:32 -03:00 2026
    tentativa = pd.to_datetime(
        texto,
        format="%m %d %H:%M:%S %z %Y",
        errors="coerce",
        utc=True,
    )

    return tentativa


def tentar_parse_generico(
    serie: pd.Series,
) -> pd.Series:
    """
    Fallback conservador.

    O parser explícito é sempre preferido.
    """

    texto = (
        substituir_meses_portugueses(
            serie
        )
    )

    return pd.to_datetime(
        texto,
        errors="coerce",
    )


def interpretar_coluna_temporal(
    serie: pd.Series,
    nome_coluna: str,
) -> pd.Series:
    nome = normalizar_para_comparacao(
        nome_coluna
    )

    if nome == "device time":
        return tentar_parse_device_time(
            serie
        )

    if nome == "gps time":
        return tentar_parse_gps_time(
            serie
        )

    return tentar_parse_generico(
        serie
    )


def escolher_coluna_temporal(
    df: pd.DataFrame,
) -> tuple[
    Optional[str],
    Optional[pd.Series],
    list[dict],
]:
    candidatos = localizar_colunas_temporais(
        df
    )

    avaliacoes = []

    for coluna in candidatos:
        timestamps = (
            interpretar_coluna_temporal(
                df[coluna],
                coluna,
            )
        )

        validos = int(
            timestamps.notna().sum()
        )

        percentual = (
            validos / len(df) * 100
            if len(df)
            else 0.0
        )

        avaliacoes.append(
            {
                "coluna": coluna,
                "prioridade": (
                    prioridade_coluna_temporal(
                        coluna
                    )
                ),
                "validos": validos,
                "percentual_valido": percentual,
                "timestamps": timestamps,
            }
        )

    if not avaliacoes:
        return None, None, []

    # Primeiro buscamos colunas com boa cobertura.
    # Entre elas, respeitamos a prioridade semântica.
    boas = [
        item
        for item in avaliacoes
        if item["percentual_valido"] >= 90.0
    ]

    if boas:
        escolhido = max(
            boas,
            key=lambda item: (
                item["prioridade"],
                item["percentual_valido"],
            ),
        )

    else:
        # Se nenhuma possui 90%, preferimos a que realmente
        # conseguiu interpretar mais timestamps.
        escolhido = max(
            avaliacoes,
            key=lambda item: (
                item["percentual_valido"],
                item["prioridade"],
            ),
        )

    return (
        escolhido["coluna"],
        escolhido["timestamps"],
        avaliacoes,
    )


def diagnosticar_tempo(
    df: pd.DataFrame,
) -> dict:
    (
        coluna,
        timestamps,
        avaliacoes,
    ) = escolher_coluna_temporal(df)

    resumo_avaliacoes = [
        {
            "coluna": item["coluna"],
            "validos": item["validos"],
            "percentual_valido": round(
                item["percentual_valido"],
                2,
            ),
        }
        for item in avaliacoes
    ]

    if (
        coluna is None
        or timestamps is None
    ):
        return {
            "coluna_temporal": None,
            "colunas_avaliadas": resumo_avaliacoes,
            "timestamps_validos": 0,
            "timestamps_totais": len(df),
            "percentual_valido": 0.0,
            "duplicados": None,
            "fora_de_ordem": None,
            "delta_min": None,
            "delta_max": None,
            "delta_medio": None,
            "delta_mediano": None,
            "delta_p05": None,
            "delta_p95": None,
            "frequencia_estimada_hz": None,
            "percentual_intervalos_compativeis": None,
            "compatibilidade_temporal": False,
            "regularizacao_necessaria": True,
            "motivo": (
                "Nenhuma coluna temporal utilizável "
                "foi encontrada."
            ),
        }

    validos = timestamps.notna()

    quantidade_validos = int(
        validos.sum()
    )

    percentual_valido = (
        quantidade_validos
        / len(df)
        * 100
        if len(df)
        else 0.0
    )

    timestamps_validos = (
        timestamps.loc[validos]
    )

    duplicados = int(
        timestamps_validos
        .duplicated()
        .sum()
    )

    deltas = (
        timestamps_validos
        .diff()
        .dt.total_seconds()
        .dropna()
    )

    fora_de_ordem = int(
        (deltas < 0).sum()
    )

    # Para estatísticas de cadência, somente deltas positivos
    # representam avanço temporal real.
    deltas_positivos = deltas[
        deltas > 0
    ]

    if deltas_positivos.empty:
        return {
            "coluna_temporal": coluna,
            "colunas_avaliadas": resumo_avaliacoes,
            "timestamps_validos": quantidade_validos,
            "timestamps_totais": len(df),
            "percentual_valido": round(
                percentual_valido,
                2,
            ),
            "duplicados": duplicados,
            "fora_de_ordem": fora_de_ordem,
            "delta_min": None,
            "delta_max": None,
            "delta_medio": None,
            "delta_mediano": None,
            "delta_p05": None,
            "delta_p95": None,
            "frequencia_estimada_hz": None,
            "percentual_intervalos_compativeis": 0.0,
            "compatibilidade_temporal": False,
            "regularizacao_necessaria": True,
            "motivo": (
                "Não existem intervalos temporais "
                "positivos suficientes para avaliar "
                "a cadência."
            ),
        }

    delta_min = float(
        deltas_positivos.min()
    )

    delta_max = float(
        deltas_positivos.max()
    )

    delta_medio = float(
        deltas_positivos.mean()
    )

    delta_mediano = float(
        deltas_positivos.median()
    )

    delta_p05 = float(
        deltas_positivos.quantile(
            0.05
        )
    )

    delta_p95 = float(
        deltas_positivos.quantile(
            0.95
        )
    )

    frequencia = (
        1.0 / delta_medio
        if delta_medio > 0
        else None
    )

    limite_inferior = (
        SAMPLING_INTERVAL_SECONDS
        - TOLERANCIA_INTERVALO_SEGUNDOS
    )

    limite_superior = (
        SAMPLING_INTERVAL_SECONDS
        + TOLERANCIA_INTERVALO_SEGUNDOS
    )

    intervalos_compativeis = (
        deltas_positivos.between(
            limite_inferior,
            limite_superior,
            inclusive="both",
        )
    )

    percentual_intervalos_compativeis = (
        float(
            intervalos_compativeis.mean()
            * 100
        )
    )

    # Critério V5:
    # - cobertura temporal >= 95%
    # - sem duplicatas
    # - sem regressões no tempo
    # - mediana dentro de 1,0 ± 0,2 s
    # - >= 90% dos intervalos dentro da tolerância
    cobertura_ok = (
        percentual_valido >= 95.0
    )

    ordem_ok = (
        fora_de_ordem == 0
    )

    duplicados_ok = (
        duplicados == 0
    )

    mediana_ok = (
        limite_inferior
        <= delta_mediano
        <= limite_superior
    )

    intervalos_ok = (
        percentual_intervalos_compativeis
        >= (
            PERCENTUAL_MIN_INTERVALOS_COMPATIVEIS
            * 100
        )
    )

    compatibilidade = all(
        [
            cobertura_ok,
            ordem_ok,
            duplicados_ok,
            mediana_ok,
            intervalos_ok,
        ]
    )

    motivos = []

    if not cobertura_ok:
        motivos.append(
            "cobertura temporal abaixo de 95%"
        )

    if not duplicados_ok:
        motivos.append(
            f"{duplicados} timestamp(s) duplicado(s)"
        )

    if not ordem_ok:
        motivos.append(
            f"{fora_de_ordem} regressão(ões) temporal(is)"
        )

    if not mediana_ok:
        motivos.append(
            (
                "intervalo mediano incompatível "
                f"com {SAMPLING_INTERVAL_SECONDS:.3f} s"
            )
        )

    if not intervalos_ok:
        motivos.append(
            (
                "menos de 90% dos intervalos "
                "estão dentro da tolerância"
            )
        )

    if compatibilidade:
        motivo = (
            "Cadência considerada diretamente "
            "compatível com o intervalo temporal "
            "do modelo."
        )
    else:
        motivo = "; ".join(motivos)

    return {
        "coluna_temporal": coluna,
        "colunas_avaliadas": resumo_avaliacoes,
        "timestamps_validos": quantidade_validos,
        "timestamps_totais": len(df),
        "percentual_valido": round(
            percentual_valido,
            2,
        ),
        "duplicados": duplicados,
        "fora_de_ordem": fora_de_ordem,
        "delta_min": delta_min,
        "delta_max": delta_max,
        "delta_medio": delta_medio,
        "delta_mediano": delta_mediano,
        "delta_p05": delta_p05,
        "delta_p95": delta_p95,
        "frequencia_estimada_hz": frequencia,
        "percentual_intervalos_compativeis": round(
            percentual_intervalos_compativeis,
            2,
        ),
        "compatibilidade_temporal": compatibilidade,
        "regularizacao_necessaria": (
            not compatibilidade
        ),
        "motivo": motivo,
    }
# ============================================================
# REGULARIZAÇÃO TEMPORAL CONSERVADORA - NOVO NA V6
# ============================================================

def regularizar_para_1hz(
    df_estrutura: pd.DataFrame,
    df_upx: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Regulariza a telemetria para uma grade temporal de 1 Hz.

    Política conservadora da V6:

    - utiliza a mesma coluna temporal selecionada pelo diagnóstico;
    - cada timestamp é associado ao segundo civil correspondente;
    - uma observação no segundo: preserva o valor;
    - duas ou mais observações no segundo: utiliza a mediana;
    - segundos sem observação permanecem NaN;
    - NÃO interpola;
    - NÃO utiliza forward-fill;
    - NÃO transforma ausência em zero;
    - NÃO cria sensores inexistentes.

    O resultado possui uma coluna explícita chamada "timestamp",
    seguida somente das features UPX realmente reconhecidas.
    """

    (
        coluna_temporal,
        timestamps,
        _,
    ) = escolher_coluna_temporal(
        df_estrutura
    )

    # --------------------------------------------------------
    # CASO 1: NÃO EXISTE COLUNA TEMPORAL UTILIZÁVEL
    # --------------------------------------------------------

    if (
        coluna_temporal is None
        or timestamps is None
    ):
        relatorio = {
            "sucesso": False,
            "coluna_temporal": None,
            "motivo": (
                "Não foi possível construir a grade de 1 Hz "
                "porque nenhuma coluna temporal utilizável "
                "foi encontrada."
            ),
            "medicoes_originais": len(df_upx),
            "timestamps_validos": 0,
            "inicio_grade": None,
            "fim_grade": None,
            "segundos_grade": 0,
            "segundos_sem_observacao": None,
            "segundos_com_1_observacao": None,
            "segundos_com_2_observacoes": None,
            "segundos_com_3_ou_mais_observacoes": None,
            "max_observacoes_por_segundo": None,
            "completude_features": {},
            "linhas_completas_8_features": 0,
            "janelas_validas_30s": 0,
            "pronto_para_modelo": False,
        }

        return pd.DataFrame(), relatorio

    # --------------------------------------------------------
    # ALINHAMENTO ENTRE TEMPO E FEATURES
    # --------------------------------------------------------

    trabalho = df_upx.copy()

    # Os índices devem estar alinhados porque ambos derivam
    # do mesmo dataframe após a limpeza estrutural.
    timestamps = timestamps.reindex(
        trabalho.index
    )

    trabalho.insert(
        0,
        "_timestamp",
        timestamps,
    )

    # Somente linhas com timestamp válido podem participar
    # da construção da grade temporal.
    trabalho = trabalho.loc[
        trabalho["_timestamp"].notna()
    ].copy()

    if trabalho.empty:
        relatorio = {
            "sucesso": False,
            "coluna_temporal": coluna_temporal,
            "motivo": (
                "A coluna temporal foi identificada, "
                "mas nenhum timestamp válido permaneceu "
                "para a regularização."
            ),
            "medicoes_originais": len(df_upx),
            "timestamps_validos": 0,
            "inicio_grade": None,
            "fim_grade": None,
            "segundos_grade": 0,
            "segundos_sem_observacao": None,
            "segundos_com_1_observacao": None,
            "segundos_com_2_observacoes": None,
            "segundos_com_3_ou_mais_observacoes": None,
            "max_observacoes_por_segundo": None,
            "completude_features": {},
            "linhas_completas_8_features": 0,
            "janelas_validas_30s": 0,
            "pronto_para_modelo": False,
        }

        return pd.DataFrame(), relatorio

    # --------------------------------------------------------
    # SEGUNDO CIVIL
    # --------------------------------------------------------

    # Exemplo:
    #
    # 07:02:30.380 -> 07:02:30
    # 07:02:30.941 -> 07:02:30
    # 07:02:31.672 -> 07:02:31
    #
    # Não usamos o primeiro timestamp como âncora.
    # A grade é baseada no relógio civil.
    trabalho["_segundo"] = (
        trabalho["_timestamp"].dt.floor("s")
    )

    # --------------------------------------------------------
    # CONTAGEM DE OBSERVAÇÕES REAIS POR SEGUNDO
    # --------------------------------------------------------

    contagens = (
        trabalho.groupby(
            "_segundo"
        )
        .size()
        .sort_index()
    )

    inicio = contagens.index.min()
    fim = contagens.index.max()

    # Grade contínua de exatamente 1 segundo.
    grade = pd.date_range(
        start=inicio,
        end=fim,
        freq="1s",
    )

    # Reindexamos as contagens para descobrir segundos
    # nos quais nenhuma observação real existiu.
    contagens_grade = (
        contagens
        .reindex(
            grade,
            fill_value=0,
        )
    )

    segundos_sem = int(
        (contagens_grade == 0).sum()
    )

    segundos_com_1 = int(
        (contagens_grade == 1).sum()
    )

    segundos_com_2 = int(
        (contagens_grade == 2).sum()
    )

    segundos_com_3_mais = int(
        (contagens_grade >= 3).sum()
    )

    max_observacoes = int(
        contagens_grade.max()
    )

    # --------------------------------------------------------
    # AGREGAÇÃO DAS FEATURES
    # --------------------------------------------------------

    features_disponiveis = [
        feature
        for feature in FEATURES_UPX
        if feature in trabalho.columns
    ]

    if features_disponiveis:
        # A mediana tem duas propriedades importantes aqui:
        #
        # 1 valor no segundo  -> o próprio valor é preservado.
        # 2+ valores          -> mediana dos valores reais.
        #
        # NaN não vira zero.
        agregado = (
            trabalho.groupby(
                "_segundo"
            )[
                features_disponiveis
            ]
            .median()
        )

        # A grade contínua cria explicitamente os segundos
        # ausentes. Eles permanecem NaN.
        agregado = agregado.reindex(
            grade
        )

    else:
        agregado = pd.DataFrame(
            index=grade
        )

    agregado.index.name = "timestamp"

    df_regularizado = (
        agregado.reset_index()
    )

    # --------------------------------------------------------
    # COMPLETUDE PÓS-REGULARIZAÇÃO
    # --------------------------------------------------------

    completude_features = {}

    for feature in features_disponiveis:
        if len(df_regularizado):
            percentual = float(
                df_regularizado[
                    feature
                ]
                .notna()
                .mean()
                * 100
            )
        else:
            percentual = 0.0

        completude_features[
            feature
        ] = round(
            percentual,
            2,
        )

    # --------------------------------------------------------
    # LINHAS COMPLETAS NAS 8 FEATURES
    # --------------------------------------------------------

    todas_features_presentes = all(
        feature in df_regularizado.columns
        for feature in FEATURES_UPX
    )

    if todas_features_presentes:
        mascara_linha_completa = (
            df_regularizado[
                FEATURES_UPX
            ]
            .notna()
            .all(
                axis=1
            )
        )

        linhas_completas = int(
            mascara_linha_completa.sum()
        )

    else:
        mascara_linha_completa = pd.Series(
            False,
            index=df_regularizado.index,
            dtype=bool,
        )

        linhas_completas = 0

    # --------------------------------------------------------
    # JANELAS CONSECUTIVAS VÁLIDAS DE 30 SEGUNDOS
    # --------------------------------------------------------

    # O modelo trabalha com WINDOW_SIZE = 30.
    #
    # Uma janela somente é válida se TODAS as 8 features
    # possuírem dados em TODOS os 30 segundos consecutivos.
    #
    # Como a grade já é contínua em 1 Hz, uma janela com
    # qualquer NaN automaticamente deixa de ser válida.
    if (
        todas_features_presentes
        and len(df_regularizado) >= WINDOW_SIZE
    ):
        quantidade_validos_janela = (
            mascara_linha_completa
            .astype(int)
            .rolling(
                window=WINDOW_SIZE,
                min_periods=WINDOW_SIZE,
            )
            .sum()
        )

        janelas_validas = int(
            (
                quantidade_validos_janela
                == WINDOW_SIZE
            ).sum()
        )

    else:
        janelas_validas = 0

    # --------------------------------------------------------
    # RESULTADO FINAL DA REGULARIZAÇÃO
    # --------------------------------------------------------

    pronto_para_modelo = (
        todas_features_presentes
        and janelas_validas > 0
    )

    if pronto_para_modelo:
        motivo = (
            "Grade temporal de 1 Hz construída e existe "
            "pelo menos uma janela consecutiva completa "
            f"de {WINDOW_SIZE} segundos."
        )

    elif not todas_features_presentes:
        faltantes = [
            feature
            for feature in FEATURES_UPX
            if feature not in df_regularizado.columns
        ]

        motivo = (
            "A grade temporal de 1 Hz foi construída, "
            "mas faltam features exigidas pelo modelo: "
            + ", ".join(faltantes)
        )

    else:
        motivo = (
            "A grade temporal de 1 Hz foi construída, "
            "mas não existe nenhuma janela consecutiva "
            f"completa de {WINDOW_SIZE} segundos nas "
            "8 features."
        )

    relatorio = {
        "sucesso": True,
        "coluna_temporal": coluna_temporal,
        "motivo": motivo,
        "medicoes_originais": len(df_upx),
        "timestamps_validos": len(trabalho),
        "inicio_grade": inicio,
        "fim_grade": fim,
        "segundos_grade": len(df_regularizado),
        "segundos_sem_observacao": segundos_sem,
        "segundos_com_1_observacao": segundos_com_1,
        "segundos_com_2_observacoes": segundos_com_2,
        "segundos_com_3_ou_mais_observacoes": (
            segundos_com_3_mais
        ),
        "max_observacoes_por_segundo": (
            max_observacoes
        ),
        "completude_features": (
            completude_features
        ),
        "linhas_completas_8_features": (
            linhas_completas
        ),
        "janelas_validas_30s": (
            janelas_validas
        ),
        "pronto_para_modelo": (
            pronto_para_modelo
        ),
    }

    return (
        df_regularizado,
        relatorio,
    )

# ============================================================
# VALIDAÇÃO PARA O MODELO
# ============================================================

def validar_para_modelo(
    df_upx: pd.DataFrame,
    diagnostico_temporal: dict,
    relatorio_regularizacao: dict,
) -> dict:
    presentes = [
        feature
        for feature in FEATURES_UPX
        if feature in df_upx.columns
    ]

    ausentes = [
        feature
        for feature in FEATURES_UPX
        if feature not in df_upx.columns
    ]

    sem_dados = [
        feature
        for feature in presentes
        if df_upx[feature].notna().sum() == 0
    ]

    com_dados = [
        feature
        for feature in presentes
        if df_upx[feature].notna().sum() > 0
    ]

    todas_presentes = len(ausentes) == 0
    todas_com_dados = len(sem_dados) == 0
    linhas_suficientes = len(df_upx) >= WINDOW_SIZE

    compatibilidade_temporal_direta = (
        diagnostico_temporal.get(
            "compatibilidade_temporal",
            False,
        )
    )

    regularizacao_concluida = (
        relatorio_regularizacao.get(
            "sucesso",
            False,
        )
    )

    janelas_validas = int(
        relatorio_regularizacao.get(
            "janelas_validas_30s",
            0,
        )
        or 0
    )

    pronto_estruturalmente = (
        todas_presentes
        and todas_com_dados
        and linhas_suficientes
    )

    # Na V6, a decisão final é tomada sobre a grade regularizada.
    # Não basta possuir 30 linhas brutas: é necessário existir
    # ao menos uma janela consecutiva de 30 segundos em 1 Hz,
    # completa nas oito features exigidas pelo modelo.
    pronto_para_modelo = (
        regularizacao_concluida
        and janelas_validas > 0
        and todas_presentes
        and todas_com_dados
    )

    return {
        "features_presentes": presentes,
        "features_ausentes": ausentes,
        "features_sem_dados": sem_dados,
        "features_com_dados": com_dados,
        "linhas": len(df_upx),
        "window_size": WINDOW_SIZE,
        "linhas_suficientes": linhas_suficientes,
        "compatibilidade_features": (
            todas_presentes and todas_com_dados
        ),
        "compatibilidade_temporal_direta": (
            compatibilidade_temporal_direta
        ),
        "regularizacao_concluida": (
            regularizacao_concluida
        ),
        "janelas_validas_30s": janelas_validas,
        "compatibilidade_temporal": (
            regularizacao_concluida
        ),
        "pronto_estruturalmente": (
            pronto_estruturalmente
        ),
        "pronto_para_modelo": (
            pronto_para_modelo
        ),
    }


# ============================================================
# EXPORTAÇÃO
# ============================================================

def exportar_csv_normalizado(
    arquivo_original: Path,
    df_upx: pd.DataFrame,
) -> Path:
    destino = arquivo_original.with_name(
        (
            f"{arquivo_original.stem}"
            "_normalizado_upx.csv"
        )
    )

    df_upx.to_csv(
        destino,
        index=False,
        encoding="utf-8-sig",
    )

    return destino


def exportar_csv_regularizado(
    arquivo_original: Path,
    df_regularizado: pd.DataFrame,
    relatorio_regularizacao: dict,
) -> Optional[Path]:
    if not relatorio_regularizacao.get("sucesso", False):
        return None

    destino = arquivo_original.with_name(
        f"{arquivo_original.stem}_regularizado_1hz_upx.csv"
    )

    df_regularizado.to_csv(
        destino,
        index=False,
        encoding="utf-8-sig",
    )

    return destino



# ============================================================
# PERFIS DE SCANNER - V7.5 / BLOCO 3
# ============================================================

def assinatura_estrutural_colunas(df: pd.DataFrame) -> str:
    """Gera uma assinatura estável a partir dos cabeçalhos normalizados."""
    cabecalhos = [normalizar_para_comparacao(coluna) for coluna in df.columns]
    base = "\n".join(sorted(cabecalhos))
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def carregar_perfis_scanner() -> dict:
    """Carrega conhecimento persistido. Arquivo ausente equivale a base vazia."""
    if not ARQUIVO_PERFIS.exists():
        return {"versao": 1, "perfis": {}}
    try:
        with open(ARQUIVO_PERFIS, "r", encoding="utf-8") as arquivo:
            dados = json.load(arquivo)
    except (OSError, json.JSONDecodeError) as erro:
        print(f"AVISO: perfis_scanner.json não pôde ser carregado: {erro}")
        return {"versao": 1, "perfis": {}}
    if not isinstance(dados, dict):
        return {"versao": 1, "perfis": {}}
    dados.setdefault("versao", 1)
    dados.setdefault("perfis", {})
    return dados


def salvar_perfis_scanner(perfis: dict) -> None:
    """Persiste perfis de forma legível e auditável."""
    temporario = ARQUIVO_PERFIS.with_suffix(".json.tmp")
    with open(temporario, "w", encoding="utf-8") as arquivo:
        json.dump(perfis, arquivo, ensure_ascii=False, indent=2)
        arquivo.write("\n")
    temporario.replace(ARQUIVO_PERFIS)


def aplicar_perfil_confirmado(
    df: pd.DataFrame,
    inventario: pd.DataFrame,
    mapeamentos: list[Mapeamento],
    assinatura: str,
    perfis: dict,
) -> tuple[list[Mapeamento], list[dict]]:
    """Reutiliza somente associações salvas que continuam fisicamente válidas."""
    perfil = perfis.get("perfis", {}).get(assinatura)
    if not perfil:
        return mapeamentos, []

    novos = [
        Mapeamento(
            feature_upx=item.feature_upx,
            coluna_origem=item.coluna_origem,
            confianca=item.confianca,
            metodo=item.metodo,
            observacao=item.observacao,
        )
        for item in mapeamentos
    ]
    por_feature = {item.feature_upx: item for item in novos}
    aplicacoes = []

    for feature, regra in perfil.get("mapeamentos", {}).items():
        if feature not in FEATURES_UPX or not isinstance(regra, dict):
            continue
        coluna = regra.get("coluna_origem")
        if not coluna or coluna not in df.columns:
            continue

        linhas = inventario[inventario["coluna_original"] == coluna]
        if linhas.empty:
            continue
        linha = linhas.iloc[0]
        compativeis = {
            x.strip() for x in str(linha["features_fisicamente_compativeis"]).split(",")
            if x.strip() and x.strip() != "-"
        }
        if int(linha["dados_validos"]) <= 0 or feature not in compativeis:
            continue

        item = por_feature.get(feature)
        anterior = item.coluna_origem if item else None
        if item is None:
            item = Mapeamento(feature, coluna, 1.0, "perfil_confirmado", "Fonte reutilizada de perfil confirmado V7.5.")
            novos.append(item)
            por_feature[feature] = item
        else:
            item.coluna_origem = coluna
            item.confianca = 1.0
            item.metodo = "perfil_confirmado"
            item.observacao = "Fonte reutilizada de perfil confirmado V7.5."

        aplicacoes.append({
            "feature_upx": feature,
            "fonte_anterior": anterior,
            "fonte_aplicada": coluna,
            "origem_decisao": "perfil confirmado",
        })

    return novos, aplicacoes


def persistir_associacao_confirmada(
    perfis: dict,
    assinatura: str,
    df: pd.DataFrame,
    feature: str,
    fonte: str,
    unidade: str,
    conversao: str,
) -> None:
    perfil = perfis.setdefault("perfis", {}).setdefault(
        assinatura,
        {
            "assinatura": assinatura,
            "quantidade_colunas": len(df.columns),
            "cabecalhos_normalizados": sorted(
                normalizar_para_comparacao(coluna) for coluna in df.columns
            ),
            "mapeamentos": {},
        },
    )
    perfil.setdefault("mapeamentos", {})[feature] = {
        "coluna_origem": fonte,
        "unidade_origem": unidade,
        "conversao": conversao,
        "origem": "confirmação humana",
    }
    salvar_perfis_scanner(perfis)


# ============================================================
# RELATÓRIO DE FONTES - V7.5 / BLOCOS 2B + 2C
# ============================================================

def construir_relatorio_fontes(
    df: pd.DataFrame,
    inventario: pd.DataFrame,
    mapeamentos: list[Mapeamento],
    catalogo: dict,
) -> tuple[pd.DataFrame, dict]:
    """Compara fontes fisicamente válidas sem alterar o mapeamento.

    Bloco 2B: somente investigação e preferência técnica. A fonte usada
    pelo pipeline continua sendo a confirmada pelo mapeamento existente.
    Nenhuma troca é aplicada aqui; confirmação humana pertence ao 2C.
    """
    fonte_atual = {
        item.feature_upx: item.coluna_origem
        for item in mapeamentos
        if item.coluna_origem is not None
    }

    registros = []
    resumo = {}

    for feature in FEATURES_UPX:
        config = catalogo.get("features", {}).get(feature, {})
        # Compatibilidade física já foi filtrada conservadoramente no 2A.
        fontes = inventario[
            (inventario["dados_validos"] > 0)
            & inventario["features_fisicamente_compativeis"].apply(
                lambda valor: feature in {
                    item.strip()
                    for item in str(valor).split(",")
                    if item.strip() and item.strip() != "-"
                }
            )
        ].copy()

        avaliadas = []
        for _, item in fontes.iterrows():
            coluna = item["coluna_original"]
            unidade = item["unidade"]
            unidade_origem = None if unidade == "-" else unidade

            serie_convertida, _, descricao_conversao = aplicar_conversao(
                df[coluna],
                unidade_origem,
                config,
            )
            validos_apos = int(serie_convertida.notna().sum())
            completude_apos = (
                validos_apos / len(df) * 100
                if len(df)
                else 0.0
            )
            plaus = analisar_plausibilidade(
                serie_convertida,
                config,
            )
            plaus_pct = plaus["percentual_plausivel"]

            avaliadas.append({
                "feature_upx": feature,
                "coluna_origem": coluna,
                "unidade_origem": unidade_origem or "-",
                "conversao": descricao_conversao,
                "dados_validos": validos_apos,
                "total_linhas": len(df),
                "completude_pct": round(completude_apos, 2),
                "plausibilidade_pct": plaus_pct,
                "fonte_atual": coluna == fonte_atual.get(feature),
            })

        # Só depois das barreiras físicas/unidade comparamos qualidade.
        # Plausibilidade vem antes da completude. Empate preserva a fonte
        # atualmente confirmada; persistindo empate, usa nome só para
        # produzir ordem determinística, sem significado físico.
        def chave_preferencia(item: dict):
            plaus = item["plausibilidade_pct"]
            plaus_ordenacao = -1.0 if plaus is None else float(plaus)
            return (
                plaus_ordenacao,
                float(item["completude_pct"]),
                1 if item["fonte_atual"] else 0,
                item["coluna_origem"].lower(),
            )

        ordenadas = sorted(
            avaliadas,
            key=chave_preferencia,
            reverse=True,
        )

        preferida = ordenadas[0]["coluna_origem"] if ordenadas else None
        atual = fonte_atual.get(feature)

        for posicao, item in enumerate(ordenadas, start=1):
            item["ordem_tecnica"] = posicao
            item["preferencia_tecnica"] = item["coluna_origem"] == preferida
            item["troca_aplicada"] = False
            registros.append(item)

        resumo[feature] = {
            "quantidade_fontes_validas": len(ordenadas),
            "fonte_atual": atual,
            "fonte_tecnicamente_preferida": preferida,
            "alternativa_preferivel": bool(
                preferida is not None
                and atual is not None
                and preferida != atual
            ),
            "troca_aplicada": False,
        }

    colunas = [
        "feature_upx", "ordem_tecnica", "coluna_origem",
        "unidade_origem", "conversao", "dados_validos",
        "total_linhas", "completude_pct", "plausibilidade_pct",
        "fonte_atual", "preferencia_tecnica", "troca_aplicada",
    ]
    relatorio = pd.DataFrame(registros)
    if relatorio.empty:
        relatorio = pd.DataFrame(columns=colunas)
    else:
        relatorio = relatorio[colunas]

    return relatorio, resumo

# ============================================================
# CONFIRMAÇÃO HUMANA DE FONTE - V7.5 / BLOCO 2C + PERFIS
# ============================================================

def perguntar_sim_nao(mensagem: str) -> bool:
    """Aceita somente S ou N. Qualquer outra entrada repete a pergunta."""
    while True:
        resposta = input(mensagem).strip().lower()
        if resposta in {"s", "sim"}:
            return True
        if resposta in {"n", "nao", "não"}:
            return False
        print("Resposta inválida. Digite apenas S ou N.")


def confirmar_fontes_preferidas(
    mapeamentos: list[Mapeamento],
    relatorio_fontes: pd.DataFrame,
    resumo_fontes: dict,
    df: pd.DataFrame,
    assinatura_perfil: str,
    perfis: dict,
) -> tuple[list[Mapeamento], list[dict], pd.DataFrame, dict]:
    """Solicita autorização somente para trocar uma fonte já confirmada.

    A decisão vale apenas para a execução atual. Não cria nem altera perfil
    persistente de scanner. A recomendação técnica do 2B nunca é aplicada
    sem uma resposta humana afirmativa.
    """
    novos = [
        Mapeamento(
            feature_upx=item.feature_upx,
            coluna_origem=item.coluna_origem,
            confianca=item.confianca,
            metodo=item.metodo,
            observacao=item.observacao,
        )
        for item in mapeamentos
    ]
    decisoes = []
    relatorio = relatorio_fontes.copy()
    resumo = {feature: dict(info) for feature, info in resumo_fontes.items()}

    por_feature = {item.feature_upx: item for item in novos}

    for feature in FEATURES_UPX:
        info = resumo.get(feature, {})
        if not info.get("alternativa_preferivel", False):
            continue

        atual = info.get("fonte_atual")
        preferida = info.get("fonte_tecnicamente_preferida")
        if not atual or not preferida or atual == preferida:
            continue

        fontes = relatorio[relatorio["feature_upx"] == feature]
        linha_atual = fontes[fontes["coluna_origem"] == atual]
        linha_pref = fontes[fontes["coluna_origem"] == preferida]
        if linha_atual.empty or linha_pref.empty:
            continue

        a = linha_atual.iloc[0]
        b = linha_pref.iloc[0]

        print()
        print("=" * 82)
        print("CONFIRMAÇÃO HUMANA DE FONTE - V7.5")
        print("=" * 82)
        print(f"Feature: {feature}")
        print()
        print("Fonte atual:")
        print(f"  {atual} — {float(a['completude_pct']):.2f}%")
        print()
        print("Fonte tecnicamente recomendada:")
        print(f"  {preferida} — {float(b['completude_pct']):.2f}%")
        print(f"  Conversão: {b['conversao']}")
        print(f"  Plausibilidade: {float(b['plausibilidade_pct']):.2f}%")
        print()

        aceitou = perguntar_sim_nao(
            "Deseja usar a fonte recomendada nesta execução? [S/N]: "
        )

        if aceitou:
            item = por_feature.get(feature)
            if item is None:
                item = Mapeamento(
                    feature_upx=feature,
                    coluna_origem=preferida,
                    confianca=1.0,
                    metodo="confirmacao_humana",
                    observacao="Fonte escolhida após recomendação técnica V7.5.",
                )
                novos.append(item)
                por_feature[feature] = item
            else:
                item.coluna_origem = preferida
                item.confianca = 1.0
                item.metodo = "confirmacao_humana"
                item.observacao = "Fonte escolhida após recomendação técnica V7.5."

            relatorio.loc[
                (relatorio["feature_upx"] == feature)
                & (relatorio["coluna_origem"] == preferida),
                "troca_aplicada",
            ] = True
            resumo[feature]["troca_aplicada"] = True
            resumo[feature]["fonte_aplicada"] = preferida
            decisao = "ACEITA"
            origem = "confirmação humana"

            persistir = perguntar_sim_nao(
                "Deseja salvar esta associação para arquivos futuros deste perfil? [S/N]: "
            )
            if persistir:
                persistir_associacao_confirmada(
                    perfis,
                    assinatura_perfil,
                    df,
                    feature,
                    preferida,
                    str(b["unidade_origem"]),
                    str(b["conversao"]),
                )
        else:
            persistir = False
            resumo[feature]["troca_aplicada"] = False
            resumo[feature]["fonte_aplicada"] = atual
            decisao = "RECUSADA"
            origem = "confirmação humana"

        decisoes.append({
            "feature_upx": feature,
            "fonte_anterior": atual,
            "fonte_recomendada": preferida,
            "decisao": decisao,
            "fonte_aplicada": resumo[feature]["fonte_aplicada"],
            "origem_decisao": origem,
            "persistida": bool(persistir),
        })

    return novos, decisoes, relatorio, resumo


# ============================================================
# ANÁLISE COMPLETA
# ============================================================

def analisar_csv(
    caminho: Path,
    catalogo: dict,
) -> dict:
    df_bruto, metadados = ler_csv(
        caminho
    )

    # V5: limpeza estrutural ANTES de qualquer
    # avaliação de completude ou tempo.
    df_estrutura, limpeza_estrutural = (
        limpar_estrutura(
            df_bruto
        )
    )

    # V5: diagnóstico temporal utiliza o conteúdo textual
    # já estruturalmente limpo, antes da tentativa de converter
    # colunas para números.
    diagnostico_temporal = (
        diagnosticar_tempo(
            df_estrutura
        )
    )

    # Pipeline V4 continua daqui.
    df = limpar_dataframe(
        df_estrutura
    )

    df = converter_colunas_numericas(
        df
    )

    (
        mapeamentos,
        candidatos,
    ) = mapear_features(
        df,
        catalogo,
    )

    # V7 - Bloco 1: inventário descritivo de todas as colunas.
    # Não altera o mapeamento nem seleciona candidatos.
    inventario_sensores = construir_inventario_sensores(
        df,
        mapeamentos,
        candidatos,
    )

    resumo_inventario = resumir_inventario(
        inventario_sensores
    )

    # V7.5 / Bloco 3: perfil é identificado pela estrutura dos cabeçalhos.
    # Associações persistidas só são reutilizadas se ainda passarem pela
    # equivalência física e possuírem dados nesta execução.
    assinatura_perfil = assinatura_estrutural_colunas(df_estrutura)
    perfis_scanner = carregar_perfis_scanner()
    mapeamentos, aplicacoes_perfil = aplicar_perfil_confirmado(
        df,
        inventario_sensores,
        mapeamentos,
        assinatura_perfil,
        perfis_scanner,
    )

    # V7.3 / 2B: compara fontes sem aplicar troca automaticamente.
    relatorio_fontes, resumo_fontes = construir_relatorio_fontes(
        df,
        inventario_sensores,
        mapeamentos,
        catalogo,
    )

    # V7.4 / 2C: somente uma confirmação humana explícita pode trocar
    # uma fonte já confirmada. A decisão vale apenas nesta execução.
    (
        mapeamentos_aplicados,
        decisoes_fontes,
        relatorio_fontes,
        resumo_fontes,
    ) = confirmar_fontes_preferidas(
        mapeamentos,
        relatorio_fontes,
        resumo_fontes,
        df,
        assinatura_perfil,
        perfis_scanner,
    )

    # O dataframe UPX é construído SOMENTE depois da confirmação, para
    # garantir que toda a cadeia posterior use a fonte realmente aplicada.
    (
        df_upx,
        transformacoes,
    ) = construir_dataframe_upx(
        df,
        mapeamentos_aplicados,
        catalogo,
    )

    (
        df_regularizado,
        relatorio_regularizacao,
    ) = regularizar_para_1hz(
        df_estrutura,
        df_upx,
    )

    validacao = validar_para_modelo(
        df_upx,
        diagnostico_temporal,
        relatorio_regularizacao,
    )

    destino = exportar_csv_normalizado(
        caminho,
        df_upx,
    )

    destino_regularizado = exportar_csv_regularizado(
        caminho,
        df_regularizado,
        relatorio_regularizacao,
    )

    return {
        "metadados": metadados,
        "limpeza_estrutural": limpeza_estrutural,
        "diagnostico_temporal": diagnostico_temporal,
        "mapeamentos": mapeamentos_aplicados,
        "mapeamentos_originais": mapeamentos,
        "decisoes_fontes": decisoes_fontes,
        "aplicacoes_perfil": aplicacoes_perfil,
        "assinatura_perfil": assinatura_perfil,
        "candidatos": candidatos,
        "df_upx": df_upx,
        "df_regularizado": df_regularizado,
        "transformacoes": transformacoes,
        "inventario_sensores": inventario_sensores,
        "resumo_inventario": resumo_inventario,
        "relatorio_fontes": relatorio_fontes,
        "resumo_fontes": resumo_fontes,
        "relatorio_regularizacao": relatorio_regularizacao,
        "validacao": validacao,
        "arquivo_saida": destino,
        "arquivo_saida_regularizado": destino_regularizado,
    }


# ============================================================
# EXIBIÇÃO
# ============================================================

def formatar_numero(
    valor,
    casas: int = 3,
) -> str:
    if valor is None:
        return "N/A"

    try:
        if pd.isna(valor):
            return "N/A"
    except TypeError:
        pass

    return f"{float(valor):.{casas}f}"


def imprimir_resultado(
    resultado: dict,
):
    print()
    print("=" * 82)
    print("ARQUIVO")
    print("=" * 82)

    for chave, valor in (
        resultado["metadados"].items()
    ):
        print(
            f"{chave}: {valor}"
        )

    # --------------------------------------------------------
    # LIMPEZA ESTRUTURAL
    # --------------------------------------------------------

    limpeza = (
        resultado[
            "limpeza_estrutural"
        ]
    )

    print()
    print("=" * 82)
    print("LIMPEZA ESTRUTURAL")
    print("=" * 82)

    print(
        "Linhas originais: "
        f"{limpeza['linhas_originais']}"
    )

    print(
        "Linhas vazias removidas: "
        f"{limpeza['linhas_vazias_removidas']}"
    )

    print(
        "Cabeçalhos repetidos removidos: "
        f"{limpeza['cabecalhos_repetidos_removidos']}"
    )

    if (
        limpeza[
            "indices_cabecalhos_repetidos"
        ]
    ):
        print(
            "Índices originais dos cabeçalhos: "
            + ", ".join(
                map(
                    str,
                    limpeza[
                        "indices_cabecalhos_repetidos"
                    ],
                )
            )
        )

    print(
        "Linhas válidas restantes: "
        f"{limpeza['linhas_restantes']}"
    )

    # --------------------------------------------------------
    # INVENTÁRIO INTELIGENTE DE SENSORES - V7
    # --------------------------------------------------------

    inventario = resultado["inventario_sensores"]
    resumo_inv = resultado["resumo_inventario"]

    print()
    print("=" * 82)
    print("INVENTÁRIO INTELIGENTE DE SENSORES - V7")
    print("=" * 82)

    print(
        "Colunas analisadas: "
        f"{resumo_inv['total_colunas']}"
    )
    print(
        "Colunas com dados: "
        f"{resumo_inv['com_dados']}"
    )
    print(
        "Colunas sem dados: "
        f"{resumo_inv['sem_dados']}"
    )
    print(
        "Mapeadas com segurança: "
        f"{resumo_inv['confirmadas']}"
    )
    print(
        "Candidatas para revisão: "
        f"{resumo_inv['candidatas']}"
    )
    print(
        "Reconhecidas fora do contrato atual: "
        f"{resumo_inv['fora_do_contrato']}"
    )
    print(
        "Ainda desconhecidas: "
        f"{resumo_inv['desconhecidas']}"
    )

    print()
    print(
        inventario.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # RELATÓRIO DE FONTES - V7.5 / BLOCOS 2B + 2C
    # --------------------------------------------------------

    print()
    print("=" * 82)
    print("RELATÓRIO DE FONTES - V7.5")
    print("=" * 82)

    relatorio_fontes = resultado["relatorio_fontes"]
    resumo_fontes = resultado["resumo_fontes"]

    for feature in FEATURES_UPX:
        info = resumo_fontes[feature]
        print()
        print(f"[{feature}]")

        fontes_feature = relatorio_fontes[
            relatorio_fontes["feature_upx"] == feature
        ]

        if fontes_feature.empty:
            print("  Nenhuma fonte fisicamente válida com dados.")
            continue

        print(
            "  Fontes fisicamente válidas com dados: "
            f"{info['quantidade_fontes_validas']}"
        )

        for _, fonte in fontes_feature.iterrows():
            plaus = fonte["plausibilidade_pct"]
            plaus_txt = (
                "N/A" if pd.isna(plaus)
                else f"{float(plaus):.2f}%"
            )
            print(
                f"  {int(fonte['ordem_tecnica'])}. "
                f"{fonte['coluna_origem']}"
            )
            print(
                f"     unidade: {fonte['unidade_origem']} | "
                f"conversão: {fonte['conversao']}"
            )
            print(
                f"     dados: {int(fonte['dados_validos'])}/"
                f"{int(fonte['total_linhas'])} | "
                f"completude: {float(fonte['completude_pct']):.2f}% | "
                f"plausibilidade: {plaus_txt}"
            )
            print(
                "     fonte atual: "
                f"{'SIM' if fonte['fonte_atual'] else 'NÃO'} | "
                "preferência técnica: "
                f"{'SIM' if fonte['preferencia_tecnica'] else 'NÃO'}"
            )

        print(
            "  Fonte atualmente usada: "
            f"{info['fonte_atual'] or '-'}"
        )
        print(
            "  Fonte tecnicamente preferida: "
            f"{info['fonte_tecnicamente_preferida'] or '-'}"
        )
        print(
            "  Existe alternativa tecnicamente preferível: "
            f"{'SIM' if info['alternativa_preferivel'] else 'NÃO'}"
        )
        print(
            "  Troca aplicada ao modelo: "
            f"{'SIM' if info.get('troca_aplicada') else 'NÃO'}"
        )
        if info.get("troca_aplicada"):
            print(
                "  Fonte aplicada nesta execução: "
                f"{info.get('fonte_aplicada', '-')}"
            )

    aplicacoes_perfil = resultado.get("aplicacoes_perfil", [])
    if aplicacoes_perfil:
        print()
        print("=" * 82)
        print("PERFIL DE SCANNER APLICADO - V7.5")
        print("=" * 82)
        for aplicacao in aplicacoes_perfil:
            print(
                f"{aplicacao['feature_upx']}: "
                f"{aplicacao['fonte_aplicada']} | "
                f"origem: {aplicacao['origem_decisao']}"
            )

    # --------------------------------------------------------
    # DECISÕES HUMANAS DE FONTE - V7.5 / BLOCO 2C
    # --------------------------------------------------------

    decisoes_fontes = resultado.get("decisoes_fontes", [])
    if decisoes_fontes:
        print()
        print("=" * 82)
        print("DECISÕES HUMANAS DE FONTE - V7.5")
        print("=" * 82)
        for decisao in decisoes_fontes:
            print(
                f"{decisao['feature_upx']}: {decisao['decisao']} | "
                f"fonte aplicada: {decisao['fonte_aplicada']} | "
                f"origem: {decisao['origem_decisao']} | "
                f"persistida: {'SIM' if decisao.get('persistida') else 'NÃO'}"
            )

    # --------------------------------------------------------
    # TRANSFORMAÇÕES
    # --------------------------------------------------------

    print()
    print("=" * 82)
    print("TRANSFORMAÇÕES CONFIRMADAS")
    print("=" * 82)
    print()

    transformacoes = (
        resultado["transformacoes"]
    )

    if transformacoes.empty:
        print(
            "Nenhuma feature foi mapeada "
            "com segurança."
        )
    else:
        print(
            transformacoes.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # CANDIDATOS
    # --------------------------------------------------------

    print()
    print("=" * 82)
    print(
        "CANDIDATOS SEMÂNTICOS PARA REVISÃO"
    )
    print("=" * 82)

    encontrou = False

    for feature in FEATURES_UPX:
        candidatos = (
            resultado["candidatos"].get(
                feature,
                [],
            )
        )

        if not candidatos:
            continue

        encontrou = True

        print()
        print(f"[{feature}]")

        for candidato in candidatos:
            print(
                f"  "
                f"{candidato.score * 100:.2f}%  "
                f"{candidato.coluna_origem}"
            )

            print(
                f"  Motivo: "
                f"{candidato.motivo}"
            )

            print(
                "  Decisão: "
                "NÃO MAPEADO AUTOMATICAMENTE"
            )

    if not encontrou:
        print()
        print(
            "Nenhum candidato semanticamente "
            "aceitável foi encontrado."
        )

    # --------------------------------------------------------
    # DIAGNÓSTICO TEMPORAL
    # --------------------------------------------------------

    temporal = (
        resultado[
            "diagnostico_temporal"
        ]
    )

    print()
    print("=" * 82)
    print("DIAGNÓSTICO TEMPORAL")
    print("=" * 82)

    if temporal["colunas_avaliadas"]:
        print(
            "Colunas temporais avaliadas:"
        )

        for item in (
            temporal[
                "colunas_avaliadas"
            ]
        ):
            print(
                f"  - {item['coluna']}: "
                f"{item['validos']} válidos "
                f"({item['percentual_valido']:.2f}%)"
            )

    else:
        print(
            "Colunas temporais avaliadas: nenhuma"
        )

    print(
        "Coluna temporal utilizada: "
        f"{temporal['coluna_temporal'] or 'NENHUMA'}"
    )

    print(
        "Timestamps válidos: "
        f"{temporal['timestamps_validos']}/"
        f"{temporal['timestamps_totais']} "
        f"({temporal['percentual_valido']:.2f}%)"
    )

    print(
        "Duplicados: "
        f"{temporal['duplicados']}"
    )

    print(
        "Fora de ordem: "
        f"{temporal['fora_de_ordem']}"
    )

    print(
        "Intervalo esperado pelo modelo: "
        f"{SAMPLING_INTERVAL_SECONDS:.3f} s"
    )

    print(
        "Delta mínimo: "
        f"{formatar_numero(temporal['delta_min'])} s"
    )

    print(
        "Delta médio: "
        f"{formatar_numero(temporal['delta_medio'])} s"
    )

    print(
        "Delta mediano: "
        f"{formatar_numero(temporal['delta_mediano'])} s"
    )

    print(
        "P05: "
        f"{formatar_numero(temporal['delta_p05'])} s"
    )

    print(
        "P95: "
        f"{formatar_numero(temporal['delta_p95'])} s"
    )

    print(
        "Delta máximo: "
        f"{formatar_numero(temporal['delta_max'])} s"
    )

    print(
        "Frequência estimada: "
        f"{formatar_numero(temporal['frequencia_estimada_hz'])} Hz"
    )

    percentual_intervalos = (
        temporal[
            "percentual_intervalos_compativeis"
        ]
    )

    if percentual_intervalos is None:
        percentual_texto = "N/A"
    else:
        percentual_texto = (
            f"{percentual_intervalos:.2f}%"
        )

    print(
        "Intervalos dentro de "
        "1,0 ± 0,2 s: "
        f"{percentual_texto}"
    )

    print(
        "Compatibilidade temporal direta: "
        + (
            "SIM"
            if temporal[
                "compatibilidade_temporal"
            ]
            else "NÃO"
        )
    )

    print(
        "Regularização temporal necessária: "
        + (
            "SIM"
            if temporal[
                "regularizacao_necessaria"
            ]
            else "NÃO"
        )
    )

    print(
        "Diagnóstico: "
        f"{temporal['motivo']}"
    )

    # --------------------------------------------------------
    # REGULARIZAÇÃO TEMPORAL 1 HZ
    # --------------------------------------------------------

    regularizacao = resultado["relatorio_regularizacao"]

    print()
    print("=" * 82)
    print("REGULARIZAÇÃO TEMPORAL CONSERVADORA - 1 Hz")
    print("=" * 82)

    print(
        "Regularização concluída: "
        + ("SIM" if regularizacao["sucesso"] else "NÃO")
    )
    print(
        "Medições originais: "
        f"{regularizacao['medicoes_originais']}"
    )
    print(
        "Timestamps válidos usados: "
        f"{regularizacao['timestamps_validos']}"
    )

    if regularizacao["sucesso"]:
        print(
            "Início da grade: "
            f"{regularizacao['inicio_grade']}"
        )
        print(
            "Fim da grade: "
            f"{regularizacao['fim_grade']}"
        )
        print(
            "Segundos na grade 1 Hz: "
            f"{regularizacao['segundos_grade']}"
        )
        print(
            "Segundos sem observação real: "
            f"{regularizacao['segundos_sem_observacao']}"
        )
        print(
            "Segundos com 1 observação: "
            f"{regularizacao['segundos_com_1_observacao']}"
        )
        print(
            "Segundos com 2 observações: "
            f"{regularizacao['segundos_com_2_observacoes']}"
        )
        print(
            "Segundos com 3 ou mais observações: "
            f"{regularizacao['segundos_com_3_ou_mais_observacoes']}"
        )
        print(
            "Máximo de observações em um segundo: "
            f"{regularizacao['max_observacoes_por_segundo']}"
        )

        print()
        print("Completude por feature após regularização:")
        if regularizacao["completude_features"]:
            for feature in FEATURES_UPX:
                if feature in regularizacao["completude_features"]:
                    print(
                        f"  - {feature}: "
                        f"{regularizacao['completude_features'][feature]:.2f}%"
                    )
        else:
            print("  Nenhuma feature mapeada com segurança.")

        print(
            "Linhas completas nas 8 features: "
            f"{regularizacao['linhas_completas_8_features']}"
        )
        print(
            f"Janelas válidas de {WINDOW_SIZE} s: "
            f"{regularizacao['janelas_validas_30s']}"
        )

    print(
        "Diagnóstico pós-regularização: "
        f"{regularizacao['motivo']}"
    )

    # --------------------------------------------------------
    # VALIDAÇÃO FINAL
    # --------------------------------------------------------

    validacao = resultado["validacao"]

    print()
    print("=" * 82)
    print(
        "VALIDAÇÃO PARA O MODELO AUTOMOTIVO"
    )
    print("=" * 82)

    print(
        "Features reconhecidas: "
        f"{len(validacao['features_presentes'])}"
        f"/{len(FEATURES_UPX)}"
    )

    print(
        "Features com dados: "
        f"{len(validacao['features_com_dados'])}"
        f"/{len(FEATURES_UPX)}"
    )

    print(
        "Linhas disponíveis após limpeza: "
        f"{validacao['linhas']}"
    )

    print(
        "Janela mínima do modelo: "
        f"{WINDOW_SIZE}"
    )

    print(
        "Quantidade de linhas suficiente: "
        + (
            "SIM"
            if validacao[
                "linhas_suficientes"
            ]
            else "NÃO"
        )
    )

    print(
        "Compatibilidade das features: "
        + (
            "SIM"
            if validacao[
                "compatibilidade_features"
            ]
            else "NÃO"
        )
    )

    print(
        "Compatibilidade temporal: "
        + (
            "SIM"
            if validacao[
                "compatibilidade_temporal"
            ]
            else "NÃO"
        )
    )

    if validacao["features_ausentes"]:
        print(
            "Features ausentes: "
            + ", ".join(
                validacao[
                    "features_ausentes"
                ]
            )
        )

    if (
        validacao[
            "features_sem_dados"
        ]
    ):
        print(
            "Features reconhecidas sem dados: "
            + ", ".join(
                validacao[
                    "features_sem_dados"
                ]
            )
        )

    print()

    if validacao["pronto_para_modelo"]:
        print(
            "Status: "
            "PRONTO PARA INFERÊNCIA DO MODELO"
        )
    elif resultado["relatorio_regularizacao"]["sucesso"]:
        print(
            "Status: "
            "REGULARIZAÇÃO TEMPORAL CONCLUÍDA, "
            "MAS NÃO PRONTO PARA INFERÊNCIA DO MODELO"
        )
    else:
        print(
            "Status: "
            "NÃO PRONTO PARA INFERÊNCIA DO MODELO"
        )

    print()
    print("CSV normalizado pré-regularização gerado em:")
    print(resultado["arquivo_saida"])

    if resultado["arquivo_saida_regularizado"] is not None:
        print()
        print("CSV regularizado em 1 Hz gerado em:")
        print(resultado["arquivo_saida_regularizado"])
    else:
        print()
        print("CSV regularizado em 1 Hz: NÃO GERADO")

    print("=" * 82)


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 82)
    print(
        "NORMALIZADOR INTELIGENTE DE "
        "TELEMETRIA - UPX 2.0 - V7.5"
    )
    print("=" * 82)

    try:
        catalogo = carregar_catalogo()

        print(
            "\nCatálogo carregado: versão "
            f"{catalogo.get('versao', '?')}"
        )

    except Exception as erro:
        print()
        print(
            "ERRO AO CARREGAR O CATÁLOGO"
        )
        print(str(erro))
        return

    caminho = input(
        "\nArraste um CSV ou uma PASTA "
        "para esta janela ou informe "
        "o caminho:\n\n> "
    )

    caminho = (
        caminho
        .strip()
        .strip('"')
    )

    try:
        arquivos = localizar_csvs(
            caminho
        )

        print()
        print(
            f"{len(arquivos)} arquivo(s) "
            "CSV encontrado(s)."
        )

        for numero, arquivo in enumerate(
            arquivos,
            start=1,
        ):
            print()
            print("#" * 82)

            print(
                f"ANALISANDO "
                f"{numero}/{len(arquivos)}"
            )

            print("#" * 82)

            try:
                resultado = analisar_csv(
                    arquivo,
                    catalogo,
                )

                imprimir_resultado(
                    resultado
                )

            except Exception as erro:
                print()
                print(
                    "Falha ao analisar: "
                    f"{arquivo.name}"
                )
                print(str(erro))

    except Exception as erro:
        print()
        print("=" * 82)
        print("ERRO")
        print("=" * 82)
        print(str(erro))
        print("=" * 82)


if __name__ == "__main__":
    main()