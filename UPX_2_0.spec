# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_submodules,
    copy_metadata,
)


# ============================================================
# RAIZ DO PROJETO
# ============================================================

ROOT = Path(SPECPATH).resolve()


# ============================================================
# DADOS DA NOSSA PLATAFORMA
# ============================================================

datas = [
    # --------------------------------------------------------
    # Aplicação Streamlit
    # --------------------------------------------------------
    (
        str(ROOT / "app"),
        "app",
    ),

    # --------------------------------------------------------
    # Pipelines e módulos próprios
    # --------------------------------------------------------
    (
        str(ROOT / "utils"),
        "utils",
    ),

    # --------------------------------------------------------
    # Modelos treinados
    #
    # Inclui:
    # - modelo/fd001_final
    # - modelo/automotivo_anomalia
    # - demais artefatos existentes em modelo/
    # --------------------------------------------------------
    (
        str(ROOT / "modelo"),
        "modelo",
    ),

    # --------------------------------------------------------
    # Configuração do Streamlit
    # --------------------------------------------------------
    (
        str(ROOT / ".streamlit"),
        ".streamlit",
    ),

    # --------------------------------------------------------
    # NASA C-MAPSS
    # Dataset utilizado pelo módulo NASA
    # --------------------------------------------------------
    (
        str(ROOT / "datasets" / "CMAPSSData"),
        "datasets/CMAPSSData",
    ),

    # --------------------------------------------------------
    # NASA C-MAPSS
    # Resultados finais pré-processados utilizados
    # pelo dashboard.
    #
    # Inclui:
    # - metricas_rul_integrado.json
    # - ranking_prioridade.csv
    # - resultados_100_motores.csv
    # - resumo_pipeline.csv
    # - resumo_prioridades.csv
    # - resumo_status.csv
    # --------------------------------------------------------
    (
        str(
            ROOT
            / "resultados"
            / "fd001_pipeline_integrado"
        ),
        "resultados/fd001_pipeline_integrado",
    ),

    # --------------------------------------------------------
    # Automotivo
    # Arquivos utilizados na demonstração do dashboard
    # --------------------------------------------------------
    (
        str(
            ROOT
            / "datasets"
            / "automotivo"
            / "demo_dashboard"
        ),
        "datasets/automotivo/demo_dashboard",
    ),

    # --------------------------------------------------------
    # Automotivo
    # Telemetria simulada
    # --------------------------------------------------------
    (
        str(
            ROOT
            / "datasets"
            / "automotivo"
            / "simulados"
        ),
        "datasets/automotivo/simulados",
    ),
]


# ============================================================
# DEPENDÊNCIAS QUE O PYINSTALLER PODE NÃO DESCOBRIR
# AUTOMATICAMENTE A PARTIR DO LAUNCHER
# ============================================================

hiddenimports = []

hiddenimports += collect_submodules("streamlit")
hiddenimports += collect_submodules("altair")

hiddenimports += collect_submodules("tensorflow")
hiddenimports += collect_submodules("keras")
hiddenimports += collect_submodules("tf_keras")

hiddenimports += collect_submodules("sklearn")
hiddenimports += collect_submodules("joblib")


# ============================================================
# ARQUIVOS INTERNOS DO STREAMLIT
# ============================================================

datas += collect_data_files("streamlit")


# ============================================================
# METADADOS DO STREAMLIT
# ============================================================
#
# Necessário porque o Streamlit consulta seus próprios
# metadados através de importlib.metadata durante a execução.
#
# recursive=True inclui também metadados das dependências
# declaradas pelo Streamlit.
# ============================================================

datas += copy_metadata(
    "streamlit",
    recursive=True,
)


# ============================================================
# METADADOS DE OUTRAS DEPENDÊNCIAS
# ============================================================

for pacote in [
    "altair",
    "tensorflow",
    "keras",
    "tf_keras",
    "scikit-learn",
    "joblib",
]:
    try:
        datas += copy_metadata(pacote)
    except Exception:
        # Alguns nomes de módulos Python podem não corresponder
        # exatamente ao nome da distribuição instalada.
        # Isso não deve interromper todo o build.
        pass


# ============================================================
# TENSORFLOW
# ============================================================
#
# TensorFlow possui módulos, DLLs e arquivos de dados que
# precisam ser coletados explicitamente.
# ============================================================

tf_datas, tf_binaries, tf_hiddenimports = collect_all(
    "tensorflow"
)

datas += tf_datas

binaries = tf_binaries

hiddenimports += tf_hiddenimports


# ============================================================
# ANÁLISE DO PYINSTALLER
# ============================================================

a = Analysis(
    ["distribuicao/launcher.py"],

    pathex=[
        str(ROOT),
    ],

    binaries=binaries,

    datas=datas,

    hiddenimports=hiddenimports,

    hookspath=[],

    hooksconfig={},

    runtime_hooks=[],

    excludes=[],

    noarchive=False,

    optimize=0,
)


# ============================================================
# ARQUIVO PYZ
# ============================================================

pyz = PYZ(
    a.pure
)


# ============================================================
# EXECUTÁVEL
# ============================================================

exe = EXE(
    pyz,

    a.scripts,

    [],

    exclude_binaries=True,

    name="UPX_2_0",

    debug=False,

    bootloader_ignore_signals=False,

    strip=False,

    upx=True,

    # --------------------------------------------------------
    # Enquanto validamos o executável, mantemos o terminal.
    #
    # Depois que tudo estiver funcionando no computador de
    # teste, podemos mudar para:
    #
    # console=False
    #
    # antes de gerar o instalador final no Inno Setup.
    # --------------------------------------------------------
    console=True,

    disable_windowed_traceback=False,

    argv_emulation=False,

    target_arch=None,

    codesign_identity=None,

    entitlements_file=None,
)


# ============================================================
# DISTRIBUIÇÃO ONEDIR
# ============================================================
#
# Resultado:
#
# dist/
# └── UPX_2_0/
#     ├── UPX_2_0.exe
#     └── _internal/
#         ├── app/
#         ├── utils/
#         ├── modelo/
#         ├── datasets/
#         ├── resultados/
#         ├── .streamlit/
#         └── dependências Python
#
# ============================================================

coll = COLLECT(
    exe,

    a.binaries,

    a.datas,

    strip=False,

    upx=True,

    upx_exclude=[],

    name="UPX_2_0",
)