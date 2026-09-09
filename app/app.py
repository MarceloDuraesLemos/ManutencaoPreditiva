from pathlib import Path
import os
import sys

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# AMBIENTE / CAMINHOS
# ============================================================

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.pipeline_fd001 import (
    PipelineFD001,
    carregar_cmapss_fd001,
)


DATASET_PATH = (
    ROOT
    / "datasets"
    / "CMAPSSData"
    / "test_FD001.txt"
)

RESULTADOS_PATH = (
    ROOT
    / "resultados"
    / "fd001_pipeline_integrado"
    / "resultados_100_motores.csv"
)

RANKING_PATH = (
    ROOT
    / "resultados"
    / "fd001_pipeline_integrado"
    / "ranking_prioridade.csv"
)


# ============================================================
# CONFIGURAÇÃO STREAMLIT
# ============================================================

st.set_page_config(
    page_title="UPX 2.0 | Manutenção Preditiva",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    h1, h2, h3 {
        letter-spacing: -0.02em;
    }


    .status-card {
        border-radius: 16px;
        padding: 18px;
        border: 1px solid rgba(128,128,128,0.18);
        margin-bottom: 10px;
        min-height: 105px;
    }

    .status-title {
        font-size: 13px;
        opacity: 0.75;
        font-weight: 700;
    }

    .status-value {
        font-size: 26px;
        font-weight: 800;
        margin-top: 5px;
    }

    .status-green {
        border-left: 6px solid #22c55e;
    }

    .status-yellow {
        border-left: 6px solid #eab308;
    }

    .status-orange {
        border-left: 6px solid #f97316;
    }

    .status-red {
        border-left: 6px solid #ef4444;
    }

    .status-blue {
        border-left: 6px solid #3b82f6;
    }

    .status-gray {
        border-left: 6px solid #94a3b8;
    }

    .small-note {
        opacity: 0.70;
        font-size: 12px;
        margin-top: 6px;
    }

    .recommendation {
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(128,128,128,0.18);
        background: rgba(128,128,128,0.04);
    }

    .recommendation-title {
        font-weight: 800;
        font-size: 17px;
        margin-bottom: 8px;
    }

    .motor-banner {
        border-radius: 18px;
        padding: 22px 24px;
        border: 1px solid rgba(128,128,128,0.20);
        margin-top: 10px;
        margin-bottom: 22px;
    }

    .motor-banner-green {
        border-left: 8px solid #22c55e;
        background: rgba(34,197,94,0.07);
    }

    .motor-banner-yellow {
        border-left: 8px solid #eab308;
        background: rgba(234,179,8,0.07);
    }

    .motor-banner-orange {
        border-left: 8px solid #f97316;
        background: rgba(249,115,22,0.07);
    }

    .motor-banner-red {
        border-left: 8px solid #ef4444;
        background: rgba(239,68,68,0.07);
    }

    .motor-banner-gray {
        border-left: 8px solid #94a3b8;
        background: rgba(148,163,184,0.07);
    }

    .motor-banner-title {
        font-size: 25px;
        font-weight: 900;
        margin-bottom: 5px;
    }

    .motor-banner-subtitle {
        font-size: 14px;
        opacity: 0.78;
    }

    .ranking-box {
        padding: 16px 18px;
        border-radius: 14px;
        border: 1px solid rgba(239,68,68,0.25);
        background: rgba(239,68,68,0.035);
        margin-bottom: 12px;
    }

    .ranking-title {
        font-weight: 800;
        font-size: 17px;
    }

    .ranking-subtitle {
        font-size: 12px;
        opacity: 0.70;
        margin-top: 4px;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.18);
        padding: 16px;
        border-radius: 14px;
        background: rgba(128,128,128,0.025);
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CACHE
# ============================================================

@st.cache_resource
def carregar_pipeline():
    return PipelineFD001()


@st.cache_data
def carregar_dataset():
    return carregar_cmapss_fd001(DATASET_PATH)


@st.cache_data
def carregar_resultados():
    return pd.read_csv(RESULTADOS_PATH)


@st.cache_data
def carregar_ranking():
    return pd.read_csv(RANKING_PATH)


# ============================================================
# FUNÇÕES DE FORMATAÇÃO
# ============================================================

def numero(valor, casas=1):
    if valor is None:
        return "—"

    try:
        if pd.isna(valor):
            return "—"
    except Exception:
        pass

    return f"{float(valor):.{casas}f}"


def status_legivel(status):
    mapa = {
        "SAUDAVEL": "SAUDÁVEL",
        "ATENCAO": "ATENÇÃO",
        "RISCO": "RISCO",
        "CRITICO": "CRÍTICO",
    }

    if status is None:
        return "—"

    try:
        if pd.isna(status):
            return "—"
    except Exception:
        pass

    return mapa.get(status, str(status))


def prioridade_legivel(prioridade):
    mapa = {
        "P1_IMEDIATA": "P1 - IMEDIATA",
        "P2_ALTA": "P2 - ALTA",
        "P3_MEDIA": "P3 - MÉDIA",
        "P4_BAIXA": "P4 - BAIXA",
    }

    if prioridade is None:
        return "—"

    try:
        if pd.isna(prioridade):
            return "—"
    except Exception:
        pass

    return mapa.get(prioridade, str(prioridade))


def classe_status(status):
    mapa = {
        "SAUDAVEL": "status-green",
        "ATENCAO": "status-yellow",
        "RISCO": "status-orange",
        "CRITICO": "status-red",
    }

    return mapa.get(status, "status-gray")


def classe_prioridade(prioridade):
    mapa = {
        "P4_BAIXA": "status-green",
        "P3_MEDIA": "status-yellow",
        "P2_ALTA": "status-orange",
        "P1_IMEDIATA": "status-red",
    }

    return mapa.get(prioridade, "status-gray")


def classe_banner(status):
    mapa = {
        "SAUDAVEL": "motor-banner-green",
        "ATENCAO": "motor-banner-yellow",
        "RISCO": "motor-banner-orange",
        "CRITICO": "motor-banner-red",
    }

    return mapa.get(status, "motor-banner-gray")


# ============================================================
# COMPONENTES VISUAIS
# ============================================================

def card_html(
    titulo,
    valor,
    nota="",
    classe="status-blue",
):
    st.markdown(
        f"""
        <div class="status-card {classe}">
            <div class="status-title">{titulo}</div>
            <div class="status-value">{valor}</div>
            <div class="small-note">{nota}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def banner_motor(resultado, motor):
    status = resultado.get("health_status")
    prioridade = resultado.get("priority_class")
    rul = resultado.get("rul_estimado")

    classe = classe_banner(status)

    if resultado.get("status_pipeline") != "ANALISE_COMPLETA":
        titulo = f"Motor {motor} • ANÁLISE PARCIAL"

        subtitulo = (
            f"RUL estimado: {numero(rul)} ciclos • "
            "Histórico ainda insuficiente para Health Score e prioridade."
        )

    else:
        titulo = (
            f"Motor {motor} • "
            f"{status_legivel(status)} • "
            f"{prioridade_legivel(prioridade)}"
        )

        subtitulo = (
            f"RUL estimado: {numero(rul)} ciclos • "
            f"Health Score: {numero(resultado.get('health_score'))}/100 • "
            f"Priority Score: {numero(resultado.get('priority_score'))}/100"
        )

    st.markdown(
        f"""
        <div class="motor-banner {classe}">
            <div class="motor-banner-title">
                {titulo}
            </div>
            <div class="motor-banner-subtitle">
                {subtitulo}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# GRÁFICOS COLORIDOS
# ============================================================

def grafico_condicao(
    saudaveis,
    atencao,
    risco,
    criticos,
):
    df = pd.DataFrame(
        {
            "Condição": [
                "Saudável",
                "Atenção",
                "Risco",
                "Crítico",
            ],
            "Quantidade": [
                saudaveis,
                atencao,
                risco,
                criticos,
            ],
        }
    )

    ordem = [
        "Saudável",
        "Atenção",
        "Risco",
        "Crítico",
    ]

    cores = [
        "#22c55e",
        "#eab308",
        "#f97316",
        "#ef4444",
    ]

    barras = (
        alt.Chart(df)
        .mark_bar(
            cornerRadiusTopLeft=5,
            cornerRadiusTopRight=5,
        )
        .encode(
            x=alt.X(
                "Condição:N",
                sort=ordem,
                title=None,
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y(
                "Quantidade:Q",
                title="Quantidade",
            ),
            color=alt.Color(
                "Condição:N",
                scale=alt.Scale(
                    domain=ordem,
                    range=cores,
                ),
                legend=None,
            ),
            tooltip=[
                "Condição:N",
                "Quantidade:Q",
            ],
        )
    )

    textos = (
        alt.Chart(df)
        .mark_text(
            dy=-10,
            fontSize=14,
            fontWeight="bold",
        )
        .encode(
            x=alt.X(
                "Condição:N",
                sort=ordem,
            ),
            y="Quantidade:Q",
            text="Quantidade:Q",
        )
    )

    return (
        barras + textos
    ).properties(
        height=300
    )


def grafico_prioridade(
    p4,
    p3,
    p2,
    p1,
):
    df = pd.DataFrame(
        {
            "Prioridade": [
                "P4 - Baixa",
                "P3 - Média",
                "P2 - Alta",
                "P1 - Imediata",
            ],
            "Quantidade": [
                p4,
                p3,
                p2,
                p1,
            ],
        }
    )

    ordem = [
        "P4 - Baixa",
        "P3 - Média",
        "P2 - Alta",
        "P1 - Imediata",
    ]

    cores = [
        "#22c55e",
        "#eab308",
        "#f97316",
        "#ef4444",
    ]

    barras = (
        alt.Chart(df)
        .mark_bar(
            cornerRadiusTopLeft=5,
            cornerRadiusTopRight=5,
        )
        .encode(
            x=alt.X(
                "Prioridade:N",
                sort=ordem,
                title=None,
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y(
                "Quantidade:Q",
                title="Quantidade",
            ),
            color=alt.Color(
                "Prioridade:N",
                scale=alt.Scale(
                    domain=ordem,
                    range=cores,
                ),
                legend=None,
            ),
            tooltip=[
                "Prioridade:N",
                "Quantidade:Q",
            ],
        )
    )

    textos = (
        alt.Chart(df)
        .mark_text(
            dy=-10,
            fontSize=14,
            fontWeight="bold",
        )
        .encode(
            x=alt.X(
                "Prioridade:N",
                sort=ordem,
            ),
            y="Quantidade:Q",
            text="Quantidade:Q",
        )
    )

    return (
        barras + textos
    ).properties(
        height=300
    )


# ============================================================
# INTERPRETAÇÃO
# ============================================================

def interpretar_status(status):
    if status == "SAUDAVEL":
        return (
            "O conjunto de indicadores aponta condição "
            "operacional favorável no momento."
        )

    if status == "ATENCAO":
        return (
            "O equipamento apresenta sinais que justificam "
            "acompanhamento mais próximo."
        )

    if status == "RISCO":
        return (
            "O equipamento apresenta degradação relevante. "
            "É recomendável planejar uma intervenção."
        )

    if status == "CRITICO":
        return (
            "Os indicadores combinados apontam condição crítica "
            "e necessidade de priorização operacional."
        )

    return (
        "O histórico disponível ainda não permite calcular "
        "todos os indicadores do pipeline."
    )


def interpretar_prioridade(prioridade):
    if prioridade == "P1_IMEDIATA":
        return (
            "Prioridade imediata. O equipamento deve aparecer "
            "no topo da fila de avaliação/manutenção."
        )

    if prioridade == "P2_ALTA":
        return (
            "Prioridade alta. Recomenda-se planejamento de "
            "intervenção em curto prazo."
        )

    if prioridade == "P3_MEDIA":
        return (
            "Prioridade média. Intensificar acompanhamento e "
            "considerar manutenção programada."
        )

    if prioridade == "P4_BAIXA":
        return (
            "Prioridade baixa. O equipamento pode permanecer "
            "no planejamento normal de manutenção."
        )

    return (
        "Prioridade operacional ainda indisponível devido "
        "à análise parcial."
    )


# ============================================================
# EXPLICABILIDADE
# ============================================================

def calcular_explicabilidade(resultado):
    if resultado.get("health_score") is None:
        return None

    rul_score = float(resultado["rul_score"])
    anomaly_score = float(resultado["anomaly_score"])
    trend_score = float(resultado["trend_score"])
    health_score = float(resultado["health_score"])

    rul_risk = 100 - rul_score
    health_risk = 100 - health_score

    impacto_health_rul = 0.60 * rul_risk
    impacto_health_anomalia = 0.25 * anomaly_score
    impacto_health_trend = 0.15 * trend_score

    impacto_priority_health = 0.30 * health_risk
    impacto_priority_rul = 0.70 * rul_risk

    impactos = {
        "RUL": impacto_health_rul,
        "Anomalia": impacto_health_anomalia,
        "Tendência": impacto_health_trend,
    }

    fator_principal = max(
        impactos,
        key=impactos.get,
    )

    return {
        "rul_risk": rul_risk,
        "health_risk": health_risk,
        "impacto_health_rul": impacto_health_rul,
        "impacto_health_anomalia": impacto_health_anomalia,
        "impacto_health_trend": impacto_health_trend,
        "impacto_priority_health": impacto_priority_health,
        "impacto_priority_rul": impacto_priority_rul,
        "fator_principal": fator_principal,
    }


# ============================================================
# HISTÓRICO DO PIPELINE
# ============================================================

@st.cache_data(show_spinner=False)
def gerar_historico_motor(
    unit_number,
    max_pontos=70,
):
    df = carregar_dataset()
    pipeline = carregar_pipeline()

    dados_motor = (
        df[
            df["unit_number"]
            == int(unit_number)
        ]
        .sort_values("time_in_cycles")
        .reset_index(drop=True)
    )

    quantidade = len(dados_motor)

    if quantidade < 30:
        return pd.DataFrame()

    endpoints = np.arange(
        30,
        quantidade + 1,
    )

    if len(endpoints) > max_pontos:
        indices = np.linspace(
            0,
            len(endpoints) - 1,
            max_pontos,
            dtype=int,
        )

        endpoints = np.unique(
            endpoints[indices]
        )

        if quantidade not in endpoints:
            endpoints = np.append(
                endpoints,
                quantidade,
            )

    linhas = []

    for fim in endpoints:
        inicio = max(
            0,
            fim - 39,
        )

        recorte = (
            dados_motor
            .iloc[inicio:fim]
            .copy()
        )

        resultado = (
            pipeline.analisar_motor(
                recorte,
                int(unit_number),
            )
        )

        linhas.append(
            {
                "cycle": int(
                    recorte[
                        "time_in_cycles"
                    ].iloc[-1]
                ),

                "rul_estimado":
                    resultado.get(
                        "rul_estimado"
                    ),

                "rul_score":
                    resultado.get(
                        "rul_score"
                    ),

                "anomaly_score":
                    resultado.get(
                        "anomaly_score"
                    ),

                "trend_score":
                    resultado.get(
                        "trend_score"
                    ),

                "health_score":
                    resultado.get(
                        "health_score"
                    ),

                "priority_score":
                    resultado.get(
                        "priority_score"
                    ),

                "health_status":
                    resultado.get(
                        "health_status"
                    ),

                "priority_class":
                    resultado.get(
                        "priority_class"
                    ),

                "status_pipeline":
                    resultado.get(
                        "status_pipeline"
                    ),
            }
        )

    return pd.DataFrame(linhas)


# ============================================================
# CARREGAMENTO
# ============================================================

try:
    pipeline = carregar_pipeline()
    df_test = carregar_dataset()
    df_resultados = carregar_resultados()
    df_ranking = carregar_ranking()

except Exception as erro:
    st.error(
        "Não foi possível carregar o sistema."
    )

    st.exception(erro)
    st.stop()


# ============================================================
# PREPARAÇÃO DO RANKING
# ============================================================

ranking = (
    df_ranking
    .sort_values(
        [
            "priority_score",
            "health_score",
            "rul_estimado",
        ],
        ascending=[
            False,
            True,
            True,
        ],
    )
    .reset_index(drop=True)
)

ranking[
    "ranking_prioridade"
] = np.arange(
    1,
    len(ranking) + 1,
)


# ============================================================
# HERO
# ============================================================

st.title(
    "⚙️ Plataforma de Manutenção Preditiva"
)

st.caption(
    "Monitoramento de condição, RUL, anomalias, "
    "Health Score e priorização operacional "
    "• NASA C-MAPSS FD001"
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("UPX 2.0")

    st.caption(
        "Pipeline FD001 v1.0"
    )

    st.divider()

    st.write("**Dataset**")
    st.write("NASA C-MAPSS FD001")

    st.write("**Equipamentos**")
    st.write(
        "100 motores turbofan simulados"
    )

    st.write("**Janela temporal**")
    st.write("30 ciclos")

    st.write("**Sensores utilizados**")
    st.write("14")

    st.divider()

    st.info(
        "Os ciclos do C-MAPSS não representam "
        "diretamente horas ou dias."
    )


# ============================================================
# ABAS
# ============================================================

aba_frota, aba_motor, aba_modelo = st.tabs(
    [
        "📊 Dashboard Geral",
        "🔍 Análise do Motor",
        "🧠 Modelo e Metodologia",
    ]
)


# ============================================================
# ABA 1 - DASHBOARD GERAL
# ============================================================

with aba_frota:

    st.header(
        "Visão geral da frota"
    )

    total = len(df_resultados)

    completas = int(
        (
            df_resultados[
                "status_pipeline"
            ]
            == "ANALISE_COMPLETA"
        ).sum()
    )

    parciais = int(
        (
            df_resultados[
                "status_pipeline"
            ]
            == "ANALISE_PARCIAL"
        ).sum()
    )

    criticos = int(
        (
            df_resultados[
                "health_status"
            ]
            == "CRITICO"
        ).sum()
    )

    p1 = int(
        (
            df_resultados[
                "priority_class"
            ]
            == "P1_IMEDIATA"
        ).sum()
    )

    health_medio = (
        df_resultados[
            "health_score"
        ]
        .dropna()
        .mean()
    )

    # ========================================================
    # KPIs
    # ========================================================

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    c1.metric(
        "Frota",
        total,
    )

    c2.metric(
        "Análises completas",
        completas,
    )

    c3.metric(
        "Análises parciais",
        parciais,
    )

    c4.metric(
        "Críticos",
        criticos,
    )

    c5.metric(
        "Prioridade P1",
        p1,
    )

    c6.metric(
        "Health médio",
        f"{health_medio:.1f}",
    )

    st.divider()

    # ========================================================
    # CONDIÇÃO
    # ========================================================

    st.subheader(
        "Condição atual"
    )

    saudaveis = int(
        (
            df_resultados[
                "health_status"
            ]
            == "SAUDAVEL"
        ).sum()
    )

    atencao = int(
        (
            df_resultados[
                "health_status"
            ]
            == "ATENCAO"
        ).sum()
    )

    risco = int(
        (
            df_resultados[
                "health_status"
            ]
            == "RISCO"
        ).sum()
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        card_html(
            "SAUDÁVEL",
            saudaveis,
            "Health Score ≥ 80",
            "status-green",
        )

    with c2:
        card_html(
            "ATENÇÃO",
            atencao,
            "Health Score entre 60 e 80",
            "status-yellow",
        )

    with c3:
        card_html(
            "RISCO",
            risco,
            "Health Score entre 30 e 60",
            "status-orange",
        )

    with c4:
        card_html(
            "CRÍTICO",
            criticos,
            "Health Score abaixo de 30",
            "status-red",
        )

    # ========================================================
    # PRIORIDADE
    # ========================================================

    st.subheader(
        "Prioridade de manutenção"
    )

    p4 = int(
        (
            df_resultados[
                "priority_class"
            ]
            == "P4_BAIXA"
        ).sum()
    )

    p3 = int(
        (
            df_resultados[
                "priority_class"
            ]
            == "P3_MEDIA"
        ).sum()
    )

    p2 = int(
        (
            df_resultados[
                "priority_class"
            ]
            == "P2_ALTA"
        ).sum()
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        card_html(
            "P4 - BAIXA",
            p4,
            "Priority Score < 20",
            "status-green",
        )

    with c2:
        card_html(
            "P3 - MÉDIA",
            p3,
            "Priority Score entre 20 e 45",
            "status-yellow",
        )

    with c3:
        card_html(
            "P2 - ALTA",
            p2,
            "Priority Score entre 45 e 75",
            "status-orange",
        )

    with c4:
        card_html(
            "P1 - IMEDIATA",
            p1,
            "Priority Score ≥ 75",
            "status-red",
        )

    st.divider()

    # ========================================================
    # GRÁFICOS SEMÂNTICOS
    # ========================================================

    graf1, graf2 = st.columns(2)

    with graf1:

        st.subheader(
            "Distribuição por condição"
        )

        st.altair_chart(
            grafico_condicao(
                saudaveis,
                atencao,
                risco,
                criticos,
            ),
            use_container_width=True,
        )

    with graf2:

        st.subheader(
            "Distribuição por prioridade"
        )

        st.altair_chart(
            grafico_prioridade(
                p4,
                p3,
                p2,
                p1,
            ),
            use_container_width=True,
        )

    st.divider()

    # ========================================================
    # TOP 15
    # ========================================================

    st.subheader(
        "🚨 Top 15 equipamentos por urgência operacional"
    )

    st.caption(
        "Ordenação baseada no Priority Score. "
        "O equipamento #1 é o que exige maior atenção "
        "entre as análises completas."
    )

    top15 = (
        ranking
        .head(15)
        .copy()
    )

    top15 = top15[
        [
            "ranking_prioridade",
            "unit_number",
            "rul_estimado",
            "anomaly_score",
            "trend_score",
            "health_score",
            "health_status",
            "priority_score",
            "priority_class",
        ]
    ]

    top15 = top15.rename(
        columns={
            "ranking_prioridade":
                "Ranking",

            "unit_number":
                "Motor",

            "rul_estimado":
                "RUL",

            "anomaly_score":
                "Anomaly",

            "trend_score":
                "Trend",

            "health_score":
                "Health",

            "health_status":
                "Status",

            "priority_score":
                "Priority Score",

            "priority_class":
                "Prioridade",
        }
    )

    for coluna in [
        "RUL",
        "Anomaly",
        "Trend",
        "Health",
        "Priority Score",
    ]:
        top15[coluna] = (
            top15[coluna]
            .round(1)
        )

    top15["Status"] = (
        top15["Status"]
        .apply(status_legivel)
    )

    top15["Prioridade"] = (
        top15["Prioridade"]
        .apply(prioridade_legivel)
    )

    st.dataframe(
        top15,
        use_container_width=True,
        hide_index=True,
    )

    with st.expander(
        "📋 Ver ranking completo da frota"
    ):

        ranking_completo = (
            ranking[
                [
                    "ranking_prioridade",
                    "unit_number",
                    "rul_estimado",
                    "health_score",
                    "health_status",
                    "priority_score",
                    "priority_class",
                ]
            ]
            .copy()
        )

        ranking_completo = (
            ranking_completo.rename(
                columns={
                    "ranking_prioridade":
                        "Ranking",

                    "unit_number":
                        "Motor",

                    "rul_estimado":
                        "RUL",

                    "health_score":
                        "Health",

                    "health_status":
                        "Status",

                    "priority_score":
                        "Priority Score",

                    "priority_class":
                        "Prioridade",
                }
            )
        )

        ranking_completo[
            "RUL"
        ] = ranking_completo[
            "RUL"
        ].round(1)

        ranking_completo[
            "Health"
        ] = ranking_completo[
            "Health"
        ].round(1)

        ranking_completo[
            "Priority Score"
        ] = ranking_completo[
            "Priority Score"
        ].round(1)

        ranking_completo[
            "Status"
        ] = ranking_completo[
            "Status"
        ].apply(
            status_legivel
        )

        ranking_completo[
            "Prioridade"
        ] = ranking_completo[
            "Prioridade"
        ].apply(
            prioridade_legivel
        )

        st.dataframe(
            ranking_completo,
            use_container_width=True,
            hide_index=True,
            height=550,
        )

    # ========================================================
    # PARCIAIS
    # ========================================================

    st.divider()

    st.subheader(
        "Análises parciais"
    )

    parciais_df = (
        df_resultados[
            df_resultados[
                "status_pipeline"
            ]
            == "ANALISE_PARCIAL"
        ]
        .copy()
    )

    if not parciais_df.empty:

        st.warning(
            f"{len(parciais_df)} motores possuem RUL e "
            "análise de anomalia disponíveis, porém ainda "
            "não possuem histórico suficiente para calcular "
            "o Trend Score de 10 janelas. Por segurança, "
            "eles não recebem Health Score nem prioridade."
        )

        exibir_parciais = (
            parciais_df[
                [
                    "unit_number",
                    "ciclo_atual",
                    "rul_estimado",
                    "anomaly_score",
                    "status_pipeline",
                ]
            ]
            .rename(
                columns={
                    "unit_number":
                        "Motor",

                    "ciclo_atual":
                        "Ciclo atual",

                    "rul_estimado":
                        "RUL estimado",

                    "anomaly_score":
                        "Anomaly Score",

                    "status_pipeline":
                        "Situação",
                }
            )
        )

        st.dataframe(
            exibir_parciais,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# ABA 2 - ANÁLISE DO MOTOR
# ============================================================

with aba_motor:

    st.header(
        "Análise individual do equipamento"
    )

    motores = sorted(
        df_test[
            "unit_number"
        ]
        .unique()
        .tolist()
    )

    motor_selecionado = st.selectbox(
        "Selecione o motor",
        motores,
        index=1,
    )

    with st.spinner(
        f"Executando pipeline para o motor "
        f"{motor_selecionado}..."
    ):

        resultado = (
            pipeline.analisar_motor(
                df_test,
                int(motor_selecionado),
            )
        )

    dados_motor = (
        df_test[
            df_test[
                "unit_number"
            ]
            == motor_selecionado
        ]
        .sort_values(
            "time_in_cycles"
        )
        .copy()
    )

    # ========================================================
    # BANNER PRINCIPAL
    # ========================================================

    banner_motor(
        resultado,
        motor_selecionado,
    )

    if (
        resultado["status_pipeline"]
        == "ANALISE_COMPLETA"
    ):
        st.success(
            "Análise preditiva completa disponível."
        )

    elif (
        resultado["status_pipeline"]
        == "ANALISE_PARCIAL"
    ):
        st.warning(
            resultado.get(
                "mensagem",
                "Análise parcial.",
            )
        )

    else:
        st.error(
            resultado.get(
                "mensagem",
                "Dados insuficientes.",
            )
        )

    # ========================================================
    # CARDS PRINCIPAIS
    # ========================================================

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        card_html(
            "RUL ESTIMADO",
            (
                f"{numero(resultado['rul_estimado'])} ciclos"
                if resultado["rul_estimado"] is not None
                else "—"
            ),
            "Vida útil remanescente estimada",
            "status-blue",
        )

    with c2:
        card_html(
            "HEALTH SCORE",
            (
                f"{numero(resultado['health_score'])}/100"
                if resultado["health_score"] is not None
                else "—"
            ),
            "100 representa melhor condição",
            classe_status(
                resultado["health_status"]
            ),
        )

    with c3:
        card_html(
            "STATUS",
            status_legivel(
                resultado["health_status"]
            ),
            "Condição técnica combinada",
            classe_status(
                resultado["health_status"]
            ),
        )

    with c4:
        card_html(
            "PRIORIDADE",
            prioridade_legivel(
                resultado["priority_class"]
            ),
            "Urgência operacional",
            classe_prioridade(
                resultado["priority_class"]
            ),
        )

    st.divider()

    # ========================================================
    # SCORES
    # ========================================================

    st.subheader(
        "Indicadores do pipeline"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "RUL Score",
        (
            f"{numero(resultado['rul_score'])}/100"
            if resultado["rul_score"] is not None
            else "—"
        ),
    )

    c2.metric(
        "Anomaly Score",
        (
            f"{numero(resultado['anomaly_score'])}/100"
            if resultado["anomaly_score"] is not None
            else "—"
        ),
    )

    c3.metric(
        "Trend Score",
        (
            f"{numero(resultado['trend_score'])}/100"
            if resultado["trend_score"] is not None
            else "—"
        ),
    )

    c4.metric(
        "Priority Score",
        (
            f"{numero(resultado['priority_score'])}/100"
            if resultado["priority_score"] is not None
            else "—"
        ),
    )

    # ========================================================
    # RANKING DO MOTOR
    # ========================================================

    if (
        resultado["status_pipeline"]
        == "ANALISE_COMPLETA"
    ):

        linha_ranking = (
            ranking[
                ranking[
                    "unit_number"
                ]
                == motor_selecionado
            ]
        )

        if not linha_ranking.empty:

            posicao = int(
                linha_ranking[
                    "ranking_prioridade"
                ].iloc[0]
            )

            st.info(
                f"📍 Este equipamento ocupa a posição "
                f"**#{posicao}** no ranking de urgência "
                f"da frota entre os motores com "
                f"análise completa."
            )

    # ========================================================
    # INTERPRETAÇÃO
    # ========================================================

    st.subheader(
        "Interpretação operacional"
    )

    st.markdown(
        f"""
        <div class="recommendation">

            <div class="recommendation-title">
                Condição atual
            </div>

            {interpretar_status(
                resultado["health_status"]
            )}

            <br><br>

            <div class="recommendation-title">
                Ação sugerida
            </div>

            {interpretar_prioridade(
                resultado["priority_class"]
            )}

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    st.write(
        "**Análise de anomalia:** "
        + str(
            resultado[
                "interpretacao_anomalia"
            ]
        )
    )

    if resultado[
        "anomalia_detectada"
    ]:

        st.warning(
            "O erro de reconstrução atual ultrapassa "
            "o limiar estatístico definido a partir "
            "do comportamento saudável aprendido."
        )

    else:

        st.info(
            "O erro de reconstrução atual permanece "
            "abaixo do limiar estatístico de anomalia."
        )

    # ========================================================
    # EXPLICABILIDADE
    # ========================================================

    explicacao = calcular_explicabilidade(
        resultado
    )

    if explicacao is not None:

        st.divider()

        st.subheader(
            "Por que o sistema chegou a essa conclusão?"
        )

        st.caption(
            "A explicação representa a contribuição dos "
            "indicadores na fórmula do Health Score. "
            "Ela não representa importância causal dos "
            "sensores individuais."
        )

        fator = explicacao[
            "fator_principal"
        ]

        st.info(
            f"O maior impacto atual sobre a perda de "
            f"saúde do equipamento vem do componente: "
            f"**{fator}**."
        )

        impacto_df = pd.DataFrame(
            {
                "Componente": [
                    "RUL",
                    "Anomalia",
                    "Tendência",
                ],

                "Impacto": [
                    explicacao[
                        "impacto_health_rul"
                    ],

                    explicacao[
                        "impacto_health_anomalia"
                    ],

                    explicacao[
                        "impacto_health_trend"
                    ],
                ],
            }
        )

        ordem_impacto = [
            "RUL",
            "Anomalia",
            "Tendência",
        ]

        grafico_impacto = (
            alt.Chart(
                impacto_df
            )
            .mark_bar(
                cornerRadiusTopLeft=5,
                cornerRadiusTopRight=5,
            )
            .encode(
                x=alt.X(
                    "Componente:N",
                    sort=ordem_impacto,
                    title=None,
                    axis=alt.Axis(
                        labelAngle=0
                    ),
                ),
                y=alt.Y(
                    "Impacto:Q",
                    title="Impacto na perda de Health",
                ),
                tooltip=[
                    "Componente:N",
                    alt.Tooltip(
                        "Impacto:Q",
                        format=".2f",
                    ),
                ],
            )
            .properties(
                height=280
            )
        )

        st.altair_chart(
            grafico_impacto,
            use_container_width=True,
        )

        st.caption(
            "Quanto maior a barra, maior a contribuição "
            "desse componente para reduzir o Health Score "
            "em relação ao máximo de 100."
        )

        c1, c2 = st.columns(2)

        with c1:

            st.markdown(
                "#### Composição do risco técnico"
            )

            st.write(
                f"**RUL Risk:** "
                f"{explicacao['rul_risk']:.1f}/100"
            )

            st.write(
                f"**Health Risk:** "
                f"{explicacao['health_risk']:.1f}/100"
            )

        with c2:

            st.markdown(
                "#### Composição da prioridade"
            )

            st.write(
                "• 30% do Priority Score vem "
                "do Health Risk."
            )

            st.write(
                "• 70% do Priority Score vem "
                "do RUL Risk."
            )

    # ========================================================
    # HISTÓRICO
    # ========================================================

    st.divider()

    st.subheader(
        "Evolução temporal do equipamento"
    )

    st.caption(
        "Os indicadores abaixo são reconstruídos ao longo "
        "do histórico observado do motor. Quando o histórico "
        "é longo, o sistema utiliza uma amostragem de pontos "
        "para reduzir processamento."
    )

    with st.spinner(
        "Reconstruindo histórico preditivo..."
    ):

        historico = gerar_historico_motor(
            int(motor_selecionado)
        )

    if historico.empty:

        st.warning(
            "Não há pelo menos 30 ciclos disponíveis "
            "para gerar o histórico preditivo."
        )

    else:

        # ----------------------------------------------------
        # RUL
        # ----------------------------------------------------

        st.markdown(
            "#### RUL estimado"
        )

        graf_rul = (
            historico[
                [
                    "cycle",
                    "rul_estimado",
                ]
            ]
            .dropna()
            .set_index("cycle")
        )

        if not graf_rul.empty:
            st.line_chart(
                graf_rul
            )

        # ----------------------------------------------------
        # ANOMALIA
        # ----------------------------------------------------

        st.markdown(
            "#### Anomaly Score"
        )

        graf_anomaly = (
            historico[
                [
                    "cycle",
                    "anomaly_score",
                ]
            ]
            .dropna()
            .set_index("cycle")
        )

        if not graf_anomaly.empty:
            st.line_chart(
                graf_anomaly
            )

        # ----------------------------------------------------
        # TREND / HEALTH
        # ----------------------------------------------------

        c1, c2 = st.columns(2)

        with c1:

            st.markdown(
                "#### Trend Score"
            )

            graf_trend = (
                historico[
                    [
                        "cycle",
                        "trend_score",
                    ]
                ]
                .dropna()
                .set_index("cycle")
            )

            if not graf_trend.empty:

                st.line_chart(
                    graf_trend
                )

            else:

                st.info(
                    "Histórico ainda insuficiente "
                    "para Trend Score."
                )

        with c2:

            st.markdown(
                "#### Health Score"
            )

            graf_health = (
                historico[
                    [
                        "cycle",
                        "health_score",
                    ]
                ]
                .dropna()
                .set_index("cycle")
            )

            if not graf_health.empty:

                st.line_chart(
                    graf_health
                )

            else:

                st.info(
                    "Health Score ainda indisponível."
                )

        # ----------------------------------------------------
        # PRIORITY
        # ----------------------------------------------------

        st.markdown(
            "#### Priority Score"
        )

        graf_priority = (
            historico[
                [
                    "cycle",
                    "priority_score",
                ]
            ]
            .dropna()
            .set_index("cycle")
        )

        if not graf_priority.empty:

            st.line_chart(
                graf_priority
            )

    # ========================================================
    # SENSORES
    # ========================================================

    st.divider()

    st.subheader(
        "Sensores do equipamento"
    )

    st.caption(
        "Selecione os sensores para acompanhar "
        "a evolução dos valores brutos."
    )

    sensores = pipeline.sensores

    default_sensores = [
        sensor
        for sensor in [
            "sensor_2",
            "sensor_4",
            "sensor_11",
            "sensor_15",
        ]
        if sensor in sensores
    ]

    sensores_selecionados = (
        st.multiselect(
            "Sensores exibidos",
            sensores,
            default=default_sensores,
        )
    )

    if sensores_selecionados:

        grafico_sensores = (
            dados_motor[
                ["time_in_cycles"]
                + sensores_selecionados
            ]
            .set_index(
                "time_in_cycles"
            )
        )

        st.line_chart(
            grafico_sensores
        )

        st.caption(
            "Atenção: os sensores podem possuir escalas "
            "diferentes. Este gráfico apresenta os valores "
            "brutos do C-MAPSS."
        )

    # ========================================================
    # DADOS TÉCNICOS
    # ========================================================

    with st.expander(
        "🔧 Dados técnicos completos"
    ):

        dados_tecnicos = {
            "motor":
                resultado.get(
                    "unit_number"
                ),

            "ciclo_atual":
                resultado.get(
                    "ciclo_atual"
                ),

            "ciclos_disponiveis":
                resultado.get(
                    "ciclos_disponiveis"
                ),

            "rul_estimado":
                resultado.get(
                    "rul_estimado"
                ),

            "rul_pred_raw":
                resultado.get(
                    "rul_pred_raw"
                ),

            "rul_score":
                resultado.get(
                    "rul_score"
                ),

            "erro_reconstrucao":
                resultado.get(
                    "erro_reconstrucao"
                ),

            "anomaly_score":
                resultado.get(
                    "anomaly_score"
                ),

            "anomalia_detectada":
                resultado.get(
                    "anomalia_detectada"
                ),

            "trend_slope":
                resultado.get(
                    "trend_slope"
                ),

            "trend_score":
                resultado.get(
                    "trend_score"
                ),

            "health_score":
                resultado.get(
                    "health_score"
                ),

            "health_status":
                resultado.get(
                    "health_status"
                ),

            "priority_score":
                resultado.get(
                    "priority_score"
                ),

            "priority_class":
                resultado.get(
                    "priority_class"
                ),

            "status_pipeline":
                resultado.get(
                    "status_pipeline"
                ),
        }

        st.json(
            dados_tecnicos
        )


# ============================================================
# ABA 3 - MODELO E METODOLOGIA
# ============================================================

with aba_modelo:

    st.header(
        "Modelo e metodologia"
    )

    st.info(
        "O módulo FD001 foi desenvolvido como benchmark "
        "experimental de prognóstico e manutenção preditiva."
    )

    st.markdown(
        """
### 1. Entrada do sistema

Cada análise utiliza:

- **30 ciclos temporais**
- **14 sensores selecionados**
- dados do NASA C-MAPSS FD001

A unidade analisada representa um **motor turbofan simulado**.

---

### 2. Estimativa de RUL

O modelo principal de RUL é uma **LSTM**.

Durante o treinamento foi utilizado RUL limitado a
**125 ciclos** para a região inicial da vida do equipamento.

Na operação, o sistema retorna:

- RUL estimado;
- RUL Score;
- risco associado ao RUL.

---

### 3. Detecção de anomalia

A detecção de anomalia utiliza um
**LSTM Autoencoder**.

O modelo foi treinado em uma região considerada
proxy de comportamento saudável.

O erro de reconstrução é transformado em um
**Anomaly Score de 0 a 100**.

O modelo de anomalia recebe somente dados dos sensores
durante a inferência.

---

### 4. Tendência

A tendência é calculada sobre o histórico recente
do erro de reconstrução do Autoencoder.

São utilizadas **10 janelas consecutivas**.

Uma regressão linear produz o slope da tendência,
que é convertido em **Trend Score**.

---

### 5. Health Score

A fórmula congelada na versão 1.0 é:

`Health = 0.60 × RUL Score + 0.25 × (100 − Anomaly Score) + 0.15 × (100 − Trend Score)`

Interpretação:

- **80–100:** Saudável
- **60–79:** Atenção
- **30–59:** Risco
- **0–29:** Crítico

---

### 6. Priority Score

A prioridade representa urgência operacional,
e não exatamente a mesma coisa que condição técnica.

Primeiro:

`Health Risk = 100 − Health Score`

`RUL Risk = 100 − RUL Score`

Depois:

`Priority = 0.30 × Health Risk + 0.70 × RUL Risk`

Classes:

- **P4:** score < 20
- **P3:** 20 ≤ score < 45
- **P2:** 45 ≤ score < 75
- **P1:** score ≥ 75

---

### 7. Análise parcial

O RUL e a anomalia precisam de pelo menos
**30 ciclos**.

O Trend Score exige **10 janelas de anomalia**.

Por isso, a primeira análise completa ocorre
a partir de aproximadamente **39 ciclos observados**.

Antes disso, o sistema não inventa um Trend Score:
a análise é explicitamente marcada como parcial.
"""
    )

    st.divider()

    st.subheader(
        "Validação integrada"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Motores",
        "100",
    )

    c2.metric(
        "MAE RUL",
        "11.123 ciclos",
    )

    c3.metric(
        "RMSE RUL",
        "15.079 ciclos",
    )

    c4.metric(
        "Análises completas",
        "97%",
    )

    st.caption(
        "As métricas correspondem ao benchmark integrado "
        "no conjunto test_FD001."
    )

    st.warning(
        "O conjunto oficial de teste já foi utilizado "
        "durante o desenvolvimento para comparação entre "
        "configurações. Portanto, ele não deve ser descrito "
        "como um holdout completamente intocado."
    )

    st.divider()

    st.subheader(
        "Limitações importantes"
    )

    st.markdown(
        """
- O FD001 contém **um único modo de degradação**.
- O sistema não deve inventar diagnósticos de rolamento,
  óleo, combustível ou outros modos de falha ausentes no dataset.
- O RUL representa **ciclos do benchmark**, não horas ou dias.
- O Health Score e o Priority Score são indicadores
  construídos para esta versão experimental.
- Os resultados devem ser descritos como estimativas
  baseadas nos dados disponíveis.
- O módulo FD001 não deve ser aplicado diretamente
  a veículos, trens ou outros ativos sem modelos
  e calibrações específicos.
"""
    )

    st.divider()

    st.subheader(
        "Arquitetura da plataforma"
    )

    st.code(
        """
Sensores
   ↓
Pré-processamento
   ↓
┌───────────────┬─────────────────────┐
│ LSTM RUL      │ LSTM Autoencoder    │
└───────┬───────┴──────────┬──────────┘
        │                  │
        │             Anomaly Score
        │                  │
        │              Trend Score
        │                  │
        └────────┬─────────┘
                 ↓
            Health Score
                 ↓
               Status
                 ↓
           Priority Score
                 ↓
          P1 / P2 / P3 / P4
                 ↓
              Dashboard
"""
    )


# ============================================================
# RODAPÉ
# ============================================================

st.divider()

st.caption(
    "UPX 2.0 • Plataforma modular de manutenção preditiva "
    "• Módulo NASA C-MAPSS FD001 • Pipeline v1.0"
)