from __future__ import annotations

import os
import sys
from pathlib import Path

# ============================================================
# RAIZ DO PROJETO
# ============================================================

RAIZ_PROJETO = Path(__file__).resolve().parents[2]

if str(RAIZ_PROJETO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROJETO))

os.environ["TF_USE_LEGACY_KERAS"] = "1"


# ============================================================
# IMPORTS
# ============================================================

import altair as alt
import pandas as pd
import streamlit as st

from utils.pipeline_automotivo import (
    PipelineAutomotivo,
)

from utils.simulador_automotivo import (
    CENARIOS_DISPONIVEIS,
    ConfigSimulacao,
    gerar_telemetria,
)

from utils.processador_sessoes_automotivas import (
    processar_sessoes_csv,
)


# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================



# ============================================================
# LABELS
# ============================================================

STATUS_LABELS = {
    "NORMAL": "NORMAL",
    "ATENCAO": "ATENÇÃO",
    "DESVIO_RELEVANTE": "DESVIO RELEVANTE",
    "DESVIO_ELEVADO": "DESVIO ELEVADO",
}

CENARIO_LABELS = {
    "NORMAL": "Normal",
    "AQUECIMENTO_PROGRESSIVO": "Aquecimento progressivo",
    "OPERACAO_IRREGULAR": "Operação irregular",
    "CONDICAO_SEVERA_SIMULADA": "Condição severa simulada",
}

SENSORES_PRINCIPAIS = {
    "rpm": "RPM",
    "speed_kmh": "Velocidade (km/h)",
    "coolant_temp_c": "Temperatura do arrefecimento (°C)",
    "engine_load_pct": "Carga do motor (%)",
}

SENSORES_ADICIONAIS = {
    "throttle_pct": "Posição da borboleta (%)",
    "intake_temp_c": "Temperatura de admissão (°C)",
    "map_kpa": "Pressão MAP (kPa)",
    "battery_voltage": "Tensão da bateria (V)",
}


# ============================================================
# SESSION STATE
# ============================================================

if "resultado_automotivo" not in st.session_state:
    st.session_state.resultado_automotivo = None

if "descricao_origem_automotivo" not in st.session_state:
    st.session_state.descricao_origem_automotivo = None

if "lote_csv_automotivo" not in st.session_state:
    st.session_state.lote_csv_automotivo = None

if "fonte_anterior_automotivo" not in st.session_state:
    st.session_state.fonte_anterior_automotivo = None


# ============================================================
# CACHE
# ============================================================

@st.cache_resource
def carregar_pipeline() -> PipelineAutomotivo:
    return PipelineAutomotivo()


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def formatar_status(status: str) -> str:

    return STATUS_LABELS.get(
        status,
        status.replace("_", " "),
    )


def preparar_eixo_temporal(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, str, str]:

    dados = df.copy()

    if (
        "timestamp" in dados.columns
        and dados["timestamp"].notna().any()
    ):

        dados["timestamp"] = pd.to_datetime(
            dados["timestamp"],
            errors="coerce",
        )

        return (
            dados,
            "timestamp",
            "Tempo",
        )

    dados["amostra"] = range(
        1,
        len(dados) + 1,
    )

    return (
        dados,
        "amostra",
        "Amostra",
    )


def grafico_sensor(
    dados: pd.DataFrame,
    coluna: str,
    titulo: str,
) -> None:

    if coluna not in dados.columns:

        st.info(
            f"Sensor indisponível: {titulo}"
        )

        return

    df, eixo_x, titulo_x = preparar_eixo_temporal(
        dados
    )

    tipo_x = (
        f"{eixo_x}:T"
        if eixo_x == "timestamp"
        else f"{eixo_x}:Q"
    )

    grafico = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X(
                tipo_x,
                title=titulo_x,
            ),
            y=alt.Y(
                f"{coluna}:Q",
                title=titulo,
                scale=alt.Scale(
                    zero=False,
                ),
            ),
            tooltip=[
                alt.Tooltip(
                    tipo_x,
                    title=titulo_x,
                ),
                alt.Tooltip(
                    f"{coluna}:Q",
                    title=titulo,
                    format=".2f",
                ),
            ],
        )
        .properties(
            height=260,
            title=titulo,
        )
        .interactive()
    )

    st.altair_chart(
        grafico,
        use_container_width=True,
    )


def grafico_reconstruction_error(
    historico: pd.DataFrame,
    threshold: float,
) -> None:

    if "reconstruction_error" not in historico.columns:

        st.warning(
            "Histórico de Reconstruction Error indisponível."
        )

        return

    df, eixo_x, titulo_x = preparar_eixo_temporal(
        historico
    )

    tipo_x = (
        f"{eixo_x}:T"
        if eixo_x == "timestamp"
        else f"{eixo_x}:Q"
    )

    linha_erro = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X(
                tipo_x,
                title=titulo_x,
            ),
            y=alt.Y(
                "reconstruction_error:Q",
                title="Reconstruction Error",
                scale=alt.Scale(
                    zero=False,
                ),
            ),
            tooltip=[
                alt.Tooltip(
                    tipo_x,
                    title=titulo_x,
                ),
                alt.Tooltip(
                    "reconstruction_error:Q",
                    title="Erro",
                    format=".6f",
                ),
            ],
        )
    )

    df_threshold = pd.DataFrame(
        {
            "threshold": [
                threshold
            ]
        }
    )

    linha_threshold = (
        alt.Chart(df_threshold)
        .mark_rule(
            strokeDash=[8, 6],
        )
        .encode(
            y=alt.Y(
                "threshold:Q"
            )
        )
    )

    grafico = (
        linha_erro
        + linha_threshold
    ).properties(
        height=320,
        title=(
            "Reconstruction Error "
            "e threshold de anomalia"
        ),
    ).interactive()

    st.altair_chart(
        grafico,
        use_container_width=True,
    )


def grafico_anomaly_score(
    historico: pd.DataFrame,
) -> None:

    if "anomaly_score" not in historico.columns:

        st.warning(
            "Histórico de Anomaly Score indisponível."
        )

        return

    df, eixo_x, titulo_x = preparar_eixo_temporal(
        historico
    )

    tipo_x = (
        f"{eixo_x}:T"
        if eixo_x == "timestamp"
        else f"{eixo_x}:Q"
    )

    grafico = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X(
                tipo_x,
                title=titulo_x,
            ),
            y=alt.Y(
                "anomaly_score:Q",
                title="Anomaly Score",
                scale=alt.Scale(
                    domain=[0, 100],
                ),
            ),
            tooltip=[
                alt.Tooltip(
                    tipo_x,
                    title=titulo_x,
                ),
                alt.Tooltip(
                    "anomaly_score:Q",
                    title="Anomaly Score",
                    format=".2f",
                ),
            ],
        )
        .properties(
            height=300,
            title="Evolução do Anomaly Score",
        )
        .interactive()
    )

    st.altair_chart(
        grafico,
        use_container_width=True,
    )


def mostrar_dashboard_resultado(
    resultado: dict,
    descricao_origem: str | None = None,
) -> None:

    # ========================================================
    # 2. SITUAÇÃO ATUAL
    # ========================================================

    st.divider()

    st.subheader(
        "2. Situação atual"
    )

    health = float(
        resultado[
            "automotive_health"
        ]
    )

    anomaly_score = float(
        resultado[
            "anomaly_score_atual"
        ]
    )

    persistencia = float(
        resultado[
            "taxa_anomalias_recente_pct"
        ]
    )

    status = resultado[
        "status"
    ]

    situacao_col1, situacao_col2, situacao_col3, situacao_col4 = (
        st.columns(4)
    )

    with situacao_col1:

        st.metric(
            label="Automotive Health",
            value=f"{health:.1f}",
            help=(
                "0 representa maior desvio em relação "
                "ao padrão aprendido. 100 representa "
                "maior proximidade ao padrão aprendido."
            ),
        )

        st.caption(
            "Proximidade ao padrão aprendido"
        )

    with situacao_col2:

        st.metric(
            label="Status",
            value=formatar_status(
                status
            ),
            help=(
                "Classificação do comportamento "
                "recente da telemetria."
            ),
        )

        st.caption(
            "Situação da telemetria recente"
        )

    with situacao_col3:

        st.metric(
            label="Anomaly Score",
            value=f"{anomaly_score:.1f}",
            help=(
                "Intensidade do desvio atual "
                "em relação ao padrão aprendido."
            ),
        )

        st.caption(
            "Intensidade do desvio atual"
        )

    with situacao_col4:

        st.metric(
            label="Persistência recente",
            value=f"{persistencia:.1f}%",
            help=(
                "Percentual das últimas janelas "
                "acima do threshold de anomalia."
            ),
        )

        st.caption(
            "Janelas recentes acima do threshold"
        )

    # ========================================================
    # 3. INTERPRETAÇÃO
    # ========================================================

    st.divider()

    st.subheader(
        "3. Interpretação"
    )

    if status == "NORMAL":

        st.success(
            resultado[
                "recomendacao"
            ]
        )

    elif status == "ATENCAO":

        st.warning(
            resultado[
                "recomendacao"
            ]
        )

    else:

        st.error(
            resultado[
                "recomendacao"
            ]
        )

    interpretacao_col1, interpretacao_col2 = (
        st.columns(2)
    )

    with interpretacao_col1:

        st.write(
            "**Origem dos dados:** "
            f"{resultado['origem_dados']}"
        )

        if descricao_origem:

            st.caption(
                descricao_origem
            )

    with interpretacao_col2:

        st.write(
            "**Modelo de referência:** "
            f"{resultado['modelo_referencia']}"
        )

    st.caption(
        "Automotive Health representa a proximidade "
        "da telemetria recente ao padrão aprendido. "
        "Não representa percentual de integridade "
        "mecânica, probabilidade de falha ou RUL."
    )

    # ========================================================
    # 4. TELEMETRIA
    # ========================================================

    st.divider()

    st.subheader(
        "4. Telemetria"
    )

    dados_processados = resultado[
        "dados_processados"
    ].copy()

    telemetria_linha1_col1, telemetria_linha1_col2 = (
        st.columns(2)
    )

    with telemetria_linha1_col1:

        grafico_sensor(
            dados_processados,
            "rpm",
            SENSORES_PRINCIPAIS[
                "rpm"
            ],
        )

    with telemetria_linha1_col2:

        grafico_sensor(
            dados_processados,
            "coolant_temp_c",
            SENSORES_PRINCIPAIS[
                "coolant_temp_c"
            ],
        )

    telemetria_linha2_col1, telemetria_linha2_col2 = (
        st.columns(2)
    )

    with telemetria_linha2_col1:

        grafico_sensor(
            dados_processados,
            "speed_kmh",
            SENSORES_PRINCIPAIS[
                "speed_kmh"
            ],
        )

    with telemetria_linha2_col2:

        grafico_sensor(
            dados_processados,
            "engine_load_pct",
            SENSORES_PRINCIPAIS[
                "engine_load_pct"
            ],
        )

    with st.expander(
        "Ver sensores adicionais"
    ):

        adicionais_linha1_col1, adicionais_linha1_col2 = (
            st.columns(2)
        )

        with adicionais_linha1_col1:

            grafico_sensor(
                dados_processados,
                "throttle_pct",
                SENSORES_ADICIONAIS[
                    "throttle_pct"
                ],
            )

        with adicionais_linha1_col2:

            grafico_sensor(
                dados_processados,
                "intake_temp_c",
                SENSORES_ADICIONAIS[
                    "intake_temp_c"
                ],
            )

        adicionais_linha2_col1, adicionais_linha2_col2 = (
            st.columns(2)
        )

        with adicionais_linha2_col1:

            grafico_sensor(
                dados_processados,
                "map_kpa",
                SENSORES_ADICIONAIS[
                    "map_kpa"
                ],
            )

        with adicionais_linha2_col2:

            grafico_sensor(
                dados_processados,
                "battery_voltage",
                SENSORES_ADICIONAIS[
                    "battery_voltage"
                ],
            )

    # ========================================================
    # 5. COMPORTAMENTO DA IA
    # ========================================================

    st.divider()

    st.subheader(
        "5. Comportamento da IA"
    )

    historico = resultado[
        "historico"
    ].copy()

    grafico_reconstruction_error(
        historico,
        float(
            resultado[
                "threshold"
            ]
        ),
    )

    grafico_anomaly_score(
        historico
    )

    st.caption(
        "O Reconstruction Error mede a diferença "
        "entre a telemetria observada e a reconstrução "
        "produzida pelo Autoencoder. Valores acima do "
        "threshold são classificados como anômalos "
        "em relação ao padrão aprendido."
    )

    # ========================================================
    # 6. DETALHES
    # ========================================================

    st.divider()

    st.subheader(
        "6. Detalhes da análise"
    )

    detalhes_col1, detalhes_col2, detalhes_col3, detalhes_col4 = (
        st.columns(4)
    )

    with detalhes_col1:

        st.metric(
            "Amostras",
            resultado[
                "amostras"
            ],
        )

    with detalhes_col2:

        st.metric(
            "Janelas",
            resultado[
                "janelas"
            ],
        )

    with detalhes_col3:

        st.metric(
            "Janela temporal",
            resultado[
                "window_size"
            ],
        )

    with detalhes_col4:

        st.metric(
            "Threshold",
            f"{resultado['threshold']:.6f}",
        )

    detalhes2_col1, detalhes2_col2, detalhes2_col3 = (
        st.columns(3)
    )

    with detalhes2_col1:

        st.metric(
            "Erro atual",
            f"{resultado['reconstruction_error_atual']:.6f}",
        )

    with detalhes2_col2:

        st.metric(
            "Anomalias totais",
            f"{resultado['taxa_anomalias_total_pct']:.2f}%",
        )

    with detalhes2_col3:

        st.metric(
            "Anomalias recentes",
            f"{resultado['taxa_anomalias_recente_pct']:.2f}%",
        )

    with st.expander(
        "Features utilizadas pela IA"
    ):

        for feature in resultado[
            "features_utilizadas"
        ]:

            st.write(
                f"• {feature}"
            )

    st.warning(
        "Modelo automotivo demonstrativo: o Autoencoder "
        "foi desenvolvido a partir de telemetria simulada. "
        "Resultados sobre arquivos CSV externos indicam "
        "desvio em relação a essa referência demonstrativa "
        "e não constituem diagnóstico mecânico validado."
    )


# ============================================================
# CABEÇALHO
# ============================================================

st.title(
    "🚗 Monitoramento Automotivo"
)

st.caption(
    "Análise de telemetria e detecção de desvios "
    "multivariados por Inteligência Artificial."
)

st.info(
    "O módulo automotivo identifica desvios em relação "
    "ao padrão de telemetria aprendido pelo modelo. "
    "Ele não estima RUL automotivo nem probabilidade "
    "de falha mecânica."
)


# ============================================================
# CARREGAR PIPELINE
# ============================================================

try:

    pipeline = carregar_pipeline()

except Exception as erro:

    st.error(
        "Não foi possível carregar o modelo automotivo."
    )

    st.exception(
        erro
    )

    st.stop()


# ============================================================
# 1. FONTE DOS DADOS
# ============================================================

st.divider()

st.subheader(
    "1. Fonte da telemetria"
)

fonte = st.radio(
    "Selecione a origem dos dados:",
    options=[
        "Simulação",
        "Importar CSV",
    ],
    horizontal=True,
)


# ============================================================
# LIMPAR RESULTADO AO TROCAR DE FONTE
# ============================================================

if (
    st.session_state.fonte_anterior_automotivo
    is not None
    and
    st.session_state.fonte_anterior_automotivo
    != fonte
):

    st.session_state.resultado_automotivo = None
    st.session_state.descricao_origem_automotivo = None
    st.session_state.lote_csv_automotivo = None


st.session_state.fonte_anterior_automotivo = fonte


# ============================================================
# SIMULAÇÃO
# ============================================================

if fonte == "Simulação":

    entrada_col1, entrada_col2, entrada_col3 = (
        st.columns(3)
    )

    with entrada_col1:

        vehicle_id = st.text_input(
            "Identificação do veículo",
            value="VEICULO_DEMO_01",
        )

    with entrada_col2:

        cenario = st.selectbox(
            "Cenário",
            options=list(
                CENARIOS_DISPONIVEIS
            ),
            format_func=lambda x: (
                CENARIO_LABELS.get(
                    x,
                    x,
                )
            ),
        )

    with entrada_col3:

        duracao_minutos = st.slider(
            "Duração da sessão (min)",
            min_value=5,
            max_value=30,
            value=20,
            step=1,
        )

    with st.expander(
        "Configurações da simulação"
    ):

        seed = st.number_input(
            "Seed",
            min_value=0,
            max_value=999999,
            value=42,
            step=1,
        )

        st.caption(
            "Frequência atual: 1 amostra por segundo."
        )

    analisar_simulacao = st.button(
        "Analisar telemetria simulada",
        type="primary",
        use_container_width=True,
    )

    if analisar_simulacao:

        try:

            config = ConfigSimulacao(
                vehicle_id=vehicle_id,
                duracao_minutos=duracao_minutos,
                intervalo_segundos=1,
                cenario=cenario,
                seed=int(seed),
            )

            with st.spinner(
                "Gerando e analisando telemetria..."
            ):

                dados_simulados = gerar_telemetria(
                    config
                )

                resultado_simulacao = pipeline.analisar(
                    dados_simulados,
                    origem_dados="SIMULADO",
                )

            if (
                resultado_simulacao.get(
                    "status_pipeline"
                )
                == "OK"
            ):

                st.session_state.resultado_automotivo = (
                    resultado_simulacao
                )

                st.session_state.descricao_origem_automotivo = (
                    "Telemetria gerada pelo "
                    "simulador demonstrativo."
                )

                st.session_state.lote_csv_automotivo = None

            else:

                st.warning(
                    "O pipeline não conseguiu "
                    "concluir a análise."
                )

        except Exception as erro:

            st.error(
                "Não foi possível gerar/analisar "
                "a simulação."
            )

            st.exception(
                erro
            )


# ============================================================
# IMPORTAÇÃO DE CSV
# ============================================================

else:

    st.write(
        "Cada arquivo é tratado como uma "
        "**sessão independente de telemetria**."
    )

    st.caption(
        "Os arquivos não são concatenados antes da IA. "
        "Isso evita criar janelas artificiais entre "
        "viagens ou períodos diferentes."
    )

    arquivos = st.file_uploader(
        "Selecione um ou mais arquivos CSV",
        type=["csv"],
        accept_multiple_files=True,
        help=(
            "Você pode selecionar vários arquivos "
            "do mesmo veículo ou de veículos diferentes."
        ),
    )

    if arquivos:

        upload_col1, upload_col2 = st.columns(
            2
        )

        with upload_col1:

            st.metric(
                "Arquivos selecionados",
                len(arquivos),
            )

        with upload_col2:

            tamanho_total_mb = sum(
                arquivo.size
                for arquivo in arquivos
            ) / (
                1024 * 1024
            )

            st.metric(
                "Tamanho total",
                f"{tamanho_total_mb:.2f} MB",
            )

        analisar_lote = st.button(
            (
                "Analisar arquivo"
                if len(arquivos) == 1
                else "Analisar sessões"
            ),
            type="primary",
            use_container_width=True,
        )

        if analisar_lote:

            try:

                with st.spinner(
                    "Validando e analisando sessões..."
                ):

                    resultado_lote = (
                        processar_sessoes_csv(
                            arquivos,
                            pipeline,
                        )
                    )

                st.session_state.lote_csv_automotivo = (
                    resultado_lote
                )

                st.session_state.resultado_automotivo = None

                st.session_state.descricao_origem_automotivo = None

            except Exception as erro:

                st.error(
                    "Ocorreu um erro durante "
                    "o processamento dos arquivos."
                )

                st.exception(
                    erro
                )


# ============================================================
# RESULTADO DE LOTE CSV
# ============================================================

if (
    fonte == "Importar CSV"
    and
    st.session_state.lote_csv_automotivo
    is not None
):

    lote = (
        st.session_state.lote_csv_automotivo
    )

    st.divider()

    st.subheader(
        "Resumo das sessões importadas"
    )

    resumo_col1, resumo_col2, resumo_col3 = (
        st.columns(3)
    )

    with resumo_col1:

        st.metric(
            "Arquivos recebidos",
            lote[
                "total_arquivos"
            ],
        )

    with resumo_col2:

        st.metric(
            "Sessões analisadas",
            lote[
                "total_validos"
            ],
        )

    with resumo_col3:

        st.metric(
            "Sessões rejeitadas",
            lote[
                "total_rejeitados"
            ],
        )

    # ========================================================
    # SESSÕES REJEITADAS
    # ========================================================

    if lote[
        "sessoes_rejeitadas"
    ]:

        with st.expander(
            (
                f"⚠️ Ver {lote['total_rejeitados']} "
                "sessão(ões) rejeitada(s)"
            )
        ):

            for rejeitada in lote[
                "sessoes_rejeitadas"
            ]:

                st.warning(
                    (
                        f"**{rejeitada['arquivo']}**\n\n"
                        f"{rejeitada['motivo']}"
                    )
                )

    # ========================================================
    # SESSÕES VÁLIDAS
    # ========================================================

    if lote[
        "total_validos"
    ] > 0:

        resumo = lote[
            "resumo"
        ].copy()

        resumo_exibicao = resumo.rename(
            columns={
                "arquivo": "Arquivo",
                "vehicle_id": "Veículo",
                "amostras": "Amostras",
                "janelas": "Janelas",
                "automotive_health": "Health",
                "status": "Status",
                "anomaly_score": "Anomaly Score",
                "persistencia_recente_pct": "Persistência recente (%)",
            }
        )

        resumo_exibicao[
            "Status"
        ] = resumo_exibicao[
            "Status"
        ].map(
            lambda x: formatar_status(
                x
            )
        )

        st.dataframe(
            resumo_exibicao,
            use_container_width=True,
            hide_index=True,
        )

        # ====================================================
        # VERIFICAR VEÍCULOS
        # ====================================================

        ids_validos = [
            sessao[
                "vehicle_id"
            ]
            for sessao in lote[
                "sessoes_validas"
            ]
            if (
                sessao[
                    "vehicle_id"
                ]
                not in (
                    None,
                    "MULTIPLOS_IDS",
                )
            )
        ]

        ids_unicos = sorted(
            set(
                ids_validos
            )
        )

        if len(
            ids_unicos
        ) == 1:

            st.success(
                (
                    "As sessões identificadas pertencem "
                    f"ao veículo: **{ids_unicos[0]}**."
                )
            )

        elif len(
            ids_unicos
        ) > 1:

            st.info(
                (
                    "Foram identificados múltiplos veículos "
                    "no lote: "
                    + ", ".join(
                        ids_unicos
                    )
                    + ". As sessões continuam sendo "
                    "analisadas individualmente."
                )
            )

        # ====================================================
        # SELETOR DE SESSÃO
        # ====================================================

        st.markdown(
            "### Sessão para visualizar"
        )

        nomes_sessoes = [
            sessao[
                "arquivo"
            ]
            for sessao in lote[
                "sessoes_validas"
            ]
        ]

        sessao_escolhida = st.selectbox(
            "Selecione uma sessão analisada:",
            options=nomes_sessoes,
        )

        sessao_selecionada = next(
            sessao
            for sessao in lote[
                "sessoes_validas"
            ]
            if sessao[
                "arquivo"
            ]
            == sessao_escolhida
        )

        resultado_selecionado = (
            sessao_selecionada[
                "resultado"
            ]
        )

        vehicle_id_selecionado = (
            sessao_selecionada[
                "vehicle_id"
            ]
        )

        st.write(
            f"**Arquivo:** {sessao_escolhida}"
        )

        if vehicle_id_selecionado:

            st.write(
                (
                    "**Veículo:** "
                    f"{vehicle_id_selecionado}"
                )
            )

        # ====================================================
        # DASHBOARD DA SESSÃO ESCOLHIDA
        # ====================================================

        mostrar_dashboard_resultado(
            resultado_selecionado,
            (
                "Sessão importada por CSV. "
                f"Arquivo: {sessao_escolhida}"
            ),
        )

    else:

        st.error(
            "Nenhum dos arquivos enviados pôde "
            "ser analisado pela IA."
        )


# ============================================================
# RESULTADO DA SIMULAÇÃO
# ============================================================

if (
    fonte == "Simulação"
    and
    st.session_state.resultado_automotivo
    is not None
):

    mostrar_dashboard_resultado(
        st.session_state.resultado_automotivo,
        st.session_state.descricao_origem_automotivo,
    )