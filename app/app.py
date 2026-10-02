from pathlib import Path

import streamlit as st


# ============================================================
# CAMINHOS
# ============================================================

APP_DIR = Path(__file__).resolve().parent


# ============================================================
# CONFIGURAÇÃO GLOBAL
# ============================================================

st.set_page_config(
    page_title="UPX 2.0 | Manutenção Preditiva",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PÁGINAS
# ============================================================

pagina_nasa = st.Page(
    APP_DIR / "paginas" / "01_nasa_cmapss.py",
    title="NASA C-MAPSS",
    icon="✈️",
    default=True,
)

pagina_automotivo = st.Page(
    APP_DIR / "paginas" / "02_automotivo.py",
    title="Automotivo",
    icon="🚗",
)


# ============================================================
# NAVEGAÇÃO DA PLATAFORMA
# ============================================================

navegacao = st.navigation(
    {
        "Módulos": [
            pagina_nasa,
            pagina_automotivo,
        ]
    }
)


# ============================================================
# SIDEBAR GLOBAL
# ============================================================

with st.sidebar:
    st.markdown("---")

    st.caption(
        "UPX 2.0 • Plataforma modular de manutenção preditiva"
    )


# ============================================================
# EXECUÇÃO
# ============================================================

navegacao.run()