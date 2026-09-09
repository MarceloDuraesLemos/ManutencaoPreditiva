from pathlib import Path
import json
import pickle
import os

# Força o TensorFlow a usar a implementação legada do Keras,
# necessária para carregar os modelos antigos do projeto.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model



class PipelineFD001:
    """
    Pipeline integrado FD001.

    Fluxo:
    sensores
    -> RUL
    -> erro de reconstrucao
    -> Anomaly Score
    -> Trend Score
    -> Health Score
    -> Status
    -> Priority Score
    -> Classe de prioridade
    """

    def __init__(self, projeto_root=None):
        if projeto_root is None:
            projeto_root = Path(__file__).resolve().parents[1]

        self.root = Path(projeto_root)

        config_path = (
            self.root
            / "modelo"
            / "fd001_final"
            / "config_pipeline.json"
        )

        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuração do pipeline não encontrada: {config_path}"
            )

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.sensores = self.config["rul"]["sensores"]
        self.window_size = int(self.config["rul"]["window_size"])
        self.rul_cap = float(self.config["rul"]["rul_cap"])

        self.trend_horizonte = int(
            self.config["trend"]["horizonte"]
        )

        self._carregar_modelos()

    # ==========================================================
    # CARREGAMENTO
    # ==========================================================

    def _carregar_pickle(self, caminho):
        caminho = self.root / caminho

        if not caminho.exists():
            raise FileNotFoundError(
                f"Arquivo não encontrado: {caminho}"
            )

        try:
            return joblib.load(caminho)

        except Exception:
            with open(caminho, "rb") as f:
                return pickle.load(f)

    def _carregar_modelos(self):
        rul_cfg = self.config["rul"]
        anom_cfg = self.config["anomalia"]

        caminho_modelo_rul = (
            self.root / rul_cfg["modelo_path"]
        )

        caminho_modelo_anomalia = (
            self.root / anom_cfg["modelo_path"]
        )

        if not caminho_modelo_rul.exists():
            raise FileNotFoundError(
                f"Modelo RUL não encontrado: {caminho_modelo_rul}"
            )

        if not caminho_modelo_anomalia.exists():
            raise FileNotFoundError(
                f"Autoencoder não encontrado: "
                f"{caminho_modelo_anomalia}"
            )

        self.modelo_rul = load_model(
            caminho_modelo_rul,
            compile=False
        )

        self.modelo_anomalia = load_model(
            caminho_modelo_anomalia,
            compile=False
        )

        self.scaler_rul = self._carregar_pickle(
            rul_cfg["scaler_path"]
        )

        self.scaler_anomalia = self._carregar_pickle(
            anom_cfg["scaler_path"]
        )

    # ==========================================================
    # VALIDAÇÃO DOS DADOS
    # ==========================================================

    def _validar_dataframe(self, df):
        obrigatorias = (
            ["unit_number", "time_in_cycles"]
            + self.sensores
        )

        faltantes = [
            coluna
            for coluna in obrigatorias
            if coluna not in df.columns
        ]

        if faltantes:
            raise ValueError(
                "Colunas obrigatórias ausentes: "
                + ", ".join(faltantes)
            )

        if df.empty:
            raise ValueError(
                "O DataFrame recebido está vazio."
            )

    # ==========================================================
    # JANELAS TEMPORAIS
    # ==========================================================

    def _criar_janelas(self, matriz):
        janelas = []

        for inicio in range(
            0,
            len(matriz) - self.window_size + 1
        ):
            fim = inicio + self.window_size

            janela = matriz[inicio:fim]

            janelas.append(janela)

        if not janelas:
            return np.empty(
                (
                    0,
                    self.window_size,
                    len(self.sensores)
                ),
                dtype=np.float32
            )

        return np.asarray(
            janelas,
            dtype=np.float32
        )

    # ==========================================================
    # RUL
    # ==========================================================

    def _prever_rul(self, dados_motor):
        sensores = dados_motor[self.sensores]

        sensores_normalizados = (
            self.scaler_rul.transform(sensores)
        )

        janelas = self._criar_janelas(
            sensores_normalizados
        )

        if len(janelas) == 0:
            return None

        predicoes = self.modelo_rul.predict(
            janelas,
            verbose=0
        )

        predicoes = np.asarray(
            predicoes
        ).reshape(-1)

        return predicoes

    def _rul_score(self, rul_estimado):
        score = (
            float(rul_estimado)
            / self.rul_cap
            * 100.0
        )

        return float(
            np.clip(score, 0.0, 100.0)
        )

    # ==========================================================
    # ANOMALIA
    # ==========================================================

    def _calcular_anomalias(self, dados_motor):
        sensores = dados_motor[self.sensores]

        sensores_normalizados = (
            self.scaler_anomalia.transform(sensores)
        )

        janelas = self._criar_janelas(
            sensores_normalizados
        )

        if len(janelas) == 0:
            return None

        reconstrucoes = (
            self.modelo_anomalia.predict(
                janelas,
                verbose=0
            )
        )

        erros = np.mean(
            np.square(
                janelas - reconstrucoes
            ),
            axis=(1, 2)
        )

        return erros

    def _anomaly_score(self, erro):
        cfg = self.config["anomaly_score"]

        ref_zero = float(
            cfg["referencia_zero"]
        )

        ref_threshold = float(
            cfg["referencia_threshold"]
        )

        ref_max = float(
            cfg["referencia_maxima"]
        )

        erro = float(erro)

        if erro <= ref_zero:
            return 0.0

        if erro <= ref_threshold:
            score = (
                50.0
                * (erro - ref_zero)
                / (ref_threshold - ref_zero)
            )

        else:
            score = (
                50.0
                + 50.0
                * (erro - ref_threshold)
                / (ref_max - ref_threshold)
            )

        return float(
            np.clip(score, 0.0, 100.0)
        )

    # ==========================================================
    # TREND
    # ==========================================================

    def _calcular_trend(
        self,
        ciclos_janelas,
        erros
    ):
        if len(erros) < self.trend_horizonte:
            return None, None

        ciclos = np.asarray(
            ciclos_janelas[
                -self.trend_horizonte:
            ],
            dtype=float
        )

        erros_recent = np.asarray(
            erros[
                -self.trend_horizonte:
            ],
            dtype=float
        )

        slope = np.polyfit(
            ciclos,
            erros_recent,
            1
        )[0]

        score = self._trend_score(
            slope
        )

        return float(slope), float(score)

    def _trend_score(self, slope):
        cfg = self.config["trend"]

        ref_zero = float(
            cfg["referencia_zero"]
        )

        ref_max = float(
            cfg["referencia_maxima"]
        )

        score = (
            (float(slope) - ref_zero)
            / (ref_max - ref_zero)
            * 100.0
        )

        return float(
            np.clip(score, 0.0, 100.0)
        )

    # ==========================================================
    # HEALTH
    # ==========================================================

    def _health_score(
        self,
        rul_score,
        anomaly_score,
        trend_score
    ):
        cfg = self.config["health_score"]

        peso_rul = float(
            cfg["peso_rul"]
        )

        peso_anomalia = float(
            cfg["peso_anomalia"]
        )

        peso_trend = float(
            cfg["peso_trend"]
        )

        saude_anomalia = (
            100.0 - anomaly_score
        )

        saude_trend = (
            100.0 - trend_score
        )

        health = (
            peso_rul * rul_score
            + peso_anomalia * saude_anomalia
            + peso_trend * saude_trend
        )

        return float(
            np.clip(health, 0.0, 100.0)
        )

    def _health_status(self, health):
        cfg = self.config["health_status"]

        if health >= float(
            cfg["saudavel_min"]
        ):
            return "SAUDAVEL"

        if health >= float(
            cfg["atencao_min"]
        ):
            return "ATENCAO"

        if health >= float(
            cfg["risco_min"]
        ):
            return "RISCO"

        return "CRITICO"

    # ==========================================================
    # PRIORIDADE
    # ==========================================================

    def _priority_score(
        self,
        health_score,
        rul_score
    ):
        cfg = self.config["priority_score"]

        peso_health = float(
            cfg["peso_health_risk"]
        )

        peso_rul = float(
            cfg["peso_rul_risk"]
        )

        health_risk = (
            100.0 - health_score
        )

        rul_risk = (
            100.0 - rul_score
        )

        prioridade = (
            peso_health * health_risk
            + peso_rul * rul_risk
        )

        return float(
            np.clip(prioridade, 0.0, 100.0)
        )

    def _priority_class(
        self,
        priority_score
    ):
        cfg = self.config[
            "priority_classes"
        ]

        if priority_score >= float(
            cfg["p1_min"]
        ):
            return "P1_IMEDIATA"

        if priority_score >= float(
            cfg["p2_min"]
        ):
            return "P2_ALTA"

        if priority_score >= float(
            cfg["p3_min"]
        ):
            return "P3_MEDIA"

        return "P4_BAIXA"

    # ==========================================================
    # INTERPRETAÇÃO
    # ==========================================================

    def _interpretar_anomalia(
        self,
        erro,
        anomaly_score
    ):
        threshold = float(
            self.config[
                "anomalia"
            ]["threshold"]
        )

        if erro > threshold:
            if anomaly_score >= 75:
                return (
                    "Desvio severo do padrão saudável "
                    "aprendido pelo Autoencoder."
                )

            return (
                "Comportamento anômalo detectado em "
                "relação ao padrão saudável."
            )

        return (
            "Comportamento próximo do padrão "
            "saudável aprendido."
        )

    # ==========================================================
    # ANÁLISE COMPLETA DE UM MOTOR
    # ==========================================================

    def analisar_motor(
        self,
        df,
        unit_number
    ):
        self._validar_dataframe(df)

        dados_motor = (
            df[
                df["unit_number"]
                == unit_number
            ]
            .sort_values(
                "time_in_cycles"
            )
            .reset_index(drop=True)
        )

        if dados_motor.empty:
            raise ValueError(
                f"Motor {unit_number} não encontrado."
            )

        quantidade_ciclos = len(
            dados_motor
        )

        if quantidade_ciclos < self.window_size:
            return {
                "unit_number": int(unit_number),
                "ciclos_disponiveis":
                    quantidade_ciclos,
                "status_pipeline":
                    "DADOS_INSUFICIENTES",
                "mensagem":
                    (
                        f"São necessários pelo menos "
                        f"{self.window_size} ciclos "
                        f"para gerar a primeira análise."
                    )
            }

        # ------------------------------------------------------
        # RUL
        # ------------------------------------------------------

        predicoes_rul = self._prever_rul(
            dados_motor
        )

        rul_pred_raw = float(
            predicoes_rul[-1]
        )

        # O modelo foi treinado com target capped em 125.
        rul_estimado = float(
            np.clip(
                rul_pred_raw,
                0.0,
                self.rul_cap
            )
        )

        rul_score = self._rul_score(
            rul_estimado
        )

        # ------------------------------------------------------
        # ANOMALIA
        # ------------------------------------------------------

        erros = self._calcular_anomalias(
            dados_motor
        )

        erro_atual = float(
            erros[-1]
        )

        anomaly_score = (
            self._anomaly_score(
                erro_atual
            )
        )

        threshold = float(
            self.config[
                "anomalia"
            ]["threshold"]
        )

        anomalia_detectada = bool(
            erro_atual > threshold
        )

        # ------------------------------------------------------
        # CICLOS CORRESPONDENTES ÀS JANELAS
        # ------------------------------------------------------

        ciclos_janelas = (
            dados_motor[
                "time_in_cycles"
            ]
            .iloc[
                self.window_size - 1:
            ]
            .values
        )

        # ------------------------------------------------------
        # TREND
        # ------------------------------------------------------

        slope, trend_score = (
            self._calcular_trend(
                ciclos_janelas,
                erros
            )
        )

        ciclo_atual = int(
            dados_motor[
                "time_in_cycles"
            ].iloc[-1]
        )

        # ------------------------------------------------------
        # CASO SEM HISTÓRICO SUFICIENTE PARA TREND
        # ------------------------------------------------------

        if trend_score is None:
            return {
                "unit_number": int(unit_number),
                "ciclo_atual": ciclo_atual,
                "ciclos_disponiveis":
                    quantidade_ciclos,

                "rul_estimado":
                    rul_estimado,

                "rul_pred_raw":
                    rul_pred_raw,

                "rul_score":
                    rul_score,

                "erro_reconstrucao":
                    erro_atual,

                "anomaly_score":
                    anomaly_score,

                "anomalia_detectada":
                    anomalia_detectada,

                "trend_slope":
                    None,

                "trend_score":
                    None,

                "health_score":
                    None,

                "health_status":
                    None,

                "priority_score":
                    None,

                "priority_class":
                    None,

                "status_pipeline":
                    "ANALISE_PARCIAL",

                "mensagem":
                    (
                        "RUL e anomalia disponíveis. "
                        "Histórico insuficiente para "
                        f"Trend Score de "
                        f"{self.trend_horizonte} janelas. "
                        "Health Score e prioridade ainda "
                        "não são calculados."
                    ),

                "interpretacao_anomalia":
                    self._interpretar_anomalia(
                        erro_atual,
                        anomaly_score
                    )
            }

        # ------------------------------------------------------
        # HEALTH
        # ------------------------------------------------------

        health_score = (
            self._health_score(
                rul_score,
                anomaly_score,
                trend_score
            )
        )

        health_status = (
            self._health_status(
                health_score
            )
        )

        # ------------------------------------------------------
        # PRIORIDADE
        # ------------------------------------------------------

        priority_score = (
            self._priority_score(
                health_score,
                rul_score
            )
        )

        priority_class = (
            self._priority_class(
                priority_score
            )
        )

        # ------------------------------------------------------
        # SAÍDA FINAL
        # ------------------------------------------------------

        resultado = {
            "unit_number":
                int(unit_number),

            "ciclo_atual":
                ciclo_atual,

            "ciclos_disponiveis":
                quantidade_ciclos,

            "rul_estimado":
                rul_estimado,

            "rul_pred_raw":
                rul_pred_raw,

            "rul_score":
                rul_score,

            "erro_reconstrucao":
                erro_atual,

            "anomaly_score":
                anomaly_score,

            "anomalia_detectada":
                anomalia_detectada,

            "trend_slope":
                slope,

            "trend_score":
                trend_score,

            "health_score":
                health_score,

            "health_status":
                health_status,

            "priority_score":
                priority_score,

            "priority_class":
                priority_class,

            "status_pipeline":
                "ANALISE_COMPLETA",

            "interpretacao_anomalia":
                self._interpretar_anomalia(
                    erro_atual,
                    anomaly_score
                )
        }

        return resultado


# ==============================================================
# FUNÇÃO AUXILIAR PARA LER C-MAPSS
# ==============================================================

def carregar_cmapss_fd001(
    caminho
):
    colunas = (
        [
            "unit_number",
            "time_in_cycles"
        ]
        + [
            f"operational_setting_{i}"
            for i in range(1, 4)
        ]
        + [
            f"sensor_{i}"
            for i in range(1, 22)
        ]
    )

    df = pd.read_csv(
        caminho,
        sep=r"\s+",
        header=None,
        names=colunas
    )

    return df