from __future__ import annotations
import json
import os
from pathlib import Path
# Compatibilidade com o ambiente atual do projeto.
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
# ============================================================
# CAMINHOS
# ============================================================
PASTA_MODELO_PADRAO = (
    Path("modelo")
    / "automotivo_anomalia"
)
# ============================================================
# PIPELINE
# ============================================================
class PipelineAutomotivo:
    """
    Pipeline de inferência do módulo automotivo.
    Suporta:
    - telemetria simulada;
    - upload de CSV;
    - Autoencoder demonstrativo atual;
    - futura substituição por baseline real.
    IMPORTANTE:
    O modelo atual foi treinado exclusivamente com telemetria
    simulada NORMAL. Em dados reais, o resultado deve ser
    interpretado como análise experimental de desvio.
    """
    def __init__(
        self,
        pasta_modelo: str | Path = PASTA_MODELO_PADRAO,
    ):
        self.pasta_modelo = Path(
            pasta_modelo
        )
        self._carregar_config()
        self._carregar_scaler()
        self._carregar_modelo()
    # ========================================================
    # CARREGAMENTO
    # ========================================================
    def _carregar_config(self):
        caminho = (
            self.pasta_modelo
            / "config.json"
        )
        if not caminho.exists():
            raise FileNotFoundError(
                f"Config não encontrado: {caminho}"
            )
        with open(
            caminho,
            "r",
            encoding="utf-8",
        ) as arquivo:
            self.config = json.load(
                arquivo
            )
        self.features = self.config[
            "features"
        ]
        self.window_size = int(
            self.config[
                "window_size"
            ]
        )
        self.threshold = float(
            self.config[
                "threshold"
            ]
        )
        self.mediana_normal = float(
            self.config[
                "median_normal_validation"
            ]
        )
        self.p99_normal = float(
            self.config[
                "p99_normal_validation"
            ]
        )
    def _carregar_scaler(self):
        caminho = (
            self.pasta_modelo
            / "scaler.pkl"
        )
        if not caminho.exists():
            raise FileNotFoundError(
                f"Scaler não encontrado: {caminho}"
            )
        self.scaler = joblib.load(
            caminho
        )
    def _carregar_modelo(self):
        caminho = (
            self.pasta_modelo
            / "modelo_autoencoder.keras"
        )
        if not caminho.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado: {caminho}"
            )
        self.modelo = (
            tf.keras.models.load_model(
                caminho,
                compile=False,
            )
        )
    # ========================================================
    # VALIDAÇÃO
    # ========================================================
    def validar_dados(
        self,
        df: pd.DataFrame,
    ) -> dict:
        if not isinstance(
            df,
            pd.DataFrame,
        ):
            raise TypeError(
                "Os dados devem ser um DataFrame."
            )
        if df.empty:
            return {
                "valido": False,
                "motivo": (
                    "O conjunto de dados está vazio."
                ),
                "colunas_ausentes": (
                    self.features.copy()
                ),
            }
        ausentes = [
            feature
            for feature in self.features
            if feature not in df.columns
        ]
        if ausentes:
            return {
                "valido": False,
                "motivo": (
                    "Faltam variáveis necessárias "
                    "para a IA."
                ),
                "colunas_ausentes": ausentes,
            }
        nao_numericas = []
        for feature in self.features:
            convertido = pd.to_numeric(
                df[feature],
                errors="coerce",
            )
            # Valor original preenchido que virou NaN.
            problema = (
                df[feature].notna()
                & convertido.isna()
            )
            if problema.any():
                nao_numericas.append(
                    feature
                )
        if nao_numericas:
            return {
                "valido": False,
                "motivo": (
                    "Existem valores não numéricos "
                    "nas variáveis da IA."
                ),
                "colunas_problematicas": (
                    nao_numericas
                ),
            }
        n_ausentes = int(
            df[self.features]
            .isna()
            .sum()
            .sum()
        )
        if n_ausentes > 0:
            return {
                "valido": False,
                "motivo": (
                    "Existem valores ausentes nas "
                    "variáveis necessárias para a IA."
                ),
                "valores_ausentes": n_ausentes,
            }
        if len(df) < self.window_size:
            return {
                "valido": False,
                "motivo": (
                    f"São necessárias pelo menos "
                    f"{self.window_size} amostras."
                ),
                "amostras_recebidas": len(df),
            }
        return {
            "valido": True,
            "motivo": "Dados compatíveis.",
            "colunas_ausentes": [],
            "amostras": len(df),
        }
    # ========================================================
    # PADRONIZAÇÃO
    # ========================================================
    def preparar_dados(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        dados = df.copy()
        for feature in self.features:
            dados[feature] = (
                pd.to_numeric(
                    dados[feature],
                    errors="coerce",
                )
            )
        if "timestamp" in dados.columns:
            dados["timestamp"] = (
                pd.to_datetime(
                    dados["timestamp"],
                    errors="coerce",
                )
            )
            # Se timestamps válidos existirem,
            # preservamos a ordem temporal.
            if (
                dados["timestamp"]
                .notna()
                .all()
            ):
                dados = (
                    dados
                    .sort_values(
                        "timestamp"
                    )
                    .reset_index(
                        drop=True
                    )
                )
            else:
                dados = dados.reset_index(
                    drop=True
                )
        else:
            dados = dados.reset_index(
                drop=True
            )
        return dados
    # ========================================================
    # JANELAS
    # ========================================================
    def _criar_janelas(
        self,
        dados: pd.DataFrame,
    ):
        valores = self.scaler.transform(
            dados[self.features]
        )
        janelas = []
        indices_finais = []
        for inicio in range(
            0,
            len(dados)
            - self.window_size
            + 1,
        ):
            fim = (
                inicio
                + self.window_size
            )
            janelas.append(
                valores[inicio:fim]
            )
            indices_finais.append(
                fim - 1
            )
        X = np.asarray(
            janelas,
            dtype=np.float32,
        )
        return (
            X,
            np.asarray(
                indices_finais
            ),
        )
    # ========================================================
    # RECONSTRUCTION ERROR
    # ========================================================
    def _calcular_erros(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        reconstruido = (
            self.modelo.predict(
                X,
                batch_size=256,
                verbose=0,
            )
        )
        erros = np.mean(
            np.square(
                X - reconstruido
            ),
            axis=(1, 2),
        )
        return erros
    # ========================================================
    # ANOMALY SCORE
    # ========================================================
    # ========================================================
    # ANOMALY SCORE
    # ========================================================
    def calcular_anomaly_score(
        self,
        erros: np.ndarray,
    ) -> np.ndarray:
        """
        Anomaly Score Automotivo v1.
        Calibração:
        - mediana NORMAL -> score 0
        - threshold P95  -> score 50
        - acima do P95   -> crescimento contínuo
                            assintótico até 100
        O threshold de classificação de anomalia
        continua sendo o P95 original.
        Para erro > threshold:
            score = 50 + 50 * (1 - threshold / erro)
        O score representa intensidade de desvio
        em relação ao padrão NORMAL aprendido.
        Não representa probabilidade de falha,
        condição mecânica percentual ou RUL.
        """
        erros = np.asarray(
            erros,
            dtype=float,
        )
        score = np.zeros_like(
            erros,
            dtype=float,
        )
        mediana = float(
            self.mediana_normal
        )
        threshold = float(
            self.threshold
        )
        # ----------------------------------------------------
        # Região 1:
        # erro <= mediana
        #
        # Score = 0
        # ----------------------------------------------------
        mascara_normal = (
            erros <= mediana
        )
        score[
            mascara_normal
        ] = 0.0
        # ----------------------------------------------------
        # Região 2:
        # mediana < erro <= threshold
        #
        # Interpolação linear:
        # mediana  -> 0
        # threshold -> 50
        # ----------------------------------------------------
        mascara_intermediaria = (
            (erros > mediana)
            & (erros <= threshold)
        )
        denominador = (
            threshold
            - mediana
        )
        if denominador > 0:
            score[
                mascara_intermediaria
            ] = (
                (
                    erros[
                        mascara_intermediaria
                    ]
                    - mediana
                )
                / denominador
                * 50.0
            )
        # ----------------------------------------------------
        # Região 3:
        # erro > threshold
        #
        # Fórmula B escolhida durante calibração:
        #
        # score =
        # 50 + 50 * (1 - threshold / erro)
        #
        # Exemplos:
        #
        # threshold     -> 50
        # 2x threshold  -> 75
        # 3x threshold  -> 83.33
        # 5x threshold  -> 90
        # 10x threshold -> 95
        #
        # Aproxima-se de 100 sem saturar
        # imediatamente.
        # ----------------------------------------------------
        mascara_anomala = (
            erros > threshold
        )
        score[
            mascara_anomala
        ] = (
            50.0
            + 50.0
            * (
                1.0
                - (
                    threshold
                    / erros[
                        mascara_anomala
                    ]
                )
            )
        )
        return np.clip(
            score,
            0.0,
            100.0,
        )
    # ========================================================
    # HEALTH DEMONSTRATIVO
    # ========================================================
    @staticmethod
    def calcular_automotive_health(
        anomaly_score_medio_recente: float,
        taxa_anomalias_recente: float,
    ) -> float:
        """
        Automotive Health v1.
        Combina duas dimensões do comportamento recente:
        1. Intensidade:
           média dos Anomaly Scores das últimas
           janelas analisadas.
        2. Persistência:
           percentual recente de janelas
           classificadas como anômalas.
        Fórmula:
            risco =
                0.70 * intensidade
                +
                0.30 * persistência
            health =
                100 - risco
        O indicador representa proximidade ao
        padrão de telemetria aprendido.
        NÃO representa:
        - percentual de vida mecânica;
        - probabilidade de falha;
        - RUL;
        - diagnóstico mecânico.
        """
        intensidade = float(
            np.clip(
                anomaly_score_medio_recente,
                0.0,
                100.0,
            )
        )
        persistencia = float(
            np.clip(
                taxa_anomalias_recente,
                0.0,
                100.0,
            )
        )
        risco = (
            0.70 * intensidade
            + 0.30 * persistencia
        )
        health = (
            100.0 - risco
        )
        return float(
            np.clip(
                health,
                0.0,
                100.0,
            )
        )
    # ========================================================
    # STATUS
    # ========================================================
    @staticmethod
    def classificar_status(
        health: float,
    ) -> str:
        if health >= 80:
            return "NORMAL"
        if health >= 60:
            return "ATENCAO"
        if health >= 30:
            return "DESVIO_RELEVANTE"
        return "DESVIO_ELEVADO"
    # ========================================================
    # RECOMENDAÇÃO
    # ========================================================
    @staticmethod
    def gerar_recomendacao(
        status: str,
        origem_dados: str,
    ) -> str:
        if status == "NORMAL":
            recomendacao = (
                "Telemetria próxima ao padrão "
                "aprendido pelo modelo."
            )
        elif status == "ATENCAO":
            recomendacao = (
                "Foram observados desvios moderados. "
                "Acompanhar a evolução da telemetria."
            )
        elif status == "DESVIO_RELEVANTE":
            recomendacao = (
                "O padrão atual apresenta desvio "
                "relevante em relação à referência. "
                "Recomenda-se revisar a telemetria "
                "e investigar as variáveis envolvidas."
            )
        else:
            recomendacao = (
                "O padrão atual apresenta forte "
                "desvio em relação à referência. "
                "Recomenda-se análise da telemetria "
                "antes de qualquer conclusão mecânica."
            )
        if origem_dados.upper().startswith(
            "REAL"
        ):
            recomendacao += (
                " O modelo atual foi treinado com "
                "dados simulados; portanto, este "
                "resultado em dados reais é "
                "experimental e não constitui "
                "diagnóstico mecânico."
            )
        return recomendacao
    # ========================================================
    # INFERÊNCIA
    # ========================================================
    def analisar(
        self,
        df: pd.DataFrame,
        origem_dados: str = "SIMULADO",
    ) -> dict:
        validacao = self.validar_dados(
            df
        )
        if not validacao[
            "valido"
        ]:
            return {
                "status_pipeline": (
                    "DADOS_INCOMPATIVEIS"
                ),
                "validacao": validacao,
            }
        dados = self.preparar_dados(
            df
        )
        X, indices_finais = (
            self._criar_janelas(
                dados
            )
        )
        erros = self._calcular_erros(
            X
        )
        anomaly_scores = (
            self.calcular_anomaly_score(
                erros
            )
        )
        anomalias = (
            erros > self.threshold
        )
        # ----------------------------------------------------
        # Persistência recente
        # ----------------------------------------------------
        # Últimas 60 janelas ou todas,
        # caso a sessão seja menor.
        janela_recente = min(
            60,
            len(anomalias),
        )
        taxa_anomalias_recente = float(
            np.mean(
                anomalias[
                    -janela_recente:
                ]
            )
            * 100
        )
        erro_atual = float(
            erros[-1]
        )
        anomaly_score_atual = float(
            anomaly_scores[-1]
        )
        anomalia_atual = bool(
            anomalias[-1]
        )
        # Intensidade média recente usada no Automotive Health v1.
        anomaly_score_medio_recente = float(
            np.mean(
                anomaly_scores[-janela_recente:]
            )
        )
        health = (
            self.calcular_automotive_health(
                anomaly_score_medio_recente,
                taxa_anomalias_recente,
            )
        )
        status = (
            self.classificar_status(
                health
            )
        )
        recomendacao = (
            self.gerar_recomendacao(
                status,
                origem_dados,
            )
        )
        # ----------------------------------------------------
        # Histórico por janela
        # ----------------------------------------------------
        historico = pd.DataFrame(
            {
                "indice_final": (
                    indices_finais
                ),
                "reconstruction_error": (
                    erros
                ),
                "anomaly_score": (
                    anomaly_scores
                ),
                "anomaly": (
                    anomalias
                ),
            }
        )
        if (
            "timestamp"
            in dados.columns
        ):
            historico[
                "timestamp"
            ] = (
                dados.iloc[
                    indices_finais
                ][
                    "timestamp"
                ]
                .to_numpy()
            )
        # ----------------------------------------------------
        # Resultado
        # ----------------------------------------------------
        return {
            "status_pipeline": "OK",
            "origem_dados": (
                origem_dados
            ),
            "modelo_referencia": (
                "SIMULADO_DEMONSTRATIVO"
            ),
            "amostras": int(
                len(dados)
            ),
            "janelas": int(
                len(X)
            ),
            "window_size": (
                self.window_size
            ),
            "threshold": (
                self.threshold
            ),
            "reconstruction_error_atual": (
                erro_atual
            ),
            "anomaly_score_atual": (
                anomaly_score_atual
            ),
            "anomaly_score_medio_recente": (
                anomaly_score_medio_recente
            ),
            "anomalia_atual": (
                anomalia_atual
            ),
            "taxa_anomalias_total_pct": float(
                np.mean(
                    anomalias
                )
                * 100
            ),
            "taxa_anomalias_recente_pct": (
                taxa_anomalias_recente
            ),
            "automotive_health": (
                health
            ),
            "status": status,
            "recomendacao": (
                recomendacao
            ),
            "features_utilizadas": (
                self.features.copy()
            ),
            "historico": historico,
            "dados_processados": dados,
        }
# ============================================================
# TESTE DIRETO
# ============================================================
if __name__ == "__main__":
    from utils.simulador_automotivo import (
        ConfigSimulacao,
        gerar_telemetria,
    )
    print()
    print("=" * 70)
    print("TESTE DO PIPELINE AUTOMOTIVO")
    print("=" * 70)
    pipeline = PipelineAutomotivo()
    cenarios = [
        "NORMAL",
        "AQUECIMENTO_PROGRESSIVO",
        "OPERACAO_IRREGULAR",
        "CONDICAO_SEVERA_SIMULADA",
    ]
    for i, cenario in enumerate(
        cenarios
    ):
        config = ConfigSimulacao(
            vehicle_id=(
                "VEICULO_PIPELINE_TESTE"
            ),
            duracao_minutos=20,
            intervalo_segundos=1,
            cenario=cenario,
            seed=20000 + i,
        )
        dados = gerar_telemetria(
            config
        )
        resultado = pipeline.analisar(
            dados,
            origem_dados="SIMULADO",
        )
        print()
        print("-" * 70)
        print(cenario)
        print("-" * 70)
        print(
            "Pipeline:",
            resultado[
                "status_pipeline"
            ],
        )
        if (
            resultado[
                "status_pipeline"
            ]
            != "OK"
        ):
            print(
                resultado[
                    "validacao"
                ]
            )
            continue
        print(
            "Janelas:",
            resultado["janelas"],
        )
        print(
            "Erro atual:",
            round(
                resultado[
                    "reconstruction_error_atual"
                ],
                6,
            ),
        )
        print(
            "Anomaly Score atual:",
            round(
                resultado[
                    "anomaly_score_atual"
                ],
                2,
            ),
        )
        print(
            "Anomaly Score médio recente:",
            round(
                resultado[
                    "anomaly_score_medio_recente"
                ],
                2,
            ),
        )
        print(
            "Anomalia atual:",
            resultado[
                "anomalia_atual"
            ],
        )
        print(
            "Taxa anomalias total:",
            round(
                resultado[
                    "taxa_anomalias_total_pct"
                ],
                2,
            ),
            "%",
        )
        print(
            "Taxa anomalias recente:",
            round(
                resultado[
                    "taxa_anomalias_recente_pct"
                ],
                2,
            ),
            "%",
        )
        print(
            "Automotive Health:",
            round(
                resultado[
                    "automotive_health"
                ],
                2,
            ),
        )
        print(
            "Status:",
            resultado[
                "status"
            ],
        )
        print(
            "Recomendação:",
            resultado[
                "recomendacao"
            ],
        )
    print()
    print("=" * 70)
    print("TESTE FINALIZADO")
    print("=" * 70)
