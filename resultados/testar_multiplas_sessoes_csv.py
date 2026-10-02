from __future__ import annotations

import os
from pathlib import Path


# ============================================================
# CONFIGURAÇÃO TENSORFLOW
# ============================================================

os.environ["TF_USE_LEGACY_KERAS"] = "1"


# ============================================================
# IMPORTS
# ============================================================

from utils.pipeline_automotivo import (
    PipelineAutomotivo,
)

from utils.processador_sessoes_automotivas import (
    processar_sessoes_csv,
)


# ============================================================
# CAMINHOS
# ============================================================

ARQUIVO_VALIDO = Path(
    "datasets/automotivo/simulados/telemetria_normal.csv"
)

ARQUIVO_INVALIDO = Path(
    "resultados/automotivo/testes_csv/teste_sem_rpm.csv"
)


# ============================================================
# WRAPPER PARA SIMULAR UPLOAD DO STREAMLIT
# ============================================================

class ArquivoTeste:
    """
    Simula o comportamento básico de um UploadedFile
    do Streamlit.

    Permite testar o processador sem precisar abrir
    o dashboard.
    """

    def __init__(
        self,
        caminho: Path,
        nome: str,
    ):
        self.caminho = caminho
        self.name = nome
        self._arquivo = open(
            caminho,
            "rb",
        )

    def read(self, *args, **kwargs):
        return self._arquivo.read(
            *args,
            **kwargs,
        )

    def seek(self, *args, **kwargs):
        return self._arquivo.seek(
            *args,
            **kwargs,
        )

    def tell(self):
        return self._arquivo.tell()

    def close(self):
        self._arquivo.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 76)
    print("TESTE — MÚLTIPLAS SESSÕES AUTOMOTIVAS")
    print("=" * 76)

    # --------------------------------------------------------
    # Verificar arquivos
    # --------------------------------------------------------

    if not ARQUIVO_VALIDO.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {ARQUIVO_VALIDO}"
        )

    if not ARQUIVO_INVALIDO.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {ARQUIVO_INVALIDO}"
        )

    # --------------------------------------------------------
    # Criar arquivos simulando uploads separados
    # --------------------------------------------------------

    arquivos = [
        ArquivoTeste(
            ARQUIVO_VALIDO,
            "sessao_normal_01.csv",
        ),
        ArquivoTeste(
            ARQUIVO_VALIDO,
            "sessao_normal_02.csv",
        ),
        ArquivoTeste(
            ARQUIVO_INVALIDO,
            "sessao_sem_rpm.csv",
        ),
    ]

    try:

        print()
        print(
            "Carregando PipelineAutomotivo..."
        )

        pipeline = PipelineAutomotivo()

        print(
            "Pipeline carregado."
        )

        print()
        print(
            "Processando 3 arquivos como "
            "3 sessões independentes..."
        )

        resultado = processar_sessoes_csv(
            arquivos,
            pipeline,
        )

        # ====================================================
        # TOTAIS
        # ====================================================

        print()
        print("-" * 76)
        print("RESUMO")
        print("-" * 76)

        print(
            "Arquivos recebidos:",
            resultado[
                "total_arquivos"
            ],
        )

        print(
            "Sessões válidas:",
            resultado[
                "total_validos"
            ],
        )

        print(
            "Sessões rejeitadas:",
            resultado[
                "total_rejeitados"
            ],
        )

        # ====================================================
        # TABELA DAS SESSÕES VÁLIDAS
        # ====================================================

        print()
        print("-" * 76)
        print("SESSÕES ANALISADAS")
        print("-" * 76)

        resumo = resultado[
            "resumo"
        ]

        if resumo.empty:

            print(
                "Nenhuma sessão válida."
            )

        else:

            print(
                resumo.to_string(
                    index=False
                )
            )

        # ====================================================
        # REJEITADOS
        # ====================================================

        print()
        print("-" * 76)
        print("SESSÕES REJEITADAS")
        print("-" * 76)

        rejeitadas = resultado[
            "sessoes_rejeitadas"
        ]

        if not rejeitadas:

            print(
                "Nenhuma sessão rejeitada."
            )

        else:

            for sessao in rejeitadas:

                print()
                print(
                    "Arquivo:",
                    sessao[
                        "arquivo"
                    ],
                )

                print(
                    "Motivo:",
                    sessao[
                        "motivo"
                    ],
                )

        # ====================================================
        # VERIFICAÇÕES AUTOMÁTICAS
        # ====================================================

        print()
        print("-" * 76)
        print("VERIFICAÇÕES")
        print("-" * 76)

        teste_total = (
            resultado[
                "total_arquivos"
            ]
            == 3
        )

        teste_validos = (
            resultado[
                "total_validos"
            ]
            == 2
        )

        teste_rejeitados = (
            resultado[
                "total_rejeitados"
            ]
            == 1
        )

        teste_nomes = (
            not resumo.empty
            and resumo[
                "arquivo"
            ].tolist()
            == [
                "sessao_normal_01.csv",
                "sessao_normal_02.csv",
            ]
        )

        print(
            "Total = 3:",
            "OK"
            if teste_total
            else "FALHOU",
        )

        print(
            "Válidos = 2:",
            "OK"
            if teste_validos
            else "FALHOU",
        )

        print(
            "Rejeitados = 1:",
            "OK"
            if teste_rejeitados
            else "FALHOU",
        )

        print(
            "Sessões mantidas separadas:",
            "OK"
            if teste_nomes
            else "FALHOU",
        )

        todos_ok = all(
            [
                teste_total,
                teste_validos,
                teste_rejeitados,
                teste_nomes,
            ]
        )

        print()
        print("=" * 76)

        if todos_ok:

            print(
                "RESULTADO FINAL: TODOS OS TESTES PASSARAM"
            )

        else:

            print(
                "RESULTADO FINAL: EXISTEM TESTES COM FALHA"
            )

        print("=" * 76)

    finally:

        for arquivo in arquivos:
            arquivo.close()


if __name__ == "__main__":
    main()