from __future__ import annotations

from pathlib import Path

import pandas as pd


# ============================================================
# CAMINHOS
# ============================================================

ARQUIVO_ORIGEM = Path(
    "datasets/automotivo/teste/telemetria_teste_cenarios.csv"
)

PASTA_SAIDA = Path(
    "datasets/automotivo/demo_dashboard"
)


# ============================================================
# CENÁRIOS
# ============================================================

CENARIOS = {
    "NORMAL": "01_normal.csv",
    "AQUECIMENTO_PROGRESSIVO": "02_aquecimento_progressivo.csv",
    "OPERACAO_IRREGULAR": "03_operacao_irregular.csv",
    "CONDICAO_SEVERA_SIMULADA": "04_condicao_severa.csv",
}


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 76)
    print("PREPARAÇÃO — DEMO DASHBOARD AUTOMOTIVO")
    print("=" * 76)

    # --------------------------------------------------------
    # Verificar dataset
    # --------------------------------------------------------

    if not ARQUIVO_ORIGEM.exists():

        raise FileNotFoundError(
            f"Arquivo não encontrado: {ARQUIVO_ORIGEM}"
        )

    # --------------------------------------------------------
    # Carregar lote
    # --------------------------------------------------------

    print()
    print(
        f"Carregando: {ARQUIVO_ORIGEM}"
    )

    df = pd.read_csv(
        ARQUIVO_ORIGEM
    )

    print(
        f"Linhas carregadas: {len(df)}"
    )

    # --------------------------------------------------------
    # Verificar colunas necessárias
    # --------------------------------------------------------

    colunas_necessarias = [
        "scenario",
        "session_id",
    ]

    faltantes = [
        coluna
        for coluna in colunas_necessarias
        if coluna not in df.columns
    ]

    if faltantes:

        raise ValueError(
            "Colunas necessárias ausentes: "
            + ", ".join(faltantes)
        )

    # --------------------------------------------------------
    # Criar pasta de saída
    # --------------------------------------------------------

    PASTA_SAIDA.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        f"Pasta de saída: {PASTA_SAIDA}"
    )

    # --------------------------------------------------------
    # Extrair uma sessão de cada cenário
    # --------------------------------------------------------

    resultados = []

    for cenario, nome_arquivo in CENARIOS.items():

        dados_cenario = df[
            df["scenario"] == cenario
        ].copy()

        if dados_cenario.empty:

            print()
            print(
                f"[ERRO] Cenário não encontrado: {cenario}"
            )

            continue

        # Primeira sessão existente desse cenário.
        session_id = (
            dados_cenario[
                "session_id"
            ]
            .dropna()
            .iloc[0]
        )

        sessao = dados_cenario[
            dados_cenario[
                "session_id"
            ]
            == session_id
        ].copy()

        # ----------------------------------------------------
        # Ordenação temporal
        # ----------------------------------------------------

        if "timestamp" in sessao.columns:

            sessao["timestamp"] = pd.to_datetime(
                sessao["timestamp"],
                errors="coerce",
            )

            sessao = sessao.sort_values(
                "timestamp"
            )

        sessao = sessao.reset_index(
            drop=True
        )

        # ----------------------------------------------------
        # Salvar
        # ----------------------------------------------------

        caminho_saida = (
            PASTA_SAIDA
            / nome_arquivo
        )

        sessao.to_csv(
            caminho_saida,
            index=False,
        )

        vehicle_id = (
            sessao["vehicle_id"].iloc[0]
            if "vehicle_id" in sessao.columns
            else "NÃO INFORMADO"
        )

        resultados.append(
            {
                "cenario": cenario,
                "session_id": session_id,
                "vehicle_id": vehicle_id,
                "linhas": len(sessao),
                "arquivo": str(
                    caminho_saida
                ),
            }
        )

        print()
        print(
            f"[OK] {cenario}"
        )

        print(
            f"     Sessão: {session_id}"
        )

        print(
            f"     Veículo: {vehicle_id}"
        )

        print(
            f"     Linhas: {len(sessao)}"
        )

        print(
            f"     Arquivo: {caminho_saida}"
        )

    # --------------------------------------------------------
    # Resumo
    # --------------------------------------------------------

    print()
    print("-" * 76)
    print("RESUMO")
    print("-" * 76)

    resumo = pd.DataFrame(
        resultados
    )

    if resumo.empty:

        print(
            "Nenhuma sessão foi criada."
        )

    else:

        print(
            resumo.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Verificação final
    # --------------------------------------------------------

    print()
    print("=" * 76)

    if len(resultados) == 4:

        print(
            "RESULTADO FINAL: 4 SESSÕES CRIADAS COM SUCESSO"
        )

    else:

        print(
            "RESULTADO FINAL: NEM TODOS OS CENÁRIOS "
            "FORAM ENCONTRADOS"
        )

    print("=" * 76)


if __name__ == "__main__":
    main()