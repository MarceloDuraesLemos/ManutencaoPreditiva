from pathlib import Path
import json
import pandas as pd


# =========================================================
# CAMINHOS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

LINEAR_METRICAS_PATH = (
    BASE_DIR
    / "modelo"
    / "fd001"
    / "metricas.json"
)

CAPPED_METRICAS_PATH = (
    BASE_DIR
    / "modelo"
    / "fd001_capped125"
    / "metricas.json"
)

RESULTADOS_DIR = (
    BASE_DIR
    / "resultados"
)

ARQUIVO_SAIDA = (
    RESULTADOS_DIR
    / "comparacao_modelos_fd001.csv"
)


# =========================================================
# CARREGAR JSON
# =========================================================

def carregar_json(caminho):

    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado:\n{caminho}"
        )

    with open(
        caminho,
        "r",
        encoding="utf-8"
    ) as arquivo:

        return json.load(arquivo)


# =========================================================
# EXECUÇÃO
# =========================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("COMPARAÇÃO DOS MODELOS FD001")
    print("=" * 60)

    # -----------------------------------------------------
    # 1. CARREGAR MÉTRICAS
    # -----------------------------------------------------

    metricas_linear = carregar_json(
        LINEAR_METRICAS_PATH
    )

    metricas_capped = carregar_json(
        CAPPED_METRICAS_PATH
    )

    # -----------------------------------------------------
    # 2. MÉTRICAS DO MODELO LINEAR
    # -----------------------------------------------------

    mae_validacao_linear = float(
        metricas_linear[
            "mae_validacao"
        ]
    )

    rmse_validacao_linear = float(
        metricas_linear[
            "rmse_validacao"
        ]
    )

    mae_teste_linear = float(
        metricas_linear[
            "mae_teste"
        ]
    )

    rmse_teste_linear = float(
        metricas_linear[
            "rmse_teste"
        ]
    )

    # -----------------------------------------------------
    # 3. MÉTRICAS DO MODELO CAPPED
    # -----------------------------------------------------

    mae_validacao_capped = float(
        metricas_capped[
            "mae_validacao"
        ]
    )

    rmse_validacao_capped = float(
        metricas_capped[
            "rmse_validacao"
        ]
    )

    mae_teste_capped = float(
        metricas_capped[
            "mae_teste"
        ]
    )

    rmse_teste_capped = float(
        metricas_capped[
            "rmse_teste"
        ]
    )

    # -----------------------------------------------------
    # 4. MELHORIA NO TESTE OFICIAL
    # -----------------------------------------------------

    melhoria_mae_teste = (
        (
            mae_teste_linear
            - mae_teste_capped
        )
        / mae_teste_linear
    ) * 100

    melhoria_rmse_teste = (
        (
            rmse_teste_linear
            - rmse_teste_capped
        )
        / rmse_teste_linear
    ) * 100

    # -----------------------------------------------------
    # 5. MONTAR TABELA
    # -----------------------------------------------------

    dados = [
        {
            "modelo": "LSTM Linear",
            "rul_target": "linear",
            "rul_cap": None,
            "mae_validacao": mae_validacao_linear,
            "rmse_validacao": rmse_validacao_linear,
            "mae_teste": mae_teste_linear,
            "rmse_teste": rmse_teste_linear
        },
        {
            "modelo": "LSTM Capped 125",
            "rul_target": "capped",
            "rul_cap": 125,
            "mae_validacao": mae_validacao_capped,
            "rmse_validacao": rmse_validacao_capped,
            "mae_teste": mae_teste_capped,
            "rmse_teste": rmse_teste_capped
        }
    ]

    df_comparacao = pd.DataFrame(
        dados
    )

    # -----------------------------------------------------
    # 6. MELHOR MODELO NO TESTE
    # -----------------------------------------------------

    melhor_mae = (
        df_comparacao
        .sort_values(
            "mae_teste"
        )
        .iloc[0]
    )

    melhor_rmse = (
        df_comparacao
        .sort_values(
            "rmse_teste"
        )
        .iloc[0]
    )

    # -----------------------------------------------------
    # 7. EXIBIR TABELA
    # -----------------------------------------------------

    print("\nTabela de comparação:\n")

    print(
        df_comparacao
        .round(2)
        .to_string(
            index=False
        )
    )

    print("\n" + "-" * 60)

    print(
        f"\nMelhoria do Capped 125 no MAE de teste: "
        f"{melhoria_mae_teste:.2f}%"
    )

    print(
        f"Melhoria do Capped 125 no RMSE de teste: "
        f"{melhoria_rmse_teste:.2f}%"
    )

    print("\n" + "-" * 60)

    print(
        f"\nMelhor modelo por MAE de teste: "
        f"{melhor_mae['modelo']}"
    )

    print(
        f"MAE: "
        f"{melhor_mae['mae_teste']:.2f} ciclos"
    )

    print(
        f"\nMelhor modelo por RMSE de teste: "
        f"{melhor_rmse['modelo']}"
    )

    print(
        f"RMSE: "
        f"{melhor_rmse['rmse_teste']:.2f} ciclos"
    )

    # -----------------------------------------------------
    # 8. SALVAR CSV
    # -----------------------------------------------------

    RESULTADOS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df_comparacao.to_csv(
        ARQUIVO_SAIDA,
        index=False
    )

    print("\n" + "=" * 60)
    print("COMPARAÇÃO SALVA")
    print("=" * 60)

    print(
        f"\nArquivo: "
        f"{ARQUIVO_SAIDA}"
    )