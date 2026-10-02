from pathlib import Path

import pandas as pd

from utils.simulador_automotivo import (
    ConfigSimulacao,
    gerar_telemetria,
    CENARIOS_DISPONIVEIS,
)


# ============================================================
# CONFIGURAÇÃO
# ============================================================

PASTA_SAIDA = Path("datasets") / "automotivo"

N_SESSOES_NORMAL_DESENVOLVIMENTO = 100
N_SESSOES_TESTE_POR_CENARIO = 20

SEED_BASE_DESENVOLVIMENTO = 1000
SEED_BASE_TESTE = 10000


# ============================================================
# AUXILIAR
# ============================================================

def preparar_sessao(
    session_id: str,
    vehicle_id: str,
    cenario: str,
    seed: int,
    duracao_minutos: int,
) -> pd.DataFrame:

    config = ConfigSimulacao(
        vehicle_id=vehicle_id,
        duracao_minutos=duracao_minutos,
        intervalo_segundos=1,
        cenario=cenario,
        seed=seed,
    )

    df = gerar_telemetria(config)

    # Identificador da sessão é fundamental para evitar
    # vazamento entre treino, validação e teste.
    df.insert(
        1,
        "session_id",
        session_id,
    )

    return df


# ============================================================
# DESENVOLVIMENTO
# ============================================================

def gerar_desenvolvimento():

    print()
    print("=" * 70)
    print("GERANDO BASE NORMAL DE DESENVOLVIMENTO")
    print("=" * 70)

    sessoes = []

    for i in range(N_SESSOES_NORMAL_DESENVOLVIMENTO):

        seed = SEED_BASE_DESENVOLVIMENTO + i

        # Alternamos a duração para que o modelo não aprenda
        # somente sessões de tamanho idêntico.
        duracao = 15 + (i % 11)
        # Resultado: 15 até 25 minutos.

        session_id = f"DEV_NORMAL_{i + 1:03d}"

        vehicle_id = (
            f"VEICULO_SIM_{(i % 10) + 1:02d}"
        )

        df = preparar_sessao(
            session_id=session_id,
            vehicle_id=vehicle_id,
            cenario="NORMAL",
            seed=seed,
            duracao_minutos=duracao,
        )

        sessoes.append(df)

        print(
            f"{session_id} | "
            f"{duracao:02d} min | "
            f"{len(df)} registros"
        )

    base = pd.concat(
        sessoes,
        ignore_index=True,
    )

    pasta = PASTA_SAIDA / "desenvolvimento"
    pasta.mkdir(
        parents=True,
        exist_ok=True,
    )

    caminho = (
        pasta
        / "telemetria_normal_desenvolvimento.csv"
    )

    base.to_csv(
        caminho,
        index=False,
        encoding="utf-8",
    )

    print()
    print(
        f"Base de desenvolvimento salva em: "
        f"{caminho}"
    )

    print(
        f"Sessões: "
        f"{base['session_id'].nunique()}"
    )

    print(
        f"Registros: "
        f"{len(base)}"
    )

    return base


# ============================================================
# TESTE
# ============================================================

def gerar_teste():

    print()
    print("=" * 70)
    print("GERANDO BASE DE TESTE")
    print("=" * 70)

    sessoes = []

    contador_global = 0

    for cenario in CENARIOS_DISPONIVEIS:

        print()
        print(f"CENÁRIO: {cenario}")
        print("-" * 70)

        for i in range(
            N_SESSOES_TESTE_POR_CENARIO
        ):

            seed = (
                SEED_BASE_TESTE
                + contador_global
            )

            duracao = 15 + (i % 11)

            nome_cenario = (
                cenario
                .replace(
                    "CONDICAO_SEVERA_SIMULADA",
                    "SEVERA",
                )
                .replace(
                    "AQUECIMENTO_PROGRESSIVO",
                    "AQUECIMENTO",
                )
                .replace(
                    "OPERACAO_IRREGULAR",
                    "IRREGULAR",
                )
            )

            session_id = (
                f"TEST_{nome_cenario}_"
                f"{i + 1:03d}"
            )

            vehicle_id = (
                f"VEICULO_TESTE_"
                f"{(i % 5) + 1:02d}"
            )

            df = preparar_sessao(
                session_id=session_id,
                vehicle_id=vehicle_id,
                cenario=cenario,
                seed=seed,
                duracao_minutos=duracao,
            )

            sessoes.append(df)

            print(
                f"{session_id} | "
                f"{duracao:02d} min | "
                f"{len(df)} registros"
            )

            contador_global += 1

    base = pd.concat(
        sessoes,
        ignore_index=True,
    )

    pasta = PASTA_SAIDA / "teste"
    pasta.mkdir(
        parents=True,
        exist_ok=True,
    )

    caminho = (
        pasta
        / "telemetria_teste_cenarios.csv"
    )

    base.to_csv(
        caminho,
        index=False,
        encoding="utf-8",
    )

    print()
    print(
        f"Base de teste salva em: "
        f"{caminho}"
    )

    print(
        f"Sessões: "
        f"{base['session_id'].nunique()}"
    )

    print(
        f"Registros: "
        f"{len(base)}"
    )

    return base


# ============================================================
# RESUMO
# ============================================================

def mostrar_resumo(
    desenvolvimento: pd.DataFrame,
    teste: pd.DataFrame,
):

    print()
    print("=" * 70)
    print("RESUMO FINAL")
    print("=" * 70)

    print()
    print("DESENVOLVIMENTO")

    print(
        desenvolvimento.groupby(
            "scenario"
        ).agg(
            sessoes=(
                "session_id",
                "nunique",
            ),
            registros=(
                "session_id",
                "size",
            ),
        )
    )

    print()
    print("TESTE")

    resumo_teste = (
        teste.groupby(
            "scenario"
        )
        .agg(
            sessoes=(
                "session_id",
                "nunique",
            ),
            registros=(
                "session_id",
                "size",
            ),
        )
    )

    print(resumo_teste)

    print()
    print("IMPORTANTE:")
    print(
        "- Desenvolvimento contém somente "
        "operação NORMAL."
    )
    print(
        "- Teste contém NORMAL e os três "
        "cenários artificiais."
    )
    print(
        "- A separação futura entre treino e "
        "validação será feita por session_id."
    )
    print(
        "- Cenários artificiais não representam "
        "falhas reais de veículos."
    )


# ============================================================
# EXECUÇÃO
# ============================================================

def main():

    desenvolvimento = gerar_desenvolvimento()

    teste = gerar_teste()

    mostrar_resumo(
        desenvolvimento,
        teste,
    )


if __name__ == "__main__":
    main()