from pathlib import Path
import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

from tensorflow.keras.models import load_model


ROOT = Path(__file__).resolve().parents[1]


MODELOS = [
    {
        "nome": "RUL",
        "origem": ROOT / "modelo" / "fd001_capped125" / "modelo_rul.h5",
        "destino": ROOT / "modelo" / "fd001_capped125" / "modelo_rul_convertido.keras",
    },
    {
        "nome": "ANOMALIA",
        "origem": ROOT / "modelo" / "fd001_anomalia" / "modelo_autoencoder.h5",
        "destino": ROOT / "modelo" / "fd001_anomalia" / "modelo_autoencoder_convertido.keras",
    },
]


def converter():
    print("=" * 70)
    print("CONVERSAO DE MODELOS FD001")
    print("=" * 70)

    for item in MODELOS:
        nome = item["nome"]
        origem = item["origem"]
        destino = item["destino"]

        print(f"\nModelo: {nome}")
        print(f"Origem:  {origem}")
        print(f"Destino: {destino}")

        if not origem.exists():
            raise FileNotFoundError(
                f"Modelo de origem nao encontrado: {origem}"
            )

        print("Carregando modelo legado...")

        modelo = load_model(
            origem,
            compile=False
        )

        print("Salvando copia convertida...")

        modelo.save(destino)

        if not destino.exists():
            raise RuntimeError(
                f"Arquivo convertido nao foi criado: {destino}"
            )

        print("Conversao concluida.")

    print("\n" + "=" * 70)
    print("CONVERSAO FINALIZADA")
    print("=" * 70)


if __name__ == "__main__":
    converter()