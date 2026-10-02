import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

from streamlit.web import cli as stcli


NOME_APP = "UPX 2.0 - Manutenção Preditiva"
HOST = "127.0.0.1"


def obter_raiz_aplicacao() -> Path:
    """
    Localiza a raiz que contém app/, modelo/, datasets/, utils/ etc.

    Desenvolvimento:
        UPX 2.0/
            distribuicao/
                launcher.py
            app/
            modelo/
            datasets/
            resultados/
            utils/

    PyInstaller --onedir:
        UPX_2_0/
            UPX_2_0.exe
            _internal/
                app/
                modelo/
                datasets/
                resultados/
                utils/
    """

    # Quando executado pelo PyInstaller, os arquivos adicionados
    # pelo .spec ficam disponíveis na raiz interna do pacote.
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS).resolve()

    # Durante o desenvolvimento:
    # distribuicao/launcher.py -> raiz UPX 2.0
    return Path(__file__).resolve().parents[1]


def encontrar_porta_livre() -> int:
    """
    Solicita ao Windows uma porta local disponível.
    """

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((HOST, 0))
        return sock.getsockname()[1]


def porta_esta_aberta(porta: int) -> bool:
    """
    Verifica se o servidor Streamlit já está aceitando conexões.
    """

    try:
        with socket.create_connection(
            (HOST, porta),
            timeout=1,
        ):
            return True

    except OSError:
        return False


def abrir_navegador_quando_pronto(
    porta: int,
    timeout: int = 120,
) -> None:
    """
    Aguarda o servidor Streamlit ficar disponível
    e abre o navegador automaticamente.

    Essa função roda em uma thread separada porque
    o Streamlit utiliza o fluxo principal da aplicação.
    """

    inicio = time.time()

    while time.time() - inicio < timeout:

        if porta_esta_aberta(porta):

            endereco = f"http://{HOST}:{porta}"

            print()
            print("=" * 60)
            print(NOME_APP)
            print("=" * 60)
            print(f"Plataforma disponível em: {endereco}")
            print("=" * 60)
            print()

            webbrowser.open(endereco)

            return

        time.sleep(0.5)

    print()
    print("=" * 60)
    print("ERRO AO INICIAR A PLATAFORMA")
    print("=" * 60)
    print(
        "O servidor Streamlit não ficou disponível "
        "no tempo esperado."
    )
    print("=" * 60)


def iniciar_plataforma() -> None:
    """
    Prepara o ambiente e inicia a aplicação Streamlit.
    """

    # ========================================================
    # RAIZ DA APLICAÇÃO
    # ========================================================

    raiz = obter_raiz_aplicacao()

    app_streamlit = raiz / "app" / "app.py"


    # ========================================================
    # DIRETÓRIO DE TRABALHO
    # ========================================================
    #
    # IMPORTANTE:
    #
    # Vários módulos da plataforma utilizam caminhos relativos:
    #
    # modelo/...
    # datasets/...
    # resultados/...
    #
    # Durante o desenvolvimento, esses caminhos partem da pasta
    # UPX 2.0.
    #
    # No executável PyInstaller, os recursos ficam dentro de:
    #
    # UPX_2_0/_internal/
    #
    # Portanto, alteramos o diretório de trabalho para a raiz
    # retornada por obter_raiz_aplicacao().
    #
    # Isso faz com que:
    #
    # Path("modelo")
    # Path("datasets")
    # Path("resultados")
    #
    # funcionem tanto no desenvolvimento quanto no executável.
    # ========================================================

    os.chdir(raiz)


    # ========================================================
    # VERIFICAÇÃO DA APLICAÇÃO
    # ========================================================

    if not app_streamlit.exists():

        raise FileNotFoundError(
            "Não foi possível localizar a aplicação.\n\n"
            f"Caminho procurado:\n{app_streamlit}"
        )


    # ========================================================
    # TENSORFLOW / KERAS
    # ========================================================

    os.environ.setdefault(
        "TF_USE_LEGACY_KERAS",
        "1",
    )


    # ========================================================
    # PYTHON PATH
    # ========================================================
    #
    # Permite que app.py e as páginas encontrem módulos
    # próprios da plataforma, principalmente utils/.
    # ========================================================

    raiz_str = str(raiz)

    if raiz_str not in sys.path:

        sys.path.insert(
            0,
            raiz_str,
        )


    # ========================================================
    # PORTA LOCAL
    # ========================================================

    porta = encontrar_porta_livre()


    # ========================================================
    # ABERTURA AUTOMÁTICA DO NAVEGADOR
    # ========================================================

    thread_navegador = threading.Thread(
        target=abrir_navegador_quando_pronto,
        args=(porta,),
        daemon=True,
    )

    thread_navegador.start()


    # ========================================================
    # STREAMLIT
    # ========================================================
    #
    # O Streamlit é executado diretamente dentro do processo
    # Python/PyInstaller.
    #
    # NÃO usamos:
    #
    #     sys.executable -m streamlit
    #
    # porque no executável empacotado sys.executable aponta
    # para UPX_2_0.exe e não para python.exe.
    #
    # global.developmentMode=false é necessário para permitir
    # configurar explicitamente server.port no executável.
    # ========================================================

    sys.argv = [
        "streamlit",
        "run",
        str(app_streamlit),

        "--global.developmentMode",
        "false",

        "--server.address",
        HOST,

        "--server.port",
        str(porta),

        "--server.headless",
        "true",

        "--browser.gatherUsageStats",
        "false",
    ]


    # ========================================================
    # INICIALIZAÇÃO DO STREAMLIT
    # ========================================================

    stcli.main()


def main() -> None:
    """
    Ponto de entrada da aplicação.
    """

    try:

        iniciar_plataforma()


    except KeyboardInterrupt:

        print()
        print("Encerrando a plataforma...")


    except SystemExit:

        raise


    except Exception as erro:

        print()
        print("=" * 60)
        print("ERRO AO INICIAR A PLATAFORMA")
        print("=" * 60)

        print(str(erro))

        print("=" * 60)

        input(
            "\nPressione ENTER para fechar..."
        )

        sys.exit(1)


if __name__ == "__main__":
    main()