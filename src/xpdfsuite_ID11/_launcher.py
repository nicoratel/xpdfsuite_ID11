import sys
import os
import argparse





def main_monitor():
    """Point d'entrée CLI pour l'application de suivi des pics PDF
    (monitor_pdfpeaks.py), lancée via Streamlit.

    Exemples :
        xpdfsuite-monitor
        xpdfsuite-monitor --data-root /data/experience_2026
        xpdfsuite-monitor --port 8502 --server-address 0.0.0.0 --headless
    """
    parser = argparse.ArgumentParser(
        prog="xpdfsuite-monitor",
        description="Lance l'application Streamlit de suivi des pics PDF (monitor_pdfpeaks).",
    )
    parser.add_argument(
        "--data-root", "-d",
        help="Dossier racine des données à ouvrir au démarrage "
             "(équivalent à définir la variable d'environnement ID11_DATA_ROOT).",
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="Port du serveur Streamlit (option --server.port).",
    )
    parser.add_argument(
        "--server-address", default=None,
        help="Adresse d'écoute du serveur Streamlit (option --server.address), "
             "par exemple 0.0.0.0 pour un accès depuis le réseau.",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="N'ouvre pas de navigateur automatiquement (utile sur un serveur distant).",
    )
    args, extra_args = parser.parse_known_args()

    if args.data_root:
        os.environ["ID11_DATA_ROOT"] = os.path.abspath(args.data_root)

    app_file = os.path.join(os.path.dirname(__file__), "monitor_pdfpeaks.py")
    from streamlit.web import cli as stcli

    argv = ["streamlit", "run", app_file]
    if args.port is not None:
        argv += ["--server.port", str(args.port)]
    if args.server_address is not None:
        argv += ["--server.address", args.server_address]
    if args.headless:
        argv += ["--server.headless", "true"]
    argv += extra_args

    sys.argv = argv
    sys.exit(stcli.main())
