"""Launcher script for H5 Processor Streamlit application."""

import sys
import os


def main_h5_processor():
    """Entry point for the H5 Processor Streamlit application.
    
    Usage:
        xpdfsuite-h5-processor
        xpdfsuite-h5-processor --port 8503 --headless
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        prog="xpdfsuite-h5-processor",
        description="Launch the H5 Processor Streamlit application for PDF analysis.",
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="Streamlit server port (--server.port option).",
    )
    parser.add_argument(
        "--server-address", default=None,
        help="Streamlit server address (--server.address option), "
             "e.g., 0.0.0.0 for network access.",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Do not open browser automatically.",
    )
    args, extra_args = parser.parse_known_args()

    app_file = os.path.join(os.path.dirname(__file__), "app_h5_processor.py")
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


if __name__ == "__main__":
    main_h5_processor()
