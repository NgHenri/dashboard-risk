# log_conf.py
import logging
import sys
from logging.handlers import RotatingFileHandler
import streamlit as st


class StreamlitLogHandler(logging.Handler):
    """Handler personnalisé pour afficher les logs dans Streamlit"""

    def emit(self, record):
        log_entry = self.format(record)
        if "logs" not in st.session_state:
            st.session_state.logs = []
        st.session_state.logs.append(log_entry)

        # Garde seulement les 50 derniers logs
        if len(st.session_state.logs) > 50:
            st.session_state.logs.pop(0)


# Ajoutez des couleurs avec :
class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[94m",  # Bleu
        logging.INFO: "\033[92m",  # Vert
        logging.WARNING: "\033[93m",  # Jaune
        logging.ERROR: "\033[91m",  # Rouge
        logging.CRITICAL: "\033[91m",  # Rouge
    }

    def format(self, record):
        color = self.COLORS.get(record.levelno, "")
        message = super().format(record)
        return f"{color}{message}\033[0m"  # Reset color


# Filtrez les logs sensibles :
class SensitiveDataFilter(logging.Filter):
    def filter(self, record):
        if "password" in record.getMessage().lower():
            return False
        return True


def setup_logger(name="LoanApp", log_file="app.log"):
    """
    Configure un logger avec des handlers pour console, fichier et Streamlit

    Args:
        name (str): Nom du logger
        log_file (str): Chemin du fichier de logs

    Returns:
        logging.Logger: Logger configuré
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Capture tout à partir de DEBUG

    # Formatter “classique” (fichiers & Streamlit)
    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)-15s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Formatter coloré pour la console
    color_formatter = ColorFormatter(
        "[%(asctime)s] %(levelname)-8s %(name)-15s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console avec couleur
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(color_formatter)
    console_handler.setLevel(logging.DEBUG)

    # Fichier (sans couleur)
    file_handler = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(plain_formatter)
    file_handler.setLevel(logging.DEBUG)

    # Streamlit (sans couleur)
    streamlit_handler = StreamlitLogHandler()
    streamlit_handler.setFormatter(plain_formatter)
    streamlit_handler.setLevel(logging.INFO)

    # Replace existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(streamlit_handler)

    return logger
