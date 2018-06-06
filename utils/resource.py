import logging
import os
import sys


def asset_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
        logging.info("Detected PyInstaller environment")
    except Exception:
        base_path = os.path.abspath("./assets")
        logging.info("Detected dev environment")
    logging.debug("asset path: %s", base_path)
    return os.path.join(base_path, relative_path)
