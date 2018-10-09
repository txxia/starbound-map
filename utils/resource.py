import os
import sys

from utils import application as app

DEV_ASSET_PATH = "./assets"


def asset_path(relative_path: str) -> str:
    """ Get absolute path to resource, works for dev and for PyInstaller """

    if app.development:
        base_path = os.path.abspath(DEV_ASSET_PATH)
    else:
        base_path = sys._MEIPASS
    return os.path.join(base_path, relative_path)
