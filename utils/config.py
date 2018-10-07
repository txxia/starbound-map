import atexit
from configparser import ConfigParser

CONFIG_FILE = 'config.ini'


class Config(ConfigParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load(self):
        self.read(CONFIG_FILE)

    def save(self):
        with open(CONFIG_FILE, 'w') as f:
            self.write(f)


CONFIG = None


def init():
    global CONFIG
    CONFIG = Config()
    CONFIG.load()


init()


@atexit.register
def save_config():  # pragma: no cover
    CONFIG.save()
