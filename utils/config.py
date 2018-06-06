import atexit
import configparser

CONFIG_FILE = 'config.ini'


class Config(configparser.ConfigParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load(self):
        self.read(CONFIG_FILE)

    def save(self):
        with open(CONFIG_FILE, 'w') as f:
            self.write(f)


CONFIG = Config()
CONFIG.load()


@atexit.register
def save_config():
    CONFIG.save()
