from unittest import TestCase, mock

from utils import config

mock_open = mock.mock_open()


class TestConfig(TestCase):
    def setUp(self):
        self.config = config.Config()

    @mock.patch('utils.config.Config.read')
    def test_load(self, mock_read):
        self.config.load()
        mock_read.assert_called_with(config.CONFIG_FILE)

    @mock.patch('builtins.open', mock_open)
    @mock.patch('utils.config.Config.write')
    def test_save(self, mock_write):
        self.config.save()
        print(mock_open.mock_calls)
        mock_open.assert_called_with(config.CONFIG_FILE, 'w')
        mock_write.assert_called_with(mock_open())
