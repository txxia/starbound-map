from os.path import join
from unittest import TestCase, mock

from utils import resource

MEIPASS = 'TestMEIPASS'
REL_PATH = 'TestRelPath'
ABS_ASSET_PATH = "TestAbsoluteAssetPath"


@mock.patch('utils.resource.os.path.abspath')
class TestAssetPath(TestCase):
    def test_pyinstaller(self, _):
        resource.sys._MEIPASS = MEIPASS
        asset_path = resource.asset_path(REL_PATH)
        self.assertEqual(join(MEIPASS, REL_PATH), asset_path)

    def test_dev(self, mock_abspath):
        mock_abspath.return_value = ABS_ASSET_PATH
        asset_path = resource.asset_path(REL_PATH)
        self.assertEqual(join(ABS_ASSET_PATH, REL_PATH), asset_path)
