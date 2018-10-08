from unittest import TestCase, mock

import numpy as np

from map import controller
from map import model
from utils import shape


class TestWorldViewController(TestCase):
    WORLD_SIZE = np.array([400, 300])
    CANVAS_SIZE = np.array([100, 50])
    FOCUS = np.zeros(2)
    ZOOM = 0
    CLIP_RECT = shape.Rect(x=-50, y=-25, width=100, height=50)

    def setUp(self):
        self.mock_world = mock.MagicMock(model.World)
        self.mock_world.t_size = self.WORLD_SIZE

        self.mock_world_view = mock.MagicMock(model.WorldView)
        self.mock_world_view.world = self.mock_world

        self.controller = controller.WorldViewController(self.mock_world_view)
        self.controller.canvas_size = self.CANVAS_SIZE
        self.controller.focus = self.FOCUS
        self.controller.zoom = self.ZOOM

    def test_trace__center(self):
        tile_coord = self.controller.trace(coord01=np.full(2, 0.5))
        np.testing.assert_array_equal(self.FOCUS, tile_coord)

    def test_trace__corner(self):
        tile_coord = self.controller.trace(coord01=np.full(2, 0))
        np.testing.assert_array_equal(self.CLIP_RECT.position,
                                      tile_coord)

    def test_clip_rect(self):
        clip_rect = self.controller.clip_rect()
        np.testing.assert_array_equal(self.CLIP_RECT.data, clip_rect.data)
