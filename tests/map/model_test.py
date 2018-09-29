from unittest import TestCase, mock

import numpy as np
import numpy.testing as npt

from map import model
from starbound.data import World

CENTER_REGION = np.array([4, 2])
REGION_COORD = (4, 3)
GRID_DIM = 3  # 9x9 grid


class TestWorldView(TestCase):
    def setUp(self):
        self.mock_world = mock.MagicMock(World)
        self.mock_raw_tiles = mock.MagicMock(bytes)

        self.view = model.WorldView(self.mock_world, CENTER_REGION, GRID_DIM)

    def test_init(self):
        self.assertEqual(self.mock_world, self.view.world)
        self.assertEqual(GRID_DIM, len(self.view.region_grid))
        self.assertEqual(GRID_DIM, len(self.view.region_grid[0]))
        npt.assert_array_equal(CENTER_REGION, self.view.center_region)

    def test_get_location_center(self):
        region_coord, tile_coord = self.view.get_location(np.full(2, 0.5))
        npt.assert_array_equal(CENTER_REGION, region_coord)
        npt.assert_array_equal(np.full(2, model.REGION_DIM / 2), tile_coord)

    def test_get_location_corner(self):
        region_coord, tile_coord = self.view.get_location(
            np.full(2, 5 / 6)
        )
        npt.assert_array_equal(np.array([5, 3]), region_coord)
        npt.assert_array_equal(np.full(2, model.REGION_DIM / 2), tile_coord)
