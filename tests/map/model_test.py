from unittest import TestCase, mock

import numpy as np
import numpy.testing as npt

import starbound.data as sbdata
from map import model
from utils.shape import Rect

WORLD_WIDTH_IN_TILES = 400
WORLD_HEIGHT_IN_TILES = 300
WORLD_SIZE_IN_TILES = np.array(
    [WORLD_WIDTH_IN_TILES, WORLD_HEIGHT_IN_TILES])
WORLD_TILE_COUNT = WORLD_SIZE_IN_TILES.prod()

WORLD_WIDTH_IN_REGIONS = WORLD_WIDTH_IN_TILES // model.REGION_DIM + 1
WORLD_HEIGHT_IN_REGIONS = WORLD_HEIGHT_IN_TILES // model.REGION_DIM + 1
WORLD_SIZE_IN_REGIONS = np.array(
    [WORLD_WIDTH_IN_REGIONS, WORLD_HEIGHT_IN_REGIONS])
WORLD_REGION_COUNT = WORLD_SIZE_IN_REGIONS.prod()

PADDED_TILE_DATA = b'\
\xff\xff\x00\x00\xff\xff\x00\x01\
\x03\x00\x00\xff\xff\x00\x00\x00\
\x00\x00\x00\x00\x00\x00\x00\x00\
\x01\xff\xff\x11\x10\x00\x00\xff'

TILE_DATA = b'\
\xff\xff\x00\x00\xff\xff\x00\
\x03\x00\x00\xff\xff\x00\x00\x00\
\x00\x00\x00\x00\x00\x00\x00\x00\
\x01\xff\xff\x11\x10\x00\x00\xff'

NULL_TILE_DATA = b'\0' * model.UNPADDED_TILE_SIZE

REGION_DATA = TILE_DATA * model.TILES_PER_REGION


class TestTile(TestCase):

    def setUp(self):
        self.tile = model.Tile(PADDED_TILE_DATA)

    def test_instantiate(self):
        assert PADDED_TILE_DATA == self.tile.bytes

        assert -1 == self.tile.foreground_material
        assert 0 == self.tile.foreground_hue_shift
        assert 0 == self.tile.foreground_variant

        assert -1 == self.tile.foreground_mod
        assert 0 == self.tile.foreground_mod_hue_shift

        assert 768 == self.tile.background_material
        assert 0 == self.tile.background_hue_shift
        assert 255 == self.tile.background_variant

        assert -256 == self.tile.background_mod
        assert 0 == self.tile.background_mod_hue_shift
        assert 0 == self.tile.liquid

        assert 0.0 == self.tile.liquid_level

        assert 0.0 == self.tile.liquid_pressure

        assert 1 == self.tile.liquid_infinite
        assert 255 == self.tile.collision
        assert 65297 == self.tile.dungeon_id

        assert 16 == self.tile.biome
        assert 0 == self.tile.biome_2
        assert not self.tile.indestructible

        assert self.tile.is_valid

    def test_is_valid__true(self):
        invalid_tile = model.Tile(model.NULL_TILE)
        assert not invalid_tile.is_valid


class TestWorld(TestCase):
    REGION_COORD = np.array([1, 2])

    METADATA = mock.MagicMock()

    def setUp(self):
        self.mock_dao = mock.MagicMock(sbdata.World, autospec=True)
        self.mock_dao.width = WORLD_WIDTH_IN_TILES
        self.mock_dao.height = WORLD_HEIGHT_IN_TILES
        self.mock_dao.metadata = self.METADATA

        self.world = model.World(self.mock_dao)

    def test_properties(self):
        assert WORLD_WIDTH_IN_TILES == self.world.t_width
        assert WORLD_HEIGHT_IN_TILES == self.world.t_height
        assert WORLD_TILE_COUNT == self.world.t_count

        assert WORLD_WIDTH_IN_REGIONS == self.world.r_width
        assert WORLD_HEIGHT_IN_REGIONS == self.world.r_height
        assert WORLD_REGION_COUNT == self.world.r_count

        assert self.METADATA == self.world.metadata

    def test_get_region(self):
        self.mock_dao.get_raw_tiles.return_value = REGION_DATA

        region_tiles = self.world.get_region(*self.REGION_COORD)

        self.mock_dao.get_raw_tiles.assert_called_once_with(*self.REGION_COORD)

        assert model.TILES_PER_REGION == len(region_tiles)
        assert region_tiles[0].bytes == PADDED_TILE_DATA

    def test_get_region__out_of_bound_negative(self):
        with self.assertRaises(AssertionError):
            self.world.get_region(0, -1)

    def test_get_region__out_of_bound_positive(self):
        with self.assertRaises(AssertionError):
            self.world.get_region(WORLD_WIDTH_IN_REGIONS + 1, 0)

    def test_get_region__null(self):
        self.mock_dao.get_raw_tiles = mock.Mock(side_effect=KeyError)

        region_tiles = self.world.get_region(*self.REGION_COORD)

        self.mock_dao.get_raw_tiles.assert_called_once_with(*self.REGION_COORD)

        assert model.TILES_PER_REGION == len(region_tiles)
        assert not any(t.is_valid for t in region_tiles)

    def test_is_valid_tile_coord(self):
        assert self.world.is_valid_tile_coord(WORLD_WIDTH_IN_TILES - 1,
                                              WORLD_HEIGHT_IN_TILES - 1)

    def test_is_valid_tile_coord__invalid(self):
        assert not self.world.is_valid_tile_coord(-1, 0)

    def test_get_tile(self):
        # set up a special tile at (1, 0) of the given region
        region_offset = self.REGION_COORD * model.REGION_DIM
        null_tile_id = region_offset + np.array((1, 0))
        regular_tile_id = region_offset + np.array((0, 1))
        region_data = bytearray(REGION_DATA)
        region_data[model.UNPADDED_TILE_SIZE:
                    model.UNPADDED_TILE_SIZE * 2] = NULL_TILE_DATA
        self.mock_dao.get_raw_tiles.return_value = bytes(region_data)

        regular_tile = self.world.get_tile(*regular_tile_id)
        null_tile = self.world.get_tile(*null_tile_id)
        assert PADDED_TILE_DATA == regular_tile.bytes
        assert PADDED_TILE_DATA != null_tile

        self.mock_dao.get_raw_tiles.assert_called_once_with(*self.REGION_COORD)

    def test_get_tile__out_of_bound_negative(self):
        with self.assertRaises(AssertionError):
            self.world.get_tile(-1, 0)

    def test_get_tile__out_of_bound_positive(self):
        with self.assertRaises(AssertionError):
            self.world.get_tile(0, WORLD_WIDTH_IN_TILES + 1)

    def test_get_all_tiles(self):
        self.mock_dao.get_raw_tiles.return_value = REGION_DATA
        region_iter = self.world.raw_regions()
        assert WORLD_REGION_COUNT == sum(1 for _ in region_iter)

    def test_pad_region(self):
        padded_region = model.World._pad_region(b'\1\2\1\2',
                                                tile_size=2,
                                                offset=2)
        assert b'\1\2\0\1\2\0' == padded_region


class TestWorldView(TestCase):
    FRAME_SIZE = WORLD_SIZE_IN_TILES
    ZOOM = 1
    FOCUS = np.zeros(2, dtype=np.float)
    CLIP_RECT = Rect(-FRAME_SIZE[0] / 2,
                     -FRAME_SIZE[1] / 2,
                     FRAME_SIZE[0],
                     FRAME_SIZE[1])

    def setUp(self):
        self.mock_world = mock.MagicMock(model.World)
        self.mock_world.t_width = WORLD_WIDTH_IN_TILES
        self.mock_world.t_height = WORLD_HEIGHT_IN_TILES

        self.mock_raw_tiles = mock.MagicMock(bytes)

        self.view = model.WorldView(self.mock_world)
        self.view.zoom = self.ZOOM
        self.view.focus = self.FOCUS

    def test_init(self):
        assert self.mock_world == self.view.world

    def test_focus(self):
        valid_focus = WORLD_SIZE_IN_TILES - 0.1
        self.view.focus = valid_focus
        npt.assert_array_equal(valid_focus, self.view.focus)

    def test_focus__out_of_range(self):
        invalid_focus = np.array([-1, -1])
        with self.assertRaises(AssertionError):
            self.view.focus = invalid_focus

    def test_zoom(self):
        valid_zoom = 42
        self.view.zoom = valid_zoom
        assert valid_zoom == self.view.zoom

    def test_zoom__boundary(self):
        self.view.zoom = float('-inf')
        assert model.WorldView.MIN_ZOOM == self.view.zoom
        self.view.zoom = float('inf')
        assert model.WorldView.MAX_ZOOM == self.view.zoom

    def test_clip_rect(self):
        clip_rect = self.view.clip_rect(self.FRAME_SIZE)
        npt.assert_array_equal(self.CLIP_RECT.data, clip_rect.data)

    def test_trace__center(self):
        tile_coord = self.view.trace(frame_size=self.FRAME_SIZE,
                                     coord01=np.full(2, 0.5))
        npt.assert_array_equal(self.FOCUS, tile_coord)

    def test_trace__corner(self):
        tile_coord = self.view.trace(frame_size=self.FRAME_SIZE,
                                     coord01=np.full(2, 0))
        npt.assert_array_equal(self.CLIP_RECT.position, tile_coord)
