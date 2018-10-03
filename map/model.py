import io
import math
import struct
import typing as tp
from collections import namedtuple

import numpy as np

from starbound import data as sbdata
from utils import cache
from utils.shape import Rect

REGION_DIM = 32
TILES_PER_REGION = REGION_DIM * REGION_DIM

UNPADDED_TILE_SIZE = 31
PADDING_BYTE_INDEX = 7
RAW_TILE_SIZE = 32
RAW_TILE_STRUCT = struct.Struct('> hBB hB? hBB hBB f f BBH BB? x')
_RawTile = namedtuple('RawTile', [
    'foreground_material',  # short 2
    'foreground_hue_shift',  # uchar 1
    'foreground_variant',  # uchar 1

    'foreground_mod',  # short 2
    'foreground_mod_hue_shift',  # uchar 1
    'is_valid',  # (padded) bool 1

    'background_material',  # short 2
    'background_hue_shift',  # uchar 1
    'background_variant',  # uchar 1

    'background_mod',  # short 2
    'background_mod_hue_shift',  # uchar 1
    'liquid',  # uchar 1

    'liquid_level',  # float 4

    'liquid_pressure',  # float 4

    'liquid_infinite',  # uchar 1
    'collision',  # uchar 1
    'dungeon_id',  # ushrt 2

    'biome',  # uchar 1
    'biome_2',  # uchar 1
    'indestructible',  # bool 1
    # unused uchar 1
])

REGION_SIZE = TILES_PER_REGION * RAW_TILE_SIZE
VALID_TILE_PADDING = b'\1'
NULL_TILE = b'\0' * RAW_TILE_SIZE
NULL_REGION = NULL_TILE * TILES_PER_REGION


class Tile(_RawTile):

    def __new__(cls, data: bytes, *args, **kwargs):
        assert type(data) == bytes
        assert len(data) == RAW_TILE_SIZE
        attributes = RAW_TILE_STRUCT.unpack(data)
        tile_data = super().__new__(cls, *attributes, **kwargs)
        return tile_data

    def __init__(self, data: bytes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data = data

    @property
    def bytes(self) -> bytes:
        return self._data


class World:
    def __init__(self, dao: sbdata.World):
        self.__dao = dao
        self.__dao.read_metadata()

    @cache.lazy_property
    def r_width(self) -> int:
        """Width in number of regions."""
        return math.ceil(self.t_width / REGION_DIM)

    @cache.lazy_property
    def r_height(self) -> int:
        """Height in number of regions."""
        return math.ceil(self.t_height / REGION_DIM)

    @cache.lazy_property
    def t_width(self) -> int:
        """
        Width in number of tiles, which is:
        (number of regions) x (number of tiles on one side of a region)
        """
        return self.__dao.width

    @cache.lazy_property
    def t_height(self) -> int:
        """
        Height in number of tiles, which is:
        (number of regions) x (number of tiles on one side of a region)
        """
        return self.__dao.height

    @property
    def metadata(self) -> dict:
        return self.__dao.metadata

    def get_region(self, rx: int, ry: int) -> tp.Tuple[Tile]:
        assert 0 <= rx <= self.r_width, "region X out of bound"
        assert 0 <= ry <= self.r_height, "region Y out of bound"
        tile_stream = io.BytesIO(self.get_raw_tiles(rx, ry))
        return tuple(Tile(tile_stream.read(RAW_TILE_SIZE))
                     for _ in range(TILES_PER_REGION))

    @cache.memoized_method(maxsize=65536)
    def get_tile(self, tx: int, ty: int) -> Tile:
        assert 0 <= tx <= self.t_width, "tile X out of bound"
        assert 0 <= ty <= self.t_height, "tile Y out of bound"
        region = self.get_region(tx // REGION_DIM, ty // REGION_DIM)
        tile_index_in_region = (ty % REGION_DIM) * REGION_DIM + tx % REGION_DIM
        return region[tile_index_in_region]

    @cache.lazy_property
    def bytes(self) -> bytes:
        return b''.join(self.get_raw_tiles(rx, ry)
                        for rx in range(self.r_width)
                        for ry in range(self.r_height))

    @cache.memoized_method(maxsize=1024)
    def get_raw_tiles(self, rx: int, ry: int) -> bytes:
        try:
            unpadded_region = self.__dao.get_raw_tiles(rx, ry)
            return self._pad_region(unpadded_region,
                                    tile_size=UNPADDED_TILE_SIZE,
                                    offset=PADDING_BYTE_INDEX,
                                    pad_value=VALID_TILE_PADDING)
        except KeyError or RuntimeError:
            return NULL_REGION

    # TODO optimize this
    @staticmethod
    def _pad_region(region_data: bytes, *,
                    tile_size: int,
                    offset: int,
                    pad_value: bytes = b'\0') -> bytes:
        """
        Insert padding `pad_value` to the `offset`th (zero-based) byte
        of every tile.
        """
        assert len(region_data) % tile_size == 0
        assert tile_size >= offset
        assert len(pad_value) == 1
        tile_count = len(region_data) // tile_size
        new_tile_size = tile_size + 1
        padded_region = bytearray(pad_value * tile_count * new_tile_size)
        for i in range(tile_count):
            base = i * new_tile_size
            r = base - i
            padded_region[base: base + offset] = region_data[r:r + offset]
            padded_region[base + offset + 1: base + new_tile_size] = \
                region_data[r + offset:r + tile_size]
        return bytes(padded_region)


class WorldView:
    """
    Represents a view to the world map.
    """

    # boundary to prevent division by zero
    MAX_ZOOM = 1.e4
    MIN_ZOOM = 1.e-4

    def __init__(self,
                 world: World,
                 center_region: np.ndarray,
                 grid_dim: int = 5):
        """
        :param world: starbound world
        :param center_region: center tile coordinate of the view
        :param grid_dim: number of cells on any side of the grid
        """
        assert world is not None
        self._world = world

        self._focus = np.zeros(2)
        self._zoom = 1
        self._pixel_size = np.ones(2)

        # TODO deprecate fields below
        self._on_region_updated = []

        self._grid_dim = None
        self.grid_dim = grid_dim

        self._region_grid = [[None for _ in range(grid_dim)]
                             for _ in range(grid_dim)]

        self._center_region = np.zeros(2)
        self.center_region = center_region

    @property
    def world(self) -> World:
        return self._world

    @property
    def focus(self) -> np.array:
        return self._focus

    @focus.setter
    def focus(self, value: np.array):
        """Tile-level focus point."""
        assert value.shape == (2,)
        assert 0 <= value[0] <= self.world.t_width
        assert 0 <= value[1] <= self.world.t_height
        self._focus = value

    @property
    def zoom(self) -> float:
        return self._zoom

    @zoom.setter
    def zoom(self, value: float):
        self._zoom = min(max(WorldView.MIN_ZOOM, value), WorldView.MAX_ZOOM)

    @property
    def pixel_size(self) -> np.array:
        return self._pixel_size

    @pixel_size.setter
    def pixel_size(self, value: np.array):
        assert value.shape == (2,)
        assert value[0] > 0
        assert value[1] > 0
        self._pixel_size = value

    def clip_rect(self) -> Rect:
        """Tile-level clip rect of this view."""
        rect_size = self.pixel_size / self.zoom
        position = self.focus - rect_size / 2
        return Rect(position[0], position[1], rect_size[0], rect_size[1])

    @property
    def region_grid(self):
        return self._region_grid

    @property
    def center_region(self):
        return self._center_region

    @center_region.setter
    def center_region(self, value: np.ndarray):
        assert value.size == 2
        assert value.dtype == np.int
        if any(self._center_region != value):
            self._center_region = value
            self._update_regions()

    @property
    def grid_dim(self):
        return self._grid_dim

    @grid_dim.setter
    def grid_dim(self, value: int):
        assert value > 0
        if self._grid_dim != value:
            self._grid_dim = value

    def on_region_updated(self, callback):
        self._on_region_updated.append(callback)

    def get_location(self, coord01: np.ndarray) \
            -> tp.Tuple[np.ndarray, np.ndarray]:
        """
        Find the location indicated by the given coordinate in the current view
        :param coord01: coordinate in the current view
        :return: (region_coord, tile_coord)
        """
        region_coord = self.center_region - \
                       self.grid_dim // 2 + \
                       (coord01 * self.grid_dim).astype(np.int)
        tile_coord = ((coord01 % (1 / self.grid_dim)) *
                      self.grid_dim * REGION_DIM).astype(np.int)
        return region_coord, tile_coord

    def get_region(self, region_coord: tp.Sequence[int]) -> bytes:
        """
        :param region_coord: region coordinate
        :return: regions at the given location
        """
        return self.world.get_raw_tiles(region_coord[0], region_coord[1])

    def _update_regions(self):
        """
        Updates the grid of size grid_dim^2 containing regions centered at the current view.
        """
        # FIXME handle wrapping around
        # Retrieve visible regions as a grid,
        # each item is a byte array of 1024 tiles (31 bytes per tile)
        for y in range(self.grid_dim):
            ry = self.center_region[1] + y - 1
            for x in range(self.grid_dim):
                rx = self.center_region[0] + x - 1
                self.region_grid[y][x] = self.get_region((rx, ry))

        for cb in self._on_region_updated:
            cb()
