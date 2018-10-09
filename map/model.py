import io
import math
import struct
import typing as tp
from collections import namedtuple

import numpy as np

import starbound.data as sbdata
from utils import cache

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
VALID_TILE_PADDING = 1
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
    def __init__(self, dao: sbdata.World, coordinates: str):
        self.__dao = dao
        self.__dao.read_metadata()
        self.__coordinates = coordinates

    @cache.lazy_property
    def coordinates(self) -> str:
        return self.__coordinates  # TODO prettify this

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

    @cache.lazy_property
    def t_size(self) -> np.ndarray:
        return np.array([self.t_width, self.t_height])

    @cache.lazy_property
    def t_count(self) -> int:
        return self.t_size.prod()

    @cache.lazy_property
    def r_size(self) -> np.ndarray:
        return np.array([self.r_width, self.r_height])

    @cache.lazy_property
    def r_count(self) -> int:
        return self.r_size.prod()

    @property
    def metadata(self) -> dict:
        return self.__dao.metadata

    def get_region(self, rx: int, ry: int) -> tp.Tuple[Tile]:
        assert 0 <= rx <= self.r_width, "region X out of bound"
        assert 0 <= ry <= self.r_height, "region Y out of bound"
        tile_stream = io.BytesIO(self.get_raw_tiles(rx, ry))
        return tuple(Tile(tile_stream.read(RAW_TILE_SIZE))
                     for _ in range(TILES_PER_REGION))

    def is_valid_tile_coord(self, tx: int, ty: int) -> bool:
        return 0 <= tx < self.t_width and \
               0 <= ty < self.t_height

    @cache.memoized_method(maxsize=65536)
    def get_tile(self, tx: int, ty: int) -> Tile:
        assert self.is_valid_tile_coord(tx, ty), "tile coordinates out of bound"
        region = self.get_region(tx // REGION_DIM, ty // REGION_DIM)
        tile_index_in_region = (ty % REGION_DIM) * REGION_DIM + tx % REGION_DIM
        return region[tile_index_in_region]

    def raw_regions(self) -> tp.Iterator[bytes]:
        return (self.get_raw_tiles(rx, ry)
                for ry in range(self.r_height)
                for rx in range(self.r_width))

    @cache.memoized_method(maxsize=1024)
    def get_raw_tiles(self, rx: int, ry: int) -> bytes:
        try:
            assert 0 <= rx < self.r_width
            assert 0 <= ry < self.r_height
            unpadded_region = self.__dao.get_raw_tiles(rx, ry)
            return self._pad_region(unpadded_region,
                                    tile_size=UNPADDED_TILE_SIZE,
                                    offset=PADDING_BYTE_INDEX,
                                    pad_value=VALID_TILE_PADDING)
        except (AssertionError, KeyError, RuntimeError):
            return NULL_REGION

    @staticmethod
    def _pad_region(region_data: bytes, *,
                    tile_size: int,
                    offset: int,
                    pad_value: int = 0) -> bytes:
        """
        Insert padding `pad_value` to the `offset`th (zero-based) byte
        of every tile.
        """
        region_size = len(region_data)
        assert len(region_data) % tile_size == 0
        assert tile_size >= offset
        assert pad_value < 2 ** 8
        tile_count = region_size // tile_size
        original_linear = np.frombuffer(region_data, dtype=np.ubyte)
        original_2d = original_linear.reshape(tile_count,
                                              tile_size)
        padded = np.insert(original_2d, offset, pad_value, axis=1)
        return padded.tobytes()


class WorldView:
    """
    Represents a view to the world map.
    """

    # boundary to prevent division by zero
    MAX_ZOOM = 1.e4
    MIN_ZOOM = -1.e4

    def __init__(self,
                 world: World):
        """
        :param world: starbound world
        """
        assert world is not None
        self._world = world

        self._focus = np.zeros(2)
        self._zoom = 0

    @property
    def world(self) -> World:
        return self._world

    @property
    def focus(self) -> np.ndarray:
        """Tile-level focus point."""
        return self._focus

    @focus.setter
    def focus(self, value: np.ndarray):
        assert value.dtype.kind == 'f'
        assert value.shape == (2,)
        self._focus = value

    @property
    def zoom(self) -> float:
        """
        Zoom factor.
        Use zero zoom to display one tile per pixel.
        Use higher/positive zoom to see more details.
        Use lower/negative zoom to see the overall picture.
        """
        return self._zoom

    @zoom.setter
    def zoom(self, value: float):
        self._zoom = min(max(WorldView.MIN_ZOOM, value), WorldView.MAX_ZOOM)
