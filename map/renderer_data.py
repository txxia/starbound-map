"""
Per tile data (31+1 bytes):
-------------------------------------
h short 2  2 foreground_material
B uchar 1  3 foreground_hue_shift
B uchar 1  4 foreground_variant

h short 2  6 foreground_mod
B uchar 1  7 foreground_mod_hue_shift
--> pad 1 byte here <--

h short 2  9 background_material
B uchar 1 10 background_hue_shift
B uchar 1 11 background_variant

h short 2 13 background_mod
B uchar 1 14 background_mod_hue_shift
B uchar 1 15 liquid
--------------------------------------
f float 4 19 liquid_level

f float 4 23 liquid_pressure

B uchar 1 24 liquid_infinite
B uchar 1 25 collision
H ushrt 2 27 dungeon_id

B uchar 1 28 biome
B uchar 1 29 biome_2
? bool  1 30 indestructible
x pad   1 31 (padding)
"""

import typing as tp
from functools import lru_cache


@lru_cache(maxsize=512)
def convert_region_to_layers(region_data: bytes) \
        -> tp.List[tp.Optional[bytearray]]:
    """
    Preprocess region data and convert it into a format ready for rendering.
    :param region_data: one region worth of data
    :return: list of data chunks representing layers
    """
    if region_data is None:
        return [None, None]
    else:
        # pad the 8th byte of each tile, making each tile 32 bytes in size
        region_data = __pad_region(region_data, tile_size=31, offset=8)
        # slice the region into 2 layers, due to the size limit of tex buffer
        return __slice_tiles(region_data, tile_size=32, slices=2)


def __slice_tiles(region_data: bytearray, *, tile_size: int, slices: int) -> \
        tp.List[bytearray]:
    """
    :param region_data: byte string containing the region data
    :param tile_size: size of a tile in bytes
    :param slices: number of slices per tile
    :return: list containing `slices` elements

    >>> __slice_tiles(bytearray(b'\\0\\1\\2\\3' * 2), tile_size=4, slices=2)
    [bytearray(b'\\x00\\x01\\x00\\x01'), bytearray(b'\\x02\\x03\\x02\\x03')]
    """
    assert tile_size % slices == 0
    region_size = len(region_data)
    tile_count = region_size // tile_size
    slice_size = tile_size // slices
    layer_size = region_size // slices
    layers = [bytearray(b'\0' * layer_size) for _ in range(slices)]
    for t in range(0, tile_count):
        dst_start = t * slice_size
        for s in range(slices):
            src_start = t * tile_size + s * slice_size
            layers[s][dst_start:dst_start + slice_size] = \
                region_data[src_start: src_start + slice_size]
    return layers


def __pad_region(region_data: bytes, *, tile_size: int,
                 offset: int) -> bytearray:
    """
    Insert padding to the `offset`th (zero-based) byte of every tile

    >>> __pad_region(b'\\0\\1\\2\\3' * 2, tile_size=4, offset=2)
    bytearray(b'\\x00\\x01\\x00\\x02\\x03\\x00\\x01\\x00\\x02\\x03')
    """
    assert len(region_data) % tile_size == 0
    assert tile_size >= offset
    tile_count = len(region_data) // tile_size
    new_tile_size = tile_size + 1
    padded_region = bytearray(b'\0' * tile_count * new_tile_size)
    for i in range(tile_count):
        base = i * new_tile_size
        r = base - i
        padded_region[base: base + offset] = region_data[r:r + offset]
        padded_region[base + offset + 1: base + new_tile_size] = \
            region_data[r + offset:r + tile_size]
    return padded_region
