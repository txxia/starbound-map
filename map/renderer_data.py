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

from functools import lru_cache

PADDED_TILE_SIZE = 32


@lru_cache(maxsize=512)
def pad_region(region_data: bytes, *, tile_size: int,
               offset: int) -> bytes:
    """
    Insert padding to the `offset`th (zero-based) byte of every tile

    >>> pad_region(b'\\0\\1\\2\\3' * 2, tile_size=4, offset=2)
    b'\\x00\\x01\\x00\\x02\\x03\\x00\\x01\\x00\\x02\\x03'
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
    return bytes(padded_region)
