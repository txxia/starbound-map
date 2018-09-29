import ctypes
import dataclasses as dc
import typing as tp
from functools import lru_cache

import OpenGL.GL as gl
import numpy as np
from OpenGL.GL import shaders

from utils.resource import asset_path
from .model import TILES_PER_REGION, WorldView

'''
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
'''

QUAD_VERTS_BL = 0
QUAD_VERTS_BR = 1
QUAD_VERTS_TR = 2
QUAD_VERTS_TL = 3
QUAD_VERTS = np.array([
    [-1, -1],
    [1, -1],
    [1, 1],
    [-1, 1]
], np.float32)

QUAD_IDX = np.array([
    [0, 1, 2],
    [0, 2, 3]
], np.uint16)

LAYERS_PER_REGION = 2


@dc.dataclass(frozen=True, eq=True)
class RenderDimension:
    grid_size: int

    @property
    def region_count(self):
        return self.grid_size ** 2

    @property
    def region_layer_count(self):
        return self.region_count * LAYERS_PER_REGION


@dc.dataclass
class RenderState:
    dimension: RenderDimension
    vertices: np.array
    region_validity: tp.List[bool]


@dc.dataclass(frozen=True)
class RenderTarget:
    dimension: RenderDimension

    vao: tp.Any
    vbo: tp.Any
    ebo: tp.Any
    regions_texs: tp.Any
    regions_tbos: tp.Any

    vertex_shader: tp.Any
    fragment_shader: tp.Any
    program: tp.Any

    indices: np.ndarray
    region_layers: tp.Tuple[int]


@dc.dataclass
class RenderParameters:
    frame_size: np.array = dc.field(default_factory=lambda: np.zeros(2))
    """size of the framebuffer"""
    showGrid: bool = True
    rect: np.array = dc.field(default_factory=lambda: np.array([-1, -1, 1, 1]))
    """(min_x, min_y, max_x, max_y) representing region in [-1, +1]^2 to draw the map"""
    time_in_seconds: float = 0
    """time since the application started"""


def init_target(dimension: RenderDimension, initial_state: RenderState) -> RenderTarget:
    region_layers = tuple(range(dimension.region_layer_count))

    # Setting up VAO
    vao = gl.glGenVertexArrays(1)
    gl.glBindVertexArray(vao)

    # Quad vertex buffer
    vbo = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    gl.glBufferData(gl.GL_ARRAY_BUFFER,
                    initial_state.vertices.nbytes,
                    initial_state.vertices,
                    gl.GL_DYNAMIC_DRAW)
    gl.glEnableVertexAttribArray(0)
    gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

    # Finish setting up VAO
    gl.glBindVertexArray(0)

    # Quad index buffer
    ebo = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
    gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER,
                    QUAD_IDX.nbytes,
                    QUAD_IDX,
                    gl.GL_STATIC_DRAW)
    # Regions data
    regions_texs = gl.glGenTextures(dimension.region_layer_count)
    regions_tbos = gl.glGenBuffers(dimension.region_layer_count)
    for tbo in regions_tbos:
        gl.glBindBuffer(gl.GL_TEXTURE_BUFFER, tbo)
        gl.glBufferData(gl.GL_TEXTURE_BUFFER, TILES_PER_REGION * 16, None, gl.GL_DYNAMIC_DRAW)

    # Create shaders
    vs_src, fs_src = __load_shaders()
    vs = shaders.compileShader(vs_src, gl.GL_VERTEX_SHADER)
    fs = shaders.compileShader(fs_src, gl.GL_FRAGMENT_SHADER)
    program = shaders.compileProgram(vs, fs)

    return RenderTarget(
        dimension=dimension,
        vao=vao,
        vbo=vbo,
        ebo=ebo,
        regions_texs=regions_texs,
        regions_tbos=regions_tbos,
        vertex_shader=vs,
        fragment_shader=fs,
        program=program,
        indices=np.copy(QUAD_IDX),
        region_layers=region_layers
    )


def init_state(dimension: RenderDimension) -> RenderState:
    return RenderState(
        dimension=dimension,
        vertices=np.copy(QUAD_VERTS),
        region_validity=[True] * dimension.region_count
    )


def __load_shaders():
    with open(asset_path('shader/map.vs.glsl'), 'r') as vs:
        vs_src = vs.read()
    with open(asset_path('shader/map.fs.glsl'), 'r') as fs:
        fs_src = fs.read()
    return vs_src, fs_src


class WorldRenderer:
    def __init__(self, view: WorldView, grid_dim: int):
        self._view = None
        self.view = view
        self.dimension = RenderDimension(grid_size=grid_dim)
        self.state = init_state(self.dimension)
        self.target = init_target(self.dimension, self.state)

        # initial callback
        self._update_region_textures(self.target, self.state)

    @property
    def view(self):
        return self._view

    @view.setter
    def view(self, value):
        if self._view != value:
            self._view = value
            if value is not None:
                self._view.on_region_updated(lambda: self._update_region_textures(self.target, self.state))
                self._update_region_textures(self.target, self.state)

    def draw(self, params: RenderParameters):
        """
        Draw a frame of the world view.
        """
        gl.glClearColor(0, 0, 0, 1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        target = self.target
        state = self.state
        gl.glUseProgram(target.program)

        # textures
        for i, (tex, tbo) in enumerate(zip(target.regions_texs, target.regions_tbos)):
            gl.glActiveTexture(gl.GL_TEXTURE0 + i)
            gl.glBindTexture(gl.GL_TEXTURE_BUFFER, tex)
            gl.glTexBuffer(gl.GL_TEXTURE_BUFFER, gl.GL_RGBA32UI, tbo)

        # uniforms
        rect_resolution = (
            params.frame_size[0] * ((params.rect[2] - params.rect[0]) / 2),
            params.frame_size[1] * ((params.rect[3] - params.rect[1]) / 2)
        )
        # Compute fragment projection from window space to rect space [-1,1]^2
        rect01 = tuple((r + 1) * 0.5 for r in params.rect)
        rect_in_frag_space = tuple(r * params.frame_size[i % 2] for i, r in enumerate(rect01))
        frag_projection = _project_rect(rect_in_frag_space).astype(np.float32)

        gl.glUniform2f(gl.glGetUniformLocation(target.program, "iResolution"), rect_resolution[0], rect_resolution[1])
        gl.glUniform1f(gl.glGetUniformLocation(target.program, "iTime"), params.time_in_seconds)
        gl.glUniformMatrix3fv(gl.glGetUniformLocation(target.program, "iFragProjection"), 1, True, frag_projection)
        gl.glUniform1iv(gl.glGetUniformLocation(target.program, "iRegionValid"), target.dimension.region_count,
                        state.region_validity)
        gl.glUniform1iv(gl.glGetUniformLocation(target.program, "iRegionLayer"),
                        target.dimension.region_layer_count, target.region_layers)
        gl.glUniform1i(gl.glGetUniformLocation(target.program, "iConfig.showGrid"), params.showGrid)

        gl.glBindVertexArray(target.vao)

        # quad
        state.vertices[QUAD_VERTS_BL][:] = params.rect[:2]  # min_x, min_y
        state.vertices[QUAD_VERTS_BR][:] = (params.rect[2], params.rect[1])  # max_x, min_y
        state.vertices[QUAD_VERTS_TR][:] = params.rect[2:]  # max_x, max_y
        state.vertices[QUAD_VERTS_TL][:] = (params.rect[0], params.rect[3])  # min_x, max_y
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER,
                           offset=0,
                           size=state.vertices.nbytes,
                           data=state.vertices)

        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, target.ebo)
        gl.glDrawElements(gl.GL_TRIANGLES, target.indices.size, gl.GL_UNSIGNED_SHORT, None)
        gl.glBindVertexArray(0)

    def _update_region_textures(self, target: RenderTarget, state: RenderState):
        regions = sum(self.view.region_grid, []) \
            if self.view is not None \
            else [None] * target.dimension.region_count

        for i, r in enumerate(regions):
            state.region_validity[i] = r is not None

        layers: tp.List[bytes] = sum((_region_to_layers(r) for r in regions), [])

        # upload each layer to a texture buffer
        for layer, tbo in zip(layers, target.regions_tbos):
            if layer is None:
                continue
            layer_ctypes = np.frombuffer(layer).ctypes
            gl.glBindBuffer(gl.GL_TEXTURE_BUFFER, tbo)
            buf = gl.glMapBuffer(gl.GL_TEXTURE_BUFFER, gl.GL_WRITE_ONLY)
            ctypes.memmove(buf, layer_ctypes, len(layer))
            gl.glUnmapBuffer(gl.GL_TEXTURE_BUFFER)
        gl.glBindBuffer(gl.GL_TEXTURE_BUFFER, 0)


@lru_cache(maxsize=512)
def _region_to_layers(region_data: bytes) -> tp.List[tp.Optional[bytearray]]:
    if region_data is None:
        return [None, None]
    else:
        # pad 1 byte to the index 8 of each tile, making each tile 32 bytes in size
        region_data = _pad_region(region_data, tile_size=31, offset=8)
        # slice each region into 2 layers, due to the size limit of texture buffer
        return _slice_tiles(region_data, tile_size=32, slices=2)


def _slice_tiles(region_data: bytearray, *, tile_size: int, slices: int) -> tp.List[bytearray]:
    """
    :param region_data: byte string containing the region data
    :param tile_size: size of a tile in bytes
    :param slices: number of slices per tile
    :return: list containing `slices` elements

    >>> _slice_tiles(bytearray(b'\\0\\1\\2\\3' * 2), tile_size=4, slices=2)
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
            layers[s][dst_start:dst_start + slice_size] = region_data[src_start: src_start + slice_size]
    return layers


def _pad_region(region_data: bytes, *, tile_size: int, offset: int) -> bytearray:
    """
    Insert padding to the `offset`th (zero-based) byte of every tile

    >>> _pad_region(b'\\0\\1\\2\\3' * 2, tile_size=4, offset=2)
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
        padded_region[base + offset + 1: base + new_tile_size] = region_data[r + offset:r + tile_size]
    return padded_region


def _project_rect(rect: tp.Sequence[float]) -> np.ndarray:
    """
    Returns project from a rect in window space ([0,w], [0,h]) to [0,1]^2
    :param rect: (min_x, min_y, max_x, max_y)
    :return: 3x3 projection matrix

    >>> _project_rect([0, 0, 10, 20]) # scaling only
    array([[0.1 , 0.  , 0.  ],
           [0.  , 0.05, 0.  ],
           [0.  , 0.  , 1.  ]])
    >>> _project_rect([3, 4, 4, 5])   # translation only
    array([[ 1.,  0., -3.],
           [ 0.,  1., -4.],
           [ 0.,  0.,  1.]])
    >>> _project_rect([0, 6, 10, 16]) # translation + scaling
    array([[ 0.1,  0. ,  0. ],
           [ 0. ,  0.1, -0.6],
           [ 0. ,  0. ,  1. ]])
    >>> np.matmul(_project_rect([0, 6, 10, 16]), [5, 11, 1])  # projecting the center of the rect
    array([0.5, 0.5, 1. ])
    """
    rw = rect[2] - rect[0]
    rh = rect[3] - rect[1]
    # translate to origin
    translation = [
        [1, 0, -rect[0]],
        [0, 1, -rect[1]],
        [0, 0, 1]
    ]
    # scale down
    scaling = [
        [1 / rw, 0, 0],
        [0, 1 / rh, 0],
        [0, 0, 1]
    ]
    proj = np.matmul(scaling, translation)
    return proj


if __name__ == '__main__':
    import doctest

    doctest.testmod()
