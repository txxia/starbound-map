import dataclasses as dc
import typing as tp

import OpenGL.GL as gl
import numpy as np
from OpenGL.GL import shaders

from utils.resource import asset_path
from .model import TILES_PER_REGION, WorldView
from .renderer_data import pad_region, PADDED_TILE_SIZE

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

GRID_SSBO_BINDING = 1
REGION_SIZE = TILES_PER_REGION * PADDED_TILE_SIZE
EMPTY_REGION = b'\0' * REGION_SIZE


@dc.dataclass(frozen=True, eq=True)
class RenderDimension:
    grid_size: int

    @property
    def region_count(self):
        return self.grid_size ** 2


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
    grid_regions_ssbo: tp.Any

    vertex_shader: tp.Any
    fragment_shader: tp.Any
    program: tp.Any

    indices: np.ndarray


@dc.dataclass
class RenderParameters:
    frame_size: np.array = dc.field(default_factory=lambda: np.zeros(2))
    """size of the framebuffer"""
    showGrid: bool = True
    rect: np.ndarray = dc.field(
        default_factory=lambda: np.array([[-1, -1], [1, 1]]))
    """(min, max) representing region in [-1, +1]^2 to draw the map"""
    time_in_seconds: float = 0
    """time since the application started"""


def init_target(dimension: RenderDimension,
                initial_state: RenderState) -> RenderTarget:
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
    grid_regions_ssbo = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, grid_regions_ssbo)
    gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER,
                    REGION_SIZE * dimension.region_count,
                    data=None,
                    usage=gl.GL_DYNAMIC_READ)

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
        grid_regions_ssbo=grid_regions_ssbo,
        vertex_shader=vs,
        fragment_shader=fs,
        program=program,
        indices=np.copy(QUAD_IDX)
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
                self._view.on_region_updated(
                    lambda: self._update_region_textures(self.target,
                                                         self.state))
                self._update_region_textures(self.target, self.state)

    def draw(self, params: RenderParameters):
        """
        Draw a frame of the world view.
        """
        gl.glClearColor(0, 0, 0, 1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        gl.glUseProgram(self.target.program)

        self._update_params(self.target, self.state, params)

        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.target.ebo)
        gl.glDrawElements(gl.GL_TRIANGLES, self.target.indices.size,
                          gl.GL_UNSIGNED_SHORT, None)
        gl.glBindVertexArray(0)

    def _update_params(self,
                       target: RenderTarget,
                       state: RenderState,
                       params: RenderParameters):
        # Uniforms
        rect_size = params.rect[1] - params.rect[0]
        rect_resolution = params.frame_size * rect_size / 2
        # Compute fragment projection from window space to rect space [-1,1]^2
        rect01 = (params.rect + 1) * 0.5
        rect_in_frag_space = rect01 * params.frame_size
        frag_projection = _project_rect(rect_in_frag_space).astype(np.float32)

        gl.glUniform2f(gl.glGetUniformLocation(target.program, "iResolution"),
                       rect_resolution[0], rect_resolution[1])
        gl.glUniform1f(gl.glGetUniformLocation(target.program, "iTime"),
                       params.time_in_seconds)
        gl.glUniformMatrix3fv(
            gl.glGetUniformLocation(target.program, "iFragProjection"), 1, True,
            frag_projection)
        gl.glUniform1iv(gl.glGetUniformLocation(target.program, "iRegionValid"),
                        target.dimension.region_count,
                        state.region_validity)
        gl.glUniform1i(
            gl.glGetUniformLocation(target.program, "iConfig.showGrid"),
            params.showGrid)

        gl.glBindVertexArray(target.vao)

        # quad
        state.vertices[QUAD_VERTS_BL][:] = params.rect[0]  # min_x, min_y
        state.vertices[QUAD_VERTS_BR][:] = (
            params.rect[1][0], params.rect[0][1])  # max_x, min_y
        state.vertices[QUAD_VERTS_TR][:] = params.rect[1]  # max_x, max_y
        state.vertices[QUAD_VERTS_TL][:] = (
            params.rect[0][0], params.rect[1][1])  # min_x, max_y
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER,
                           offset=0,
                           size=state.vertices.nbytes,
                           data=state.vertices)

    def _update_region_textures(self, target: RenderTarget, state: RenderState):
        regions = sum((tuple(row) for row in self.view.region_grid), tuple()) \
            if self.view \
            else (None,) * target.dimension.region_count

        for i, r in enumerate(regions):
            state.region_validity[i] = r is not None

        regions_padded = tuple(pad_region(r, tile_size=31, offset=7)
                               if r else None
                               for r in regions)

        # upload combined regions to shader storage buffer
        regions_combined = b''.join(r if r else EMPTY_REGION
                                    for r in regions_padded)
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, target.grid_regions_ssbo)
        gl.glBufferSubData(gl.GL_SHADER_STORAGE_BUFFER,
                           0,
                           REGION_SIZE * target.dimension.region_count,
                           data=regions_combined
                           )
        gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER,
                            GRID_SSBO_BINDING,
                            target.grid_regions_ssbo)


def _project_rect(rect: np.ndarray) -> np.ndarray:
    """
    Returns projection from a rect in window space ([0,w], [0,h]) to [0,1]^2
    :param rect: ((min_x, min_y), (max_x, max_y))
    :return: 3x3 projection matrix
    """
    size = rect[1] - rect[0]
    # translate to origin
    translation = [
        [1, 0, -rect[0][0]],
        [0, 1, -rect[0][1]],
        [0, 0, 1]
    ]
    # scale down
    scaling = [
        [1 / size[0], 0, 0],
        [0, 1 / size[1], 0],
        [0, 0, 1]
    ]
    proj = np.matmul(scaling, translation)
    return proj
