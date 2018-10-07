import dataclasses as dc
import typing as tp

import OpenGL.GL as gl
import numpy as np
from OpenGL.GL import shaders

from utils import asyncjob
from utils.resource import asset_path
from utils.shape import Rect
from .model import REGION_SIZE, WorldView

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

WORLD_SSBO_BINDING = 0
EMPTY_REGION = b'\0' * REGION_SIZE


@dc.dataclass
class RenderState:
    vertices: np.ndarray


@dc.dataclass(frozen=True)
class RenderTarget:
    vao: tp.Any
    vbo: tp.Any
    ebo: tp.Any
    world_ssbo: tp.Any

    vertex_shader: tp.Any
    fragment_shader: tp.Any
    program: tp.Any

    indices: np.ndarray


@dc.dataclass
class RenderParameters:
    frame_size: np.ndarray = dc.field(default_factory=lambda: np.zeros(2))
    """size of the framebuffer"""
    showGrid: bool = True
    rect: Rect = dc.field(default_factory=lambda: Rect())
    """(min, max) representing region in [-1, +1]^2 to draw the map"""
    time_in_seconds: float = 0
    """time since the application started"""


def init_target(initial_state: RenderState) -> RenderTarget:  # pragma: no cover
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
    world_ssbo = gl.glGenBuffers(1)

    # Create shaders
    vs_src, fs_src = __load_shaders()
    vs = shaders.compileShader(vs_src, gl.GL_VERTEX_SHADER)
    fs = shaders.compileShader(fs_src, gl.GL_FRAGMENT_SHADER)
    program = shaders.compileProgram(vs, fs)

    return RenderTarget(
        vao=vao,
        vbo=vbo,
        ebo=ebo,
        world_ssbo=world_ssbo,
        vertex_shader=vs,
        fragment_shader=fs,
        program=program,
        indices=np.copy(QUAD_IDX)
    )


def init_state() -> RenderState:
    return RenderState(
        vertices=np.copy(QUAD_VERTS),
    )


def __load_shaders():
    with open(asset_path('shader/map.vs.glsl'), 'r') as vs:
        vs_src = vs.read()
    with open(asset_path('shader/map.fs.glsl'), 'r') as fs:
        fs_src = fs.read()
    return vs_src, fs_src


class WorldRenderer:  # pragma: no cover
    def __init__(self, view: tp.Optional[WorldView]):
        self._view: tp.Optional[WorldView] = None
        self.change_view(view)
        self.state = init_state()
        self.target = init_target(self.state)

    def change_view(self, value: tp.Optional[WorldView]):
        if self._view != value:
            self._view = value
            if value is not None:
                self._update_region_buffer(self.target)

    def draw(self, params: RenderParameters):
        """
        Draw a frame of the world view.
        """
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
        rect_resolution = params.frame_size * params.rect.size
        # Compute fragment projection from window space to rect space [-1,1]^2
        rect_in_frag_space = params.rect.data * params.frame_size
        frag_projection = _project_rect(np.array([
            rect_in_frag_space[0],
            rect_in_frag_space[0] + rect_in_frag_space[1]
        ]))
        gl.glUniform2f(gl.glGetUniformLocation(target.program, "iResolution"),
                       rect_resolution[0], rect_resolution[1])
        gl.glUniform1f(gl.glGetUniformLocation(target.program, "iTime"),
                       params.time_in_seconds)
        gl.glUniformMatrix3fv(
            gl.glGetUniformLocation(target.program, "iFragProjection"), 1, True,
            frag_projection)
        gl.glUniform1i(
            gl.glGetUniformLocation(target.program, "iConfig.showGrid"),
            params.showGrid)

        if self._view:
            clip_rect = self._view.clip_rect(rect_resolution)
            gl.glUniform2i(
                gl.glGetUniformLocation(target.program,
                                        "iView.worldRSize"),
                self._view.world.r_width, self._view.world.r_height)
            gl.glUniform2fv(
                gl.glGetUniformLocation(target.program,
                                        "iView.clipRect.position"), 1,
                clip_rect.position)
            gl.glUniform2fv(
                gl.glGetUniformLocation(target.program,
                                        "iView.clipRect.size"), 1,
                clip_rect.size)

        gl.glBindVertexArray(target.vao)

        # quad
        gl_rect_bounds = params.rect.bounds * 2 - 1
        state.vertices[QUAD_VERTS_BL][:] = gl_rect_bounds[0:2]  # min_x, min_y
        state.vertices[QUAD_VERTS_BR][:] = (
            gl_rect_bounds[2], gl_rect_bounds[1])  # max_x, min_y
        state.vertices[QUAD_VERTS_TR][:] = gl_rect_bounds[2:]  # max_x, max_y
        state.vertices[QUAD_VERTS_TL][:] = (
            gl_rect_bounds[0], gl_rect_bounds[3])  # min_x, max_y
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER,
                           offset=0,
                           data=state.vertices)

    def _update_region_sub_data(self, region_idx: int, region_data: bytes):
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, self.target.world_ssbo)
        gl.glBufferSubData(gl.GL_SHADER_STORAGE_BUFFER,
                           offset=region_idx * REGION_SIZE,
                           data=region_data)

    def _update_region_buffer(self,
                              target: RenderTarget):
        world_data_size = self._view.world.r_count * REGION_SIZE
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, target.world_ssbo)
        gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER,
                        world_data_size,
                        data=None,
                        usage=gl.GL_DYNAMIC_DRAW)
        gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER,
                            WORLD_SSBO_BINDING,
                            target.world_ssbo)

        asyncjob.submit(asyncjob.AsyncJobParameters(
            self._view.world.raw_regions(),
            self._view.world.r_count,
            self._update_region_sub_data,
            name="LoadWorld",
            batch_size=256))


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
