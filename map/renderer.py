import dataclasses as dc
import typing as tp

import OpenGL.GL as gl
import numpy as np
from OpenGL.GL import shaders

from utils import asyncjob
from utils.resource import asset_path
from .controller import WorldViewController
from .model import REGION_SIZE, TileMaterialLayer

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

    render_fbo: tp.Any
    render_texture: tp.Any


@dc.dataclass
class RenderParameters:
    showGrid: bool = True
    canvas_size: np.ndarray = np.ones(2, dtype=np.int)
    """(min, max) representing region in viewport [0, 1]^2 to draw the map"""
    time_in_seconds: float = 0
    """time since the application started"""
    tile_selected: tp.Optional[np.ndarray] = None
    tile_mat_layer_mask: np.uint = TileMaterialLayer.FOREGROUND | TileMaterialLayer.BACKGROUND


def init_target() -> RenderTarget:  # pragma: no cover
    # Setting up VAO
    vao = gl.glGenVertexArrays(1)
    gl.glBindVertexArray(vao)

    # Quad vertex buffer
    vbo = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    gl.glBufferData(gl.GL_ARRAY_BUFFER,
                    QUAD_VERTS.nbytes,
                    QUAD_VERTS,
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

    # Create target texture
    render_texture = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, render_texture)
    gl.glTexImage2D(gl.GL_TEXTURE_2D,
                    0,  # level
                    gl.GL_RGBA,  # internalFormat
                    1,  # width
                    1,  # height
                    0,  # border
                    gl.GL_RGBA,  # format
                    gl.GL_FLOAT,  # type
                    None  # data
                    )
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)

    # Create target framebuffer
    # Create target framebuffer
    render_fbo = gl.glGenFramebuffers(1)
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, render_fbo)
    gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER,
                              gl.GL_COLOR_ATTACHMENT0,
                              gl.GL_TEXTURE_2D,
                              render_texture,
                              0)
    _validate_fbo_complete()
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

    return RenderTarget(
        vao=vao,
        vbo=vbo,
        ebo=ebo,
        world_ssbo=world_ssbo,
        vertex_shader=vs,
        fragment_shader=fs,
        program=program,
        indices=np.copy(QUAD_IDX),
        render_fbo=render_fbo,
        render_texture=render_texture,
    )


def __load_shaders():
    with open(asset_path('shader/map.vs.glsl'), 'r') as vs:
        vs_src = vs.read()
    with open(asset_path('shader/map.fs.glsl'), 'r') as fs:
        fs_src = fs.read()
    return vs_src, fs_src


class WorldRenderer:  # pragma: no cover
    def __init__(self):
        self._view: tp.Optional[WorldViewController] = None
        self.target = init_target()

    def change_view(self, value: tp.Optional[WorldViewController]):
        if self._view != value:
            self._view = value
            if value is not None:
                self._update_region_buffer(self.target)

    def draw(self, params: RenderParameters):
        """
        Draw a frame of the world view.
        """
        self._update_params(self.target, params)

        gl.glUseProgram(self.target.program)
        gl.glBindVertexArray(self.target.vao)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.target.ebo)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.target.render_fbo)
        gl.glDrawElements(gl.GL_TRIANGLES, self.target.indices.size,
                          gl.GL_UNSIGNED_SHORT, None)
        gl.glBindVertexArray(0)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

    def _update_params(self,
                       target: RenderTarget,
                       params: RenderParameters):
        assert params.canvas_size.dtype.kind == 'i'
        assert np.all(params.canvas_size >= 1)

        gl.glUseProgram(self.target.program)
        gl.glViewport(0, 0, params.canvas_size[0], params.canvas_size[1])

        # Resize framebuffer
        gl.glBindTexture(gl.GL_TEXTURE_2D, target.render_texture)
        gl.glTexImage2D(gl.GL_TEXTURE_2D,
                        0,  # level
                        gl.GL_RGBA,  # internalFormat
                        params.canvas_size[0],  # width
                        params.canvas_size[1],  # height
                        0,  # border
                        gl.GL_RGBA,  # format
                        gl.GL_FLOAT,  # type
                        None  # data
                        )
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, target.render_fbo)
        _validate_fbo_complete()
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

        # Uniforms
        # Compute fragment projection from window space to rect space [-1,1]^2
        gl.glUniform2fv(gl.glGetUniformLocation(target.program, "iResolution"), 1,
                        params.canvas_size)
        gl.glUniform1f(gl.glGetUniformLocation(target.program, "iTime"),
                       params.time_in_seconds)
        gl.glUniform1i(gl.glGetUniformLocation(target.program, "iConfig.showGrid"),
                       params.showGrid)
        gl.glUniform1ui(gl.glGetUniformLocation(target.program, "iMaterialLayers"),
                        params.tile_mat_layer_mask)

        if params.tile_selected is not None:
            gl.glUniform2iv(
                gl.glGetUniformLocation(target.program, "iTileSelected"), 1,
                params.tile_selected.astype(np.int32))

        if self._view:
            clip_rect = self._view.clip_rect()
            gl.glUniform2iv(
                gl.glGetUniformLocation(target.program, "iView.worldTSize"), 1,
                self._view.world.t_size.astype(np.int32))
            gl.glUniform2iv(
                gl.glGetUniformLocation(target.program, "iView.worldRSize"), 1,
                self._view.world.r_size.astype(np.int32))
            gl.glUniform2fv(
                gl.glGetUniformLocation(target.program,
                                        "iView.clipRect.position"), 1,
                clip_rect.position)
            gl.glUniform2fv(
                gl.glGetUniformLocation(target.program,
                                        "iView.clipRect.size"), 1,
                clip_rect.size)

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


def _validate_fbo_complete():
    fbo_status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
    assert fbo_status == gl.GL_FRAMEBUFFER_COMPLETE, f"0x{fbo_status:08X}"


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
