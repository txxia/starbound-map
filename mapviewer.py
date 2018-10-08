# -*- coding: utf-8 -*-
import asyncio
import logging
import os
import typing as tp

from utils.profiler import LineProfiler
from utils.shape import Rect

if 'PYGLFW_LIBRARY' not in os.environ:
    os.environ['PYGLFW_LIBRARY'] = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), 'glfw3.dll')

import OpenGL.GL as gl

import numpy as np
import glfw
import imgui
from imgui.integrations.glfw import GlfwRenderer

from map.directory import GameDirectory
from map.model import WorldView, Tile
from map.renderer import WorldRenderer, RenderParameters
from map.controller import WorldViewController
from utils import asyncjob
from utils.config import CONFIG

ZOOM_SPEED = 0.1
PAN_SPEED = 10

CONFIG_SECTION = 'map_viewer'
CONFIG_GAME_ROOT = 'starbound_root'

POPUP_JOB = 'Work In Progress'
POPUP_SETTINGS = 'Settings'
POPUP_SELECT_WORLD = 'Select World'


class G:
    """
    Global runtime-only data
    """
    minimized: bool = False

    gui_show_help_overlay: bool = False
    gui_config_changed: bool = False
    gui_show_tile_details: bool = True

    job: tp.Optional[asyncjob.AsyncJob] = None

    render_params = RenderParameters()

    mouse_in_map_normal01 = np.zeros(2)
    tile_selected: tp.Tuple[np.array, Tile] = None

    line_profiler: LineProfiler = None

    @staticmethod
    def on_minimization_event(is_minimized):
        G.minimized = is_minimized


class WorldViewer:

    def __init__(self):
        if not CONFIG.has_section(CONFIG_SECTION):
            CONFIG.add_section(CONFIG_SECTION)
        self.config = CONFIG[CONFIG_SECTION]
        self.gamedata = GameDirectory()
        self.gamedata.game_root = self.config.get(CONFIG_GAME_ROOT)

        self.world_coord = None
        self.world = None
        self.player_start = np.zeros(2, dtype=np.int)

        self.view: tp.Optional[WorldViewController] = None
        self.world_renderer = WorldRenderer()

        self.io = imgui.get_io()
        self.set_styles()

    async def change_world(self, world_coord):
        world = None
        try:
            world = self.gamedata.get_world(world_coord)
            logging.info("Loaded world")
        except Exception as e:
            logging.error(e)
        if world is not None:
            self.world = world
            self.world_coord = world_coord
            logging.info(f"Changed world to {world_coord} ({self.world})")

            self.player_start = np.array(self.world.metadata['playerStart'],
                                         dtype=np.int)
            logging.info(f'World size in regions: {self.world.r_size}')

            self.view = WorldViewController(WorldView(self.world))
            self.view.focus = self.player_start.astype(np.float)
            self.world_renderer.change_view(self.view)

    def draw_ui(self, frame_size: np.ndarray):
        imgui.new_frame()

        if self.view:
            self.view.canvas_size = frame_size * G.render_params.canvas_rect.size

        self.show_map_controller_window()

        if G.gui_show_help_overlay:
            self.show_help_overlay()

        job = asyncjob.current_job()
        if G.job != job:
            G.job = job
            if job:
                imgui.open_popup(POPUP_JOB)
        self.popup_job(job)

        if not job:
            self.show_tooltip()

        # imgui.show_test_window()
        # self.show_debug_window()

        imgui.render()

    def show_map_controller_window(self):
        imgui.set_next_window_position(0, G.render_params.frame_size[0])
        imgui.set_next_window_size(G.render_params.frame_size[0],
                                   G.render_params.frame_size[1] -
                                   G.render_params.frame_size[0])

        if not imgui.begin("Map",
                           closable=False,
                           flags=imgui.WINDOW_NO_RESIZE |
                                 imgui.WINDOW_NO_MOVE |
                                 imgui.WINDOW_NO_COLLAPSE |
                                 imgui.WINDOW_NO_TITLE_BAR):
            return
        G.tile_selected = None
        if self.view is not None:
            mouse_in_window = np.all(np.logical_and(G.mouse_in_map_normal01 >= 0,
                                                    G.mouse_in_map_normal01 <= 1))
            if not self.io.want_capture_mouse and mouse_in_window:
                tile_coord = self.view.trace(coord01=G.mouse_in_map_normal01)
                if self.view.world.is_valid_tile_coord(*tile_coord):
                    G.tile_selected = (
                        tile_coord,
                        self.view.world.get_tile(*tile_coord)
                    )
                    G.render_params.tile_selected = tile_coord

            # Zooming
            if not self.io.want_capture_mouse and self.io.mouse_wheel:
                zoom_pivot = G.tile_selected[0].astype(np.float) if G.tile_selected else None
                self.view.control_zoom(self.io.mouse_wheel * ZOOM_SPEED, zoom_pivot)
            if imgui.button("-"):
                self.view.control_zoom(-ZOOM_SPEED)
            imgui.same_line()
            if imgui.button("+"):
                self.view.control_zoom(+ZOOM_SPEED)
            imgui.same_line()
            zoom_slided, zoom_newval = imgui.slider_float(
                "Zoom",
                value=self.view.zoom,
                min_value=self.view.min_zoom,
                max_value=self.view.max_zoom,
                display_format='%.1f')
            if zoom_slided:
                self.view.zoom = zoom_newval

            # Panning
            focus_x, focus_y = self.view.focus
            focus_v = np.zeros(2)
            if imgui.button("<") or self.get_keys_on_map(glfw.KEY_A,
                                                         glfw.KEY_LEFT):
                focus_v[0] -= PAN_SPEED
            imgui.same_line()
            if imgui.button(">") or self.get_keys_on_map(glfw.KEY_D,
                                                         glfw.KEY_RIGHT):
                focus_v[0] += PAN_SPEED
            imgui.same_line()
            _, focus_x = imgui.slider_int("Focus.X", focus_x,
                                          min_value=self.view.min_focus[0],
                                          max_value=self.view.max_focus[0])
            if imgui.button("v") or self.get_keys_on_map(glfw.KEY_S,
                                                         glfw.KEY_DOWN):
                focus_v[1] -= PAN_SPEED
            imgui.same_line()
            if imgui.button("^") or self.get_keys_on_map(glfw.KEY_W,
                                                         glfw.KEY_UP):
                focus_v[1] += PAN_SPEED

            imgui.same_line()
            _, focus_y = imgui.slider_int("Focus.Y", focus_y,
                                          min_value=self.view.min_focus[1],
                                          max_value=self.view.max_focus[1])

            self.view.focus = np.array((focus_x, focus_y), dtype=np.float)
            self.view.control_focus(focus_v)
            imgui.separator()

        _, G.render_params.showGrid = imgui.checkbox("Grid",
                                                     G.render_params.showGrid)
        imgui.same_line()
        _, G.gui_show_tile_details = imgui.checkbox("Tile Details",
                                                    G.gui_show_tile_details)

        imgui.separator()
        if imgui.tree_node("World Info",
                           flags=imgui.TREE_NODE_DEFAULT_OPEN):
            if self.world is not None:
                imgui.label_text('Coordinates', self.world_coord)
                imgui.label_text('Size', str(
                    np.array((self.world.t_width, self.world.t_height))))
                imgui.label_text('PlayerStart', str(self.player_start))
            else:
                imgui.text('Select a world to start')
            imgui.tree_pop()

        imgui.separator()
        if imgui.button("Settings.."):
            imgui.open_popup(POPUP_SETTINGS)
        self.popup_settings()
        imgui.same_line()
        if imgui.button("Select World.."):
            imgui.open_popup(POPUP_SELECT_WORLD)
        self.popup_select_world()
        imgui.same_line()
        _, G.gui_show_help_overlay = imgui.checkbox("Usage",
                                                    G.gui_show_help_overlay)

        imgui.end()

    def show_tooltip(self):
        if self.world is None or self.view is None:
            return
        if G.tile_selected:
            coord, tile = G.tile_selected
            imgui.begin_tooltip()
            imgui.push_item_width(100)
            imgui.label_text('tile', str(coord))

            # tile details
            if G.gui_show_tile_details:
                imgui.label_text('fg.material', f"0x{tile.foreground_material & 0xffff:04X}")
                imgui.label_text('bg.material', f"0x{tile.background_material & 0xffff:04X}")
                imgui.label_text('liquid', f"{tile.liquid & 0xff:02X}")
                imgui.label_text('liquid.level', f"{tile.liquid_level:.2f}")
                imgui.label_text('liquid.pressure', f"{tile.liquid_pressure:.2f}")
                imgui.label_text('collision', f"{tile.collision}")
                imgui.label_text('dungeon_id', f"0x{tile.dungeon_id & 0xffff:04X}")
                imgui.label_text('biome', f"0x{tile.biome & 0xff:02X}")
                imgui.label_text('biome_2', f"0x{tile.biome_2 & 0xff:02X}")
                imgui.label_text('indestructible', f"{tile.indestructible}")

            imgui.pop_item_width()
            imgui.end_tooltip()

    def show_help_overlay(self):
        imgui.set_next_window_position(0, 0, imgui.ALWAYS)
        if imgui.begin("Help", closable=False,
                       flags=imgui.WINDOW_NO_MOVE |
                             imgui.WINDOW_NO_TITLE_BAR |
                             imgui.WINDOW_NO_RESIZE |
                             imgui.WINDOW_ALWAYS_AUTO_RESIZE |
                             imgui.WINDOW_NO_SAVED_SETTINGS |
                             imgui.WINDOW_NO_FOCUS_ON_APPEARING |
                             imgui.WINDOW_NO_INPUTS):
            imgui.text("Usage")
            imgui.separator()
            imgui.bullet_text(
                "Navigate with WASD/arrow-keys (click on the map first)")
            imgui.bullet_text("Zoom in/out with mouse wheel")
            # TODO imgui.bullet_text("Right-click on a tile to see details")
            imgui.end()

    def popup_job(self, job: asyncjob.AsyncJob):
        if imgui.begin_popup_modal(POPUP_JOB,
                                   flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE |
                                         imgui.WINDOW_NO_RESIZE |
                                         imgui.WINDOW_NO_MOVE)[0]:
            if job is None:
                logging.debug("Job no longer exist, closing popup")
                imgui.close_current_popup()
            else:
                imgui.label_text("Job Name", job.params.name or "N/A")
                imgui.separator()
                imgui.label_text("Progress", f"{job.progress.percentage:.0f}%")
            imgui.end_popup()

    def popup_settings(self):
        if imgui.begin_popup_modal(POPUP_SETTINGS,
                                   flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)[0]:
            changed, self.config[CONFIG_GAME_ROOT] = imgui.input_text(
                "Game Root",
                self.config.get(CONFIG_GAME_ROOT, ''),
                255)
            G.gui_config_changed |= changed

            imgui.separator()
            if imgui.button('OK'):
                if G.gui_config_changed:
                    G.gui_config_changed = False
                    logging.info('Detected config change, saving the file...')
                    CONFIG.save()
                    self.on_config_updated()
                imgui.close_current_popup()
            imgui.end_popup()

    def popup_select_world(self):
        if imgui.begin_popup(POPUP_SELECT_WORLD):
            for world_coord in self.gamedata.world_list:
                _, selected = imgui.selectable(world_coord)
                if selected:
                    logging.info(f"Changing world to {world_coord}")
                    asyncio.create_task(self.change_world(world_coord))
            imgui.end_popup()

    def show_debug_window(self):
        imgui.label_text("time", '{:.1f}'.format(glfw.get_time()))
        imgui.label_text("fps", '{:.1f}'.format(self.io.framerate))
        imgui.label_text("mouse", '{:.1f}, {:.1f}'.format(self.io.mouse_pos.x,
                                                          self.io.mouse_pos.y))
        imgui.label_text('mouse in map', str(G.mouse_in_map_normal01))

    def get_keys_on_map(self, *keys):
        return not imgui.is_window_focused() and \
               not self.io.want_capture_keyboard and \
               any(self.io.keys_down[k] for k in keys)

    def set_styles(self):
        imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, 0)

    def on_config_updated(self):
        self.gamedata.game_root = self.config[CONFIG_GAME_ROOT]


def impl_glfw_init():
    width, height = 400, 600
    window_name = "Starbound World Viewer"

    logging.info("GLFW ver: %s", glfw.get_version_string())

    if not glfw.init():
        logging.fatal("Could not initialize OpenGL context")
        exit(1)

    # OS X supports only forward-compatible core profiles from 3.2
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(
        int(width), int(height), window_name, None, None
    )
    glfw.make_context_current(window)
    glfw.set_window_iconify_callback(
        window, lambda _, minimized: G.on_minimization_event(minimized))

    if not window:
        glfw.terminate()
        logging.fatal("Could not initialize Window")
        exit(1)

    logging.info("OpenGL ver: %s", gl.glGetString(gl.GL_VERSION))
    logging.info("GLSL ver: %s", gl.glGetString(gl.GL_SHADING_LANGUAGE_VERSION))

    glfw.set_window_aspect_ratio(window, 2, 3)
    return window


async def async_main(window):
    await asyncjob.start_worker()

    logging.info(f"Running loop: {asyncio.get_running_loop()}")

    impl = GlfwRenderer(window)
    imgui_io = imgui.get_io()

    viewer = WorldViewer()

    while not glfw.window_should_close(window):
        glfw.poll_events()
        if G.minimized:  # do not render zero sized frame
            continue

        impl.process_inputs()
        frame_size = np.array(glfw.get_framebuffer_size(window))
        aspect = float(frame_size[1]) / frame_size[0]

        map_rect_normal01 = np.array(((0, 1 - 1 / max(aspect, 1)), (1, 1)))
        # = (map_rect_normal + 1) * 0.5
        map_rect_normal_size = map_rect_normal01[1] - map_rect_normal01[0]
        map_rect = map_rect_normal01 * frame_size
        map_rect_size = map_rect[1] - map_rect[0]
        mouse = np.array((
            imgui_io.mouse_pos[0],
            frame_size[1] - imgui_io.mouse_pos[1]
        ))
        G.mouse_in_map_normal01 = (mouse - map_rect[0]) / map_rect_size

        G.render_params.frame_size = frame_size
        G.render_params.canvas_rect = Rect(x=map_rect_normal01[0][0],
                                           y=map_rect_normal01[0][1],
                                           width=map_rect_normal_size[0],
                                           height=map_rect_normal_size[1])
        G.render_params.time_in_seconds = glfw.get_time()

        gl.glViewport(0, 0, frame_size[0], frame_size[1])

        gl.glClearColor(0, 0, 0, 1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        viewer.world_renderer.draw(G.render_params)
        viewer.draw_ui(frame_size)
        glfw.swap_buffers(window)

        await asyncio.sleep(0)

    impl.shutdown()
    imgui.shutdown()


def main():
    window = impl_glfw_init()
    asyncio.run(async_main(window))
    glfw.terminate()


if __name__ == "__main__":
    logging.basicConfig(
        format='%(pathname)s:%(lineno)d:\n%(asctime)-15s | %(levelname)7s | %(message)s ',
        level=logging.DEBUG)
    # from utils.profiler import MemoryProfiler
    # with MemoryProfiler():
    main()
