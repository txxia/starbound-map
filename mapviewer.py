# -*- coding: utf-8 -*-
import logging
import os

if 'PYGLFW_LIBRARY' not in os.environ:
    os.environ['PYGLFW_LIBRARY'] = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'glfw3.dll')

import OpenGL.GL as gl
import numpy as np
import glfw
import imgui
from imgui.integrations.glfw import GlfwRenderer

from starbound import GameDirectory
from map import WorldRenderer, WorldView, REGION_DIM
from utils.config import CONFIG

CONFIG_SECTION = 'map_viewer'
CONFIG_GAME_ROOT = 'starbound_root'

POPUP_SETTINGS = 'Settings'
POPUP_SELECT_WORLD = 'Select World'


class G:
    """
    Global runtime-only data
    """
    gui_show_help_overlay = False
    gui_config_changed = False

    framebuffer_size = np.zeros(2)
    mouse_in_map_normal01 = np.zeros(2)


class WorldViewer(object):

    def __init__(self):
        if not CONFIG.has_section(CONFIG_SECTION):
            CONFIG.add_section(CONFIG_SECTION)
        self.config = CONFIG[CONFIG_SECTION]
        self.gamedata = GameDirectory()
        self.gamedata.game_root = self.config.get(CONFIG_GAME_ROOT)

        self.world_coord = None
        self.world = None
        self.player_start = np.zeros(2, dtype=np.int)
        self.world_size = np.zeros(2, dtype=np.int)
        self.world_size_in_regions = np.zeros(2, dtype=np.int)

        # TODO this implementation caps at 9x9 grid, may need to look into alternatives
        '''
        https://www.khronos.org/opengl/wiki/Array_Texture + gl.glTexSubImage1Dui
        https://www.reddit.com/r/opengl/comments/4u8qyv/opengl_limited_number_of_textures_how_can_you/
        '''
        self.grid_dim = 7
        self.view = None
        self.world_renderer = WorldRenderer(self.view, grid_dim=self.grid_dim)

        self.io = imgui.get_io()
        self.set_styles()

    def change_world(self, world_coord):
        try:
            self.world = self.gamedata.get_world(world_coord)
            self.world_coord = world_coord
            logging.info("Changed world to %s (%s)", world_coord, self.world)
        except Exception as e:
            logging.error(e)
            self.world = None
            self.world_coord = None
        if self.world is not None:
            self.player_start = np.array(self.world.metadata['playerStart'], dtype=np.int)
            self.world_size = np.array((self.world.width, self.world.height), dtype=np.int)
            self.world_size_in_regions = np.ceil(self.world_size / REGION_DIM).astype(np.int)

            logging.info('World size in regions: {}'.format(self.world_size_in_regions))
            center_region = np.floor(self.player_start / REGION_DIM).astype(np.int)
            self.world_renderer.view = self.view = WorldView(self.world,
                                                             center_region=center_region,
                                                             grid_dim=self.grid_dim)
        else:
            self.world_renderer.view = self.view = None

    def render(self, framebuffer_size):
        """
        :param framebuffer_size: tuple of 2 ints indicating framesize
        """
        G.framebuffer_size = framebuffer_size
        aspect = float(framebuffer_size[1]) / framebuffer_size[0]
        map_rect_normal = np.array((-1, 1 - 2 / max(aspect, 1), 1, 1))
        map_rect_normal01 = (map_rect_normal + 1) * 0.5
        map_rect = np.array((
            map_rect_normal01[0] * framebuffer_size[0],
            map_rect_normal01[1] * framebuffer_size[1],
            map_rect_normal01[2] * framebuffer_size[0],
            map_rect_normal01[3] * framebuffer_size[1]
        ))
        map_rect_size = map_rect[2:4] - map_rect[0:2]
        mouse = np.array((
            self.io.mouse_pos[0],
            framebuffer_size[1] - self.io.mouse_pos[1]
        ))
        G.mouse_in_map_normal01 = (mouse - map_rect[:2]) / map_rect_size

        imgui.new_frame()
        self.show_map_controller_window()
        self.show_tooltip()
        if G.gui_show_help_overlay:
            self.show_help_overlay()
        self.world_renderer.draw(map_rect_normal, framebuffer_size, glfw.get_time())

        # imgui.show_test_window()
        # self.show_debug_window()

        imgui.render()

    def show_map_controller_window(self):
        imgui.set_next_window_position(0, G.framebuffer_size[0])
        imgui.set_next_window_size(G.framebuffer_size[0], G.framebuffer_size[1] - G.framebuffer_size[0])

        if imgui.begin("Map", closable=False, flags=imgui.WINDOW_NO_RESIZE |
                                                    imgui.WINDOW_NO_MOVE |
                                                    imgui.WINDOW_NO_COLLAPSE |
                                                    imgui.WINDOW_NO_TITLE_BAR):
            if self.view is not None:
                center_x, center_y = self.view.center_region
                if imgui.button("<") or self.get_keys_on_map(glfw.KEY_A, glfw.KEY_LEFT):
                    center_x = max(center_x - 1, 1)
                imgui.same_line()
                if imgui.button(">") or self.get_keys_on_map(glfw.KEY_D, glfw.KEY_RIGHT):
                    center_x = min(center_x + 1, self.world_size_in_regions[0])
                imgui.same_line()
                _, center_x = imgui.slider_int("X", center_x,
                                               min_value=1,
                                               max_value=self.world_size_in_regions[0])
                if imgui.button("^") or self.get_keys_on_map(glfw.KEY_W, glfw.KEY_UP):
                    center_y = min(center_y + 1, self.world_size_in_regions[1])
                imgui.same_line()
                if imgui.button("v") or self.get_keys_on_map(glfw.KEY_S, glfw.KEY_DOWN):
                    center_y = max(center_y - 1, 1)
                imgui.same_line()
                _, center_y = imgui.slider_int("Y", center_y,
                                               min_value=1,
                                               max_value=self.world_size_in_regions[1])
                self.view.center_region = np.array((center_x, center_y))

                imgui.separator()

            _, self.world_renderer.config.showGrid = imgui.checkbox("Grid", self.world_renderer.config.showGrid)

            imgui.separator()
            if imgui.tree_node("World Info", flags=imgui.TREE_NODE_DEFAULT_OPEN):
                if self.world is not None:
                    imgui.label_text('Coordinates', self.world_coord)
                    imgui.label_text('Size', str(np.array((self.world.width, self.world.height))))
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
            _, G.gui_show_help_overlay = imgui.checkbox("Usage", G.gui_show_help_overlay)

            imgui.end()

    def show_tooltip(self):
        if self.world is None or self.view is None:
            return
        mouse = self.io.mouse_pos
        if 0 <= mouse.x <= G.framebuffer_size[0] and 0 <= mouse.y <= G.framebuffer_size[0]:
            region_coord, tile_coord = self.view.get_location(G.mouse_in_map_normal01)

            imgui.begin_tooltip()
            imgui.push_item_width(60)
            imgui.text(str(region_coord * REGION_DIM + tile_coord))
            imgui.label_text('region', str(region_coord))
            imgui.label_text('tile', str(tile_coord))
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
            imgui.bullet_text("Navigate with WASD/arrow-keys (click on the map first)")
            # TODO imgui.bullet_text("Right-click on a tile to see details")
            imgui.end()

    def popup_settings(self):
        if imgui.begin_popup_modal(POPUP_SETTINGS, flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)[0]:
            changed, self.config[CONFIG_GAME_ROOT] = imgui.input_text("Game Root",
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
                    self.change_world(world_coord)
            imgui.end_popup()

    def show_debug_window(self):
        imgui.label_text("time", '{:.1f}'.format(glfw.get_time()))
        imgui.label_text("fps", '{:.1f}'.format(self.io.framerate))
        imgui.label_text("mouse", '{:.1f}, {:.1f}'.format(self.io.mouse_pos.x, self.io.mouse_pos.y))
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

    logging.info("GLFW version: %s", glfw.get_version_string())

    if not glfw.init():
        logging.fatal("Could not initialize OpenGL context")
        exit(1)

    # OS X supports only forward-compatible core profiles from 3.2
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(
        int(width), int(height), window_name, None, None
    )
    glfw.make_context_current(window)

    if not window:
        glfw.terminate()
        logging.fatal("Could not initialize Window")
        exit(1)

    glfw.set_window_aspect_ratio(window, 2, 3)
    return window


def main():
    window = impl_glfw_init()
    impl = GlfwRenderer(window)
    viewer = WorldViewer()
    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()
        framebuffer_size = glfw.get_framebuffer_size(window)
        gl.glViewport(0, 0, framebuffer_size[0], framebuffer_size[1])
        viewer.render(framebuffer_size)
        glfw.swap_buffers(window)
    impl.shutdown()
    imgui.shutdown()
    glfw.terminate()


if __name__ == "__main__":
    logging.basicConfig(format='%(pathname)s:%(lineno)d:\n%(levelname)7s | %(message)s ', level=logging.DEBUG)
    main()
