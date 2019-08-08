import logging

from gui.core import *
from map.directory import GameDirectory
from map.model import TileMaterialLayer, WorldView


class MapControllerWindow(GUIWindow):
    CONFIG_GAME_ROOT = 'starbound_root'
    POPUP_SETTINGS = 'Settings'
    POPUP_SELECT_WORLD = 'Select World'

    ZOOM_SPEED = 0.1
    PAN_SPEED = 10

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.gamedata = GameDirectory()
        self.gamedata.game_root = self.config.get(self.CONFIG_GAME_ROOT)

    def handle_event(self, event: GUIEventType, arg=None):
        if event == GUIEventType.CONFIG_UPDATED:
            self.gamedata.game_root = self.config[self.CONFIG_GAME_ROOT]

    def begin_gui(self) -> bool:
        imgui.set_next_window_position(400, 100, imgui.FIRST_USE_EVER)
        imgui.set_next_window_size(400, 400, imgui.FIRST_USE_EVER)
        if not imgui.begin("Map Info", closable=False):
            return False
        if self.state.view is not None:

            # Zooming
            if imgui.button("-"):
                self.state.view.control_zoom(-self.ZOOM_SPEED)
            imgui.same_line()
            if imgui.button("+"):
                self.state.view.control_zoom(+self.ZOOM_SPEED)
            imgui.same_line()
            zoom_slided, zoom_newval = imgui.slider_float(
                "Zoom",
                value=self.state.view.zoom,
                min_value=self.state.view.min_zoom,
                max_value=self.state.view.max_zoom,
                display_format='%.1f')
            if zoom_slided:
                self.state.view.zoom = zoom_newval

            # Panning
            focus_x, focus_y = self.state.view.focus
            focus_v = np.zeros(2)
            if imgui.button("<"):
                focus_v[0] -= self.PAN_SPEED
            imgui.same_line()
            if imgui.button(">"):
                focus_v[0] += self.PAN_SPEED
            imgui.same_line()
            _, focus_x = imgui.slider_int("Focus.X", focus_x,
                                          min_value=self.state.view.min_focus[0],
                                          max_value=self.state.view.max_focus[0])
            if imgui.button("v"):
                focus_v[1] -= self.PAN_SPEED
            imgui.same_line()
            if imgui.button("^"):
                focus_v[1] += self.PAN_SPEED
            imgui.same_line()
            _, focus_y = imgui.slider_int("Focus.Y", focus_y,
                                          min_value=self.state.view.min_focus[1],
                                          max_value=self.state.view.max_focus[1])

            self.state.view.focus = np.array((focus_x, focus_y), dtype=np.float)
            self.state.view.control_focus(focus_v)
            imgui.separator()

        if imgui.collapsing_header("Material")[0]:
            _, self.state.render_params.tile_mat_layer_mask = imgui.checkbox_flags(
                "Foreground",
                self.state.render_params.tile_mat_layer_mask,
                TileMaterialLayer.FOREGROUND)
            _, self.state.render_params.tile_mat_layer_mask = imgui.checkbox_flags(
                "Background",
                self.state.render_params.tile_mat_layer_mask,
                TileMaterialLayer.BACKGROUND
            )

        if imgui.collapsing_header("Misc")[0]:
            _, self.state.render_params.showGrid = imgui.checkbox(
                "Grid",
                self.state.render_params.showGrid)
            imgui.same_line()
            _, self.state.show_tile_details = imgui.checkbox(
                "Tile Details",
                self.state.show_tile_details)

        imgui.separator()
        if imgui.tree_node("World", flags=imgui.TREE_NODE_DEFAULT_OPEN):
            if self.state.view is not None:
                imgui.label_text('Coordinates', self.state.view.world.coordinates)
                imgui.label_text('Size', str(
                    np.array((self.state.view.world.t_width, self.state.view.world.t_height))))
            else:
                imgui.text('Select a world to start')
            imgui.tree_pop()

        imgui.separator()
        if imgui.button("Settings.."):
            imgui.open_popup(self.POPUP_SETTINGS)
        self.popup_settings()
        imgui.same_line()
        if imgui.button("Select World.."):
            imgui.open_popup(self.POPUP_SELECT_WORLD)
        self.popup_select_world()
        return True

    def change_world(self, world_coord):
        world = None
        try:
            world = self.gamedata.get_world(world_coord)
            logging.info("Loaded world")
        except Exception as e:
            logging.error(e)
        if world is not None:
            logging.info(f"Changed world to {world_coord} ({world})")

            self.state.view = WorldViewController(WorldView(world))
            self.state.view.focus = np.array(world.metadata['playerStart'],
                                             dtype=np.float)
            self.send_event(GUIEventType.WORLD_CHANGED)

    def popup_settings(self):
        if imgui.begin_popup_modal(self.POPUP_SETTINGS,
                                   flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)[0]:
            changed, self.config[self.CONFIG_GAME_ROOT] = imgui.input_text(
                "Game Root",
                self.config.get(self.CONFIG_GAME_ROOT, ''),
                255)
            self.state.config_changed |= changed

            imgui.separator()
            if imgui.button('OK'):
                if self.state.config_changed:
                    self.state.config_changed = False
                    logging.info('Detected config change, saving the file...')
                    CONFIG.save()
                    self.send_event(GUIEventType.CONFIG_UPDATED)
                imgui.close_current_popup()
            imgui.end_popup()

    def popup_select_world(self):
        if imgui.begin_popup(self.POPUP_SELECT_WORLD):
            for world_coord in self.gamedata.world_list:
                _, selected = imgui.selectable(world_coord)
                if selected:
                    logging.info(f"Changing world to {world_coord}")
                    self.change_world(world_coord)
            imgui.end_popup()
