import glfw

from gui.core import *
from map.renderer import WorldRenderer


class MapWindow(GUIWindow):
    WINDOW_NAME = "Map View"
    WINDOW_SIZE_ADJUSTMENT = np.array((-16, -36))
    BORDER_COLOR = (0.5, 0.5, 0.5, 1.0)

    ZOOM_SPEED = 0.1
    PAN_SPEED = 10

    def __init__(self, *args, renderer: WorldRenderer = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.renderer = renderer

    def handle_event(self, event: GUIEventType, arg=None):
        if event == GUIEventType.WORLD_CHANGED:
            self.renderer.change_view(self.state.view)

    def begin_gui(self) -> bool:
        if not imgui.begin(self.WINDOW_NAME, closable=False,
                           flags=imgui.WINDOW_NO_SCROLLBAR):
            return False

        window_size = np.array(imgui.get_window_size(), dtype=np.int)
        if self.state.view:
            self.state.view.canvas_size = window_size

        if np.all(window_size >= 1):
            # TODO When imgui==1.0.0, use get_cursor_pos()
            canvas_size = window_size + self.WINDOW_SIZE_ADJUSTMENT
            if np.all(canvas_size >= 1):
                self.state.render_params.canvas_size = canvas_size
                self.renderer.draw(self.state.render_params)
                imgui.image(self.renderer.target.render_texture,
                            width=canvas_size[0],
                            height=canvas_size[1],
                            uv0=(0, 1),
                            uv1=(1, 0),
                            border_color=self.BORDER_COLOR)
        if self.state.view:
            if imgui.is_item_hovered():
                image_rect = np.array((
                    imgui.get_item_rect_min(),
                    imgui.get_item_rect_size()
                ))
                # Find the selected tile
                self.state.tile_selected = None
                tile_coord = self._get_selected_tile_coord(image_rect)
                if tile_coord is not None:
                    self.state.tile_selected = (
                        tile_coord,
                        self.state.view.world.get_tile(*tile_coord)
                    )
                    self.state.render_params.tile_selected = tile_coord
                self._show_tooltip()

                # Zooming
                if self.io.mouse_wheel:
                    zoom_pivot = self.state.tile_selected[0].astype(np.float) \
                        if self.state.tile_selected \
                        else None
                    self.state.view.control_zoom(self.io.mouse_wheel * self.ZOOM_SPEED, zoom_pivot)

            # Panning
            if imgui.is_item_hovered() or imgui.is_window_focused():
                focus_v = np.zeros(2)
                if self.get_keys(glfw.KEY_A, glfw.KEY_LEFT):
                    focus_v[0] -= self.PAN_SPEED
                if self.get_keys(glfw.KEY_D, glfw.KEY_RIGHT):
                    focus_v[0] += self.PAN_SPEED
                if self.get_keys(glfw.KEY_S, glfw.KEY_DOWN):
                    focus_v[1] -= self.PAN_SPEED
                if self.get_keys(glfw.KEY_W, glfw.KEY_UP):
                    focus_v[1] += self.PAN_SPEED
                self.state.view.control_focus(focus_v)
        return True

    def _get_selected_tile_coord(self, canvas_rect: np.ndarray) -> \
            tp.Optional[np.ndarray]:
        if not self.state.view:
            return None
        mouse_pos = np.array(self.io.mouse_pos, dtype=np.float)
        mouse_in_map01 = (mouse_pos - canvas_rect[0]) / canvas_rect[1]
        mouse_in_map01[1] = 1.0 - mouse_in_map01[1]

        if imgui.is_mouse_hovering_window():
            tile_coord = self.state.view.trace(coord01=mouse_in_map01)
            if self.state.view.world.is_valid_tile_coord(*tile_coord):
                return tile_coord
        return None

    def get_keys(self, *keys):
        return any(self.io.keys_down[k] for k in keys)

    def _show_tooltip(self):
        if self.state.view is None:
            return
        if self.state.tile_selected:
            coord, tile = self.state.tile_selected
            imgui.begin_tooltip()
            imgui.push_item_width(100)
            imgui.label_text('tile', str(coord))

            # tile details
            if self.state.show_tile_details:
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
