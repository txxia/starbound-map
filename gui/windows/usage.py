from gui.core import *


class UsageWindow(GUIWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.opened = False

    def handle_event(self, event: GUIEventType, arg: WindowName = None):
        if event == GUIEventType.OPEN_WINDOW and arg == WindowName.USER_GUIDE:
            self.opened = True
            self.gui()

    def begin_gui(self) -> bool:
        if self.opened:
            self.begin_window()
            return True
        return False

    def begin_window(self):
        imgui.set_next_window_position(30, 30, imgui.FIRST_USE_EVER)
        _, self.opened = imgui.begin(
            WindowName.USER_GUIDE.value, closable=True,
            flags=imgui.WINDOW_NO_RESIZE |
                  imgui.WINDOW_ALWAYS_AUTO_RESIZE |
                  imgui.WINDOW_NO_SAVED_SETTINGS |
                  imgui.WINDOW_NO_FOCUS_ON_APPEARING)
        if self.opened:
            imgui.bullet_text("Navigate with WASD/arrow-keys")
            imgui.bullet_text("Zoom in/out with mouse wheel")
            # TODO imgui.bullet_text("Right-click on a tile to see details")
