from gui.core import *


class AboutWindow(GUIWindow):
    FIELD_COLOR = (0.7, 0.7, 0.7)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.opened = False

    def handle_event(self, event: GUIEventType, arg: WindowName = None):
        if event == GUIEventType.OPEN_WINDOW and arg == WindowName.ABOUT:
            self.opened = True
            self.gui()

    def begin_gui(self) -> bool:
        if self.opened:
            self.begin_window()
            return True
        return False

    def begin_window(self):
        imgui.set_next_window_size(300, 150)
        imgui.set_next_window_centered(imgui.ALWAYS)
        _, self.opened = imgui.begin(WindowName.ABOUT.value, closable=True,
                                     flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE |
                                           imgui.WINDOW_NO_COLLAPSE |
                                           imgui.WINDOW_NO_SCROLLBAR)
        if self.opened:
            imgui.text("Starbound Map")
            imgui.columns(2, border=True)
            imgui.set_column_offset(1, 60)
            imgui.separator()
            imgui.text("Author")
            imgui.next_column()
            imgui.text_colored("txxia", *self.FIELD_COLOR)
            imgui.next_column()

            imgui.text("Source")
            imgui.next_column()
            imgui.text_colored("github.com/txxia/starbound-map", *self.FIELD_COLOR)

            imgui.columns(1)
            imgui.separator()
