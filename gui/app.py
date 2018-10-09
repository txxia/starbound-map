import glfw

from gui.core import *
from gui.widgets.job_overlay import JobOverlay
from gui.windows.about import AboutWindow
from gui.windows.mapcontrol import MapControllerWindow
from gui.windows.mapview import MapWindow
from gui.windows.usage import UsageWindow
from map.renderer import WorldRenderer


class StarboundMap(GUIElement):

    def __init__(self, world_renderer: WorldRenderer):
        state = GUIState(root=self)
        super().__init__(state)

        self.set_styles()

        self.add_child(MapWindow(self.state, renderer=world_renderer))
        self.add_child(MapControllerWindow(self.state))
        self.add_child(UsageWindow(self.state))
        self.add_child(AboutWindow(self.state))
        self.add_child(JobOverlay(self.state))

    def gui(self):
        imgui.new_frame()
        self.show_menu_bar()
        self.state.render_params.time_in_seconds = glfw.get_time()  # TODO remove glfw calls

        for child in self.children:
            child.gui()

        # imgui.show_test_window()
        # self.show_debug_window()
        imgui.render()

    def show_menu_bar(self):
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("Help"):
                if imgui.menu_item("User Guide")[0]:
                    self.send_event(GUIEventType.OPEN_WINDOW, arg=WindowName.USER_GUIDE)
                if imgui.menu_item("About")[0]:
                    self.send_event(GUIEventType.OPEN_WINDOW, arg=WindowName.ABOUT)
                imgui.end_menu()
            imgui.end_main_menu_bar()

    def show_debug_window(self):
        imgui.label_text("time", '{:.1f}'.format(glfw.get_time()))
        imgui.label_text("fps", '{:.1f}'.format(self.io.framerate))
        imgui.label_text("mouse", '{:.1f}, {:.1f}'.format(self.io.mouse_pos.x,
                                                          self.io.mouse_pos.y))

    def set_styles(self):
        pass
