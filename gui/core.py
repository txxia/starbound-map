from __future__ import annotations

from .common import *


# ImGUI IO reference:
# https://github.com/ocornut/imgui/blob/e623be998d008abed89f260010469c4f6210bb20/imgui.h#L1085


class GUIElement(GUIBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.children: tp.List[GUIElement] = list()

    def add_child(self, child: GUIElement):
        self.children.append(child)

    def send_event(self, event: GUIEventType,
                   scope: GUIEventScope = GUIEventScope.ROOT,
                   arg: tp.Any = None):
        targets = None
        if scope == GUIEventScope.ROOT:
            targets = (self.state.root,)
        elif scope == GUIEventScope.SELF:
            targets = (self,)
        elif scope == GUIEventScope.CHILDREN:
            targets = self.children

        for target in targets:
            target._receive_event(event, arg)

    @abc.abstractmethod
    def gui(self):
        pass

    def handle_event(self, event: GUIEventType, arg: tp.Any = None):
        pass

    def _receive_event(self, event: GUIEventType, arg: tp.Any):
        self.handle_event(event, arg)
        for child in self.children:
            child._receive_event(event, arg)


class GUIWindow(GUIElement):
    @abc.abstractmethod
    def begin_gui(self) -> bool:
        return False

    def end_gui(self):
        imgui.end()

    def gui(self):
        if self.begin_gui():
            for child in self.children:
                child.gui()
            self.end_gui()
