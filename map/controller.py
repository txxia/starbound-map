import math

import numpy as np

from utils.shape import Rect
from .model import World, WorldView, REGION_DIM


def clamp(value, min_value, max_value):
    return min(max(min_value, value), max_value)


class WorldViewController:

    def __init__(self, m: WorldView):
        self._m = m
        self._frame_size = np.ones(2)

    @property
    def canvas_size(self) -> np.ndarray:
        return self._frame_size

    @canvas_size.setter
    def canvas_size(self, value: np.ndarray):
        assert value.size == 2
        assert np.all(value > 0)
        self._frame_size = value

    @property
    def world(self) -> World:
        return self._m.world

    @property
    def focus(self) -> np.ndarray:
        return self._m.focus

    @focus.setter
    def focus(self, value: np.ndarray):
        value = np.clip(value, np.zeros(2), self.world.t_size - 1)
        self._m.focus = value

    @property
    def min_focus(self) -> np.ndarray:
        return np.zeros(2, dtype=np.float)

    @property
    def max_focus(self) -> np.ndarray:
        return self.world.t_size - 1

    @property
    def zoom(self) -> float:
        return self._m.zoom

    @zoom.setter
    def zoom(self, value: float):
        self._m.zoom = value

    @property
    def max_zoom(self) -> float:
        """
        Zoom upperbound.
        This value allows at least one region to be displayed.
        :return: suggested max zoom
        """
        return math.log(np.min(self.canvas_size) / REGION_DIM)

    @property
    def min_zoom(self) -> float:
        """
        Zoom lowerbound.
        This value allows the entire map to be displayed.
        :return: suggested min zoom
        """
        return math.log(np.min(self.canvas_size) / np.max(self._m.world.t_size))

    def control_focus(self, delta: np.ndarray):
        self._m.focus = np.clip(self.focus + delta, np.zeros(2), self.world.t_size)

    def control_zoom(self, delta: float, pivot: np.ndarray = None):
        prev_zoom = self.zoom
        new_zoom = clamp(self.zoom + delta, self.min_zoom, self.max_zoom)
        if pivot is not None:
            exp_zoom_diff = math.exp(prev_zoom - new_zoom) - 1
            self.focus += (self.focus - pivot) * exp_zoom_diff
        self._m.zoom = new_zoom

    def trace(self, coord01: np.ndarray) -> np.ndarray:
        """
        Find the location indicated by the given coordinates in this view.
        Note that the coordinate might not be valid.
        :param coord01: coordinate in the current view
        :return: tile coordinates
        """
        rect = self.clip_rect()
        return (rect.position + coord01 * rect.size).astype(np.int)

    def clip_rect(self) -> Rect:
        """
        Tile-level clipping rectangle of this view.
        Note that the vertices of this rect is not necessarily inside the map.
        :param canvas_size: size of the canvas to draw this view
        """
        rect_size = self.canvas_size / math.exp(self.zoom)
        position = self.focus - rect_size / 2
        return Rect(position[0], position[1], rect_size[0], rect_size[1])
