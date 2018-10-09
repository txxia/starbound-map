from __future__ import annotations

import abc
import dataclasses as dc
import enum
import typing as tp

import imgui
import numpy as np

from map.controller import WorldViewController
from map.model import Tile
from map.renderer import RenderParameters
from utils.asyncjob import AsyncJob
from utils.config import CONFIG


class AutoNameEnum(enum.Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name


class GUIEventType(AutoNameEnum):
    CONFIG_UPDATED = enum.auto()
    WORLD_CHANGED = enum.auto()

    OPEN_WINDOW = enum.auto()


class GUIEventScope(AutoNameEnum):
    ROOT = enum.auto()
    SELF = enum.auto()
    CHILDREN = enum.auto()


class WindowName(enum.Enum):
    USER_GUIDE = "User Guide"
    ABOUT = "About"

    POPUP_JOB = "Work In Progress"


@dc.dataclass
class GUIState:
    """
    GUI run-time state.
    """
    root: GUIBase
    view: WorldViewController = None

    config_changed: bool = False
    current_job: tp.Optional[AsyncJob] = None

    render_params: RenderParameters = RenderParameters()
    tile_selected: tp.Tuple[np.array, Tile] = None
    show_tile_details: bool = True  # TODO remember this


class GUIBase(abc.ABC):
    CONFIG_SECTION = 'map_viewer'

    def __init__(self, state: GUIState):
        if not CONFIG.has_section(self.CONFIG_SECTION):
            CONFIG.add_section(self.CONFIG_SECTION)
        self.config = CONFIG[self.CONFIG_SECTION]
        self.io = imgui.get_io()
        self.state: GUIState = state
