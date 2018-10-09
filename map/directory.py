import glob
import logging
import mmap
import os
import typing as tp

import starbound.data as sbdata
from map.model import World


class GameDirectory:
    WORLD_PATH = 'storage/universe/{coordinates}.world'

    def __init__(self):
        self._game_root = None
        self._world_list = []

    @property
    def game_root(self):
        return self._game_root

    @game_root.setter
    def game_root(self, value):
        if self._game_root != value:
            self._game_root = value
            self.synchronize()

    @property
    def world_list(self):
        return self._world_list

    def get_file(self, relpath: str) -> str:
        return os.path.join(self.game_root, relpath)

    def get_files(self, relpath: str) -> tp.Sequence[str]:
        return glob.glob(self.get_file(relpath))

    def get_world(self, coord: str) -> tp.Optional[World]:
        world_file = self.get_file(
            GameDirectory.WORLD_PATH.format(coordinates=coord))
        if not os.path.isfile(world_file):
            logging.warning("Failed to load world, not a file: %s", world_file)
            return None
        else:
            world_fd = open(world_file, 'rb')
            world_mm = mmap.mmap(world_fd.fileno(), 0, access=mmap.ACCESS_READ)
            world = World(sbdata.World(world_mm), coordinates=coord)
            logging.debug('Loaded world [%s] at %s', coord, world_file)
            return world

    def synchronize(self):
        if not self.valid():
            return
        self._sync_world_list()

    def valid(self) -> bool:
        return self.game_root is not None and os.path.isdir(self.game_root)

    def _sync_world_list(self):
        world_files = self.get_files(
            GameDirectory.WORLD_PATH.format(coordinates='*'))
        self._world_list[:] = (os.path.splitext(os.path.basename(f))[0] for f in
                               world_files)
        logging.debug("Updated world list, found %d worlds",
                      len(self.world_list))
