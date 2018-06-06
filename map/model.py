import numpy as np

from utils import cache

REGION_DIM = 32
TILES_PER_REGION = REGION_DIM * REGION_DIM
TILE_SIZE = 31
REGION_BYTES = TILES_PER_REGION * TILE_SIZE


class WorldView:
    """
    Represents a view to the world map.
    """

    def __init__(self, world, center_region, grid_dim=5):
        """
        :param world: world object
        :param center_region: center tile coordinate of the view, as tuple
        :param grid_dim: number of cells on any side of the grid
        """
        self._on_region_updated = []

        assert world is not None
        self._world = world

        self._grid_dim = None
        self.grid_dim = grid_dim

        self._region_grid = [[None for _ in range(grid_dim)] for _ in range(grid_dim)]

        self._center_region = np.zeros(2)
        self.center_region = center_region

    @property
    def world(self):
        return self._world

    @property
    def region_grid(self):
        return self._region_grid

    @property
    def center_region(self):
        return self._center_region

    @center_region.setter
    def center_region(self, value):
        assert isinstance(value, np.ndarray)
        assert value.size == 2
        assert value.dtype == np.int
        if any(self._center_region != value):
            self._center_region = value
            self._update_regions()

    @property
    def grid_dim(self):
        return self._grid_dim

    @grid_dim.setter
    def grid_dim(self, value):
        assert type(value) == int
        assert 0 < value
        if self._grid_dim != value:
            self._grid_dim = value

    def on_region_updated(self, callback):
        self._on_region_updated.append(callback)

    def get_location(self, coord01):
        """
        Find the location indicated by the given coordinate in the current view
        :param coord01: coordinate in the current view
        :return: (region_coord, tile_coord)
        """
        region_coord = self.center_region - self.grid_dim // 2 + (coord01 * self.grid_dim).astype(np.int)
        tile_coord = (coord01 % (1 / self.grid_dim) * self.grid_dim * REGION_DIM).astype(np.int)
        return region_coord, tile_coord

    @cache.memoized_method(maxsize=1024)
    def get_region(self, region_coord):
        """
        :param region_coord: region coordinate
        :return: regions at the given location, None if not available
        """
        try:
            return self.world.get_raw_tiles(region_coord[0], region_coord[1])
        except KeyError or RuntimeError:
            return None

    def _update_regions(self):
        """
        Updates the grid of size grid_dim^2 containing regions centered at the current view.
        """
        # FIXME handle wrapping around
        # Retrieve visible regions as a grid, each item is a byte array of 1024 tiles (31 bytes per tile)
        for y in range(self.grid_dim):
            ry = self.center_region[1] + y - 1
            for x in range(self.grid_dim):
                rx = self.center_region[0] + x - 1
                self.region_grid[y][x] = self.get_region((rx, ry))

        for cb in self._on_region_updated:
            cb()
