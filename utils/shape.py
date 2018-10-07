import numpy as np
from pyrr import rectangle


class Rect:  # pragma: no cover
    @staticmethod
    def from_bounds(x0: float = 0, y0: float = 0,
                    x1: float = 1, y1: float = 1):
        return Rect(x0, y0, x1 - x0, y1 - x0)

    def __init__(self, x=0., y=0., width=1., height=1.):
        self._m = rectangle.create(x=x, y=y, width=width, height=height,
                                   dtype=np.float32)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and \
               np.all(self._m == other._m)

    def __repr__(self):
        return f'{self.__class__}(x={self.x}, y={self.y}, ' \
               f'width={self.width}, height={self.height})'

    def __hash__(self):
        return hash(self._m)

    @property
    def data(self) -> np.ndarray:
        return self._m

    @property
    def x(self):
        return rectangle.x(self._m)

    @property
    def y(self):
        return rectangle.y(self._m)

    @property
    def size(self) -> np.ndarray:
        return rectangle.size(self._m)

    @property
    def height(self) -> float:
        return rectangle.height(self._m)

    @property
    def width(self) -> float:
        return rectangle.width(self._m)

    @property
    def bounds(self) -> np.ndarray:
        return np.concatenate((self.position, self.position + self.size))

    @property
    def aspect_ratio(self) -> float:
        return rectangle.aspect_ratio(self._m)

    @property
    def bottom(self) -> float:
        return rectangle.bottom(self._m)

    @property
    def top(self) -> float:
        return rectangle.top(self._m)

    @property
    def left(self) -> float:
        return rectangle.left(self._m)

    @property
    def right(self) -> float:
        return rectangle.right(self._m)

    @property
    def position(self) -> np.ndarray:
        return rectangle.position(self._m)
