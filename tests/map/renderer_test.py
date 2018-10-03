from unittest import TestCase

import numpy as np
import numpy.testing as npt

from map import renderer


class TestRenderer(TestCase):
    def setUp(self):
        pass

    def test_project_rect__scaling_only(self):
        npt.assert_almost_equal(
            np.array([[0.1, 0., 0.],
                      [0., 0.05, 0.],
                      [0., 0., 1.]]),
            renderer._project_rect(np.array([[0, 0], [10, 20]]))
        )

    def test_project_rect__translation_only(self):
        npt.assert_almost_equal(
            np.array([[1., 0., -3.],
                      [0., 1., -4.],
                      [0., 0., 1.]]),
            renderer._project_rect(np.array([[3, 4], [4, 5]]))
        )

    def test_project_rect__translation_and_scaling(self):
        proj = renderer._project_rect(np.array([[0, 6], [10, 16]]))
        npt.assert_almost_equal(
            np.array([[0.1, 0., 0.],
                      [0., 0.1, -0.6],
                      [0., 0., 1.]]),
            proj
        )
        # projecting center of the rect
        npt.assert_almost_equal(
            np.array([0.5, 0.5, 1.]),
            np.matmul(proj, [5, 11, 1])
        )
