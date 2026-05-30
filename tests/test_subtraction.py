from __future__ import annotations

import unittest

import numpy as np

from scarlet.reduction import subtract_scattering_references


class TestSubtraction(unittest.TestCase):
    def test_subtract_scattering_references_applies_sample_and_empty_cell_formula(self) -> None:
        image = np.full((2, 2), 100.0, dtype=np.float64)
        dark = np.full((2, 2), 10.0, dtype=np.float64)
        empty_beam = np.full((2, 2), 30.0, dtype=np.float64)
        empty_cell = np.full((2, 2), 50.0, dtype=np.float64)

        corrected = subtract_scattering_references(
            image,
            0.5,
            dark=dark,
            empty_beam=empty_beam,
            empty_beam_transmission=1.0,
            empty_cell=empty_cell,
            empty_cell_transmission=0.8,
            distance=4.0,
            beam_center=(128.0, 128.0),
        )

        expected = (image - dark) / 0.5 - (empty_cell - dark) / 0.8
        np.testing.assert_allclose(corrected, expected)

    def test_subtract_scattering_references_requires_reference_transmission(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty_cell_transmission"):
            subtract_scattering_references(
                np.ones((2, 2), dtype=np.float64),
                0.5,
                empty_cell=np.ones((2, 2), dtype=np.float64),
            )

    def test_subtract_scattering_references_without_empty_cell_only_normalizes_sample(self) -> None:
        image = np.full((2, 2), 100.0, dtype=np.float64)
        dark = np.full((2, 2), 10.0, dtype=np.float64)

        corrected = subtract_scattering_references(image, 0.5, dark=dark)

        np.testing.assert_allclose(corrected, (image - dark) / 0.5)


if __name__ == "__main__":
    unittest.main()
