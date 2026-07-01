from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from scarlet.reduction import subtract_scattering_references


@unittest.skipIf(importlib.util.find_spec("scipp") is None, "scipp is required for subtraction tests")
class TestSubtraction(unittest.TestCase):
    def test_subtract_scattering_references_propagates_variances_with_dataarrays(self) -> None:
        import scipp as sc

        sample = sc.DataArray(
            data=sc.array(
                dims=["y", "x"],
                values=np.full((2, 2), 100.0, dtype=np.float64),
                variances=np.full((2, 2), 100.0, dtype=np.float64),
            )
        )
        dark = sc.DataArray(
            data=sc.array(
                dims=["y", "x"],
                values=np.full((2, 2), 10.0, dtype=np.float64),
                variances=np.full((2, 2), 10.0, dtype=np.float64),
            )
        )
        empty_cell = sc.DataArray(
            data=sc.array(
                dims=["y", "x"],
                values=np.full((2, 2), 50.0, dtype=np.float64),
                variances=np.full((2, 2), 50.0, dtype=np.float64),
            )
        )
        transmission = sc.DataArray(data=sc.scalar(0.5, variance=0.01))
        empty_cell_transmission = sc.DataArray(data=sc.scalar(0.8, variance=0.02))

        corrected = subtract_scattering_references(
            sample,
            transmission,
            dark=dark,
            empty_cell=empty_cell,
            empty_cell_transmission=empty_cell_transmission,
        )

        expected = (sample - dark) / sc.scalar(0.5) - (empty_cell - dark) / sc.scalar(0.8)
        np.testing.assert_allclose(corrected.data.values, expected.data.values)
        np.testing.assert_allclose(corrected.data.variances, expected.data.variances)

    def test_subtract_scattering_references_applies_sample_and_empty_cell_formula(self) -> None:
        import scipp as sc

        sample = sc.DataArray(data=sc.array(dims=["y", "x"], values=np.full((2, 2), 100.0, dtype=np.float64)))
        dark = sc.DataArray(data=sc.array(dims=["y", "x"], values=np.full((2, 2), 10.0, dtype=np.float64)))
        empty_beam = sc.DataArray(data=sc.array(dims=["y", "x"], values=np.full((2, 2), 30.0, dtype=np.float64)))
        empty_cell = sc.DataArray(data=sc.array(dims=["y", "x"], values=np.full((2, 2), 50.0, dtype=np.float64)))
        transmission = sc.DataArray(data=sc.scalar(0.5))
        empty_cell_transmission = sc.DataArray(data=sc.scalar(0.8))
        empty_beam_transmission = sc.DataArray(data=sc.scalar(1.0))

        corrected = subtract_scattering_references(
            sample,
            transmission,
            dark=dark,
            empty_beam=empty_beam,
            empty_beam_transmission=empty_beam_transmission,
            empty_cell=empty_cell,
            empty_cell_transmission=empty_cell_transmission,
            distance=4.0,
            beam_center=(128.0, 128.0),
        )

        expected = (sample - dark) / transmission.data - (empty_cell - dark) / empty_cell_transmission.data
        np.testing.assert_allclose(corrected.data.values, expected.data.values)

    def test_subtract_scattering_references_requires_reference_transmission(self) -> None:
        import scipp as sc

        with self.assertRaisesRegex(ValueError, "empty_cell_transmission"):
            subtract_scattering_references(
                sc.DataArray(data=sc.array(dims=["y", "x"], values=np.ones((2, 2), dtype=np.float64))),
                sc.DataArray(data=sc.scalar(0.5)),
                empty_cell=sc.DataArray(data=sc.array(dims=["y", "x"], values=np.ones((2, 2), dtype=np.float64))),
            )

    def test_subtract_scattering_references_without_empty_cell_only_normalizes_sample(self) -> None:
        import scipp as sc

        sample = sc.DataArray(data=sc.array(dims=["y", "x"], values=np.full((2, 2), 100.0, dtype=np.float64)))
        dark = sc.DataArray(data=sc.array(dims=["y", "x"], values=np.full((2, 2), 10.0, dtype=np.float64)))
        transmission = sc.DataArray(data=sc.scalar(0.5))

        corrected = subtract_scattering_references(sample, transmission, dark=dark)

        expected = (sample - dark) / transmission.data
        np.testing.assert_allclose(corrected.data.values, expected.data.values)

    def test_subtract_scattering_references_ignores_transmission_variance(self) -> None:
        import scipp as sc

        sample = sc.DataArray(
            data=sc.array(
                dims=["y", "x"],
                values=np.full((2, 2), 100.0, dtype=np.float64),
                variances=np.full((2, 2), 100.0, dtype=np.float64),
            )
        )
        dark = sc.DataArray(
            data=sc.array(
                dims=["y", "x"],
                values=np.full((2, 2), 10.0, dtype=np.float64),
                variances=np.full((2, 2), 10.0, dtype=np.float64),
            )
        )
        corrected = subtract_scattering_references(
            sample,
            sc.DataArray(data=sc.scalar(0.5, variance=99.0)),
            dark=dark,
        )

        expected_variances = (np.full((2, 2), 100.0) + np.full((2, 2), 10.0)) / (0.5 ** 2)
        np.testing.assert_allclose(corrected.data.variances, expected_variances)

    def test_subtract_scattering_references_rejects_non_dataarray_sample(self) -> None:
        import scipp as sc

        with self.assertRaisesRegex(TypeError, "sample must be a scipp.DataArray"):
            subtract_scattering_references(np.ones((2, 2), dtype=np.float64), sc.DataArray(data=sc.scalar(0.5)))


if __name__ == "__main__":
    unittest.main()
