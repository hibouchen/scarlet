from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from scarlet.reduction import compute_direct_flux


def _write_direct_beam_file(
    path: Path,
    *,
    data: np.ndarray,
    monitor: float,
    attenuator_transmission: float | None = None,
    create_attenuator: bool = False,
    entry_name: str = "entry",
) -> None:
    with h5py.File(path, "w") as handle:
        entry = handle.create_group(entry_name)
        entry.attrs["NX_class"] = np.bytes_("NXentry")
        control = entry.create_group("control")
        control.attrs["NX_class"] = np.bytes_("NXmonitor")
        control.create_dataset("integral", data=float(monitor))
        instrument = entry.create_group("instrument")
        instrument.attrs["NX_class"] = np.bytes_("NXinstrument")
        detector = instrument.create_group("detector0")
        detector.attrs["NX_class"] = np.bytes_("NXdetector")
        detector.create_dataset("data", data=np.asarray(data, dtype=np.float64))
        if create_attenuator or attenuator_transmission is not None:
            attenuator = instrument.create_group("attenuator")
            attenuator.attrs["NX_class"] = np.bytes_("NXattenuator")
            attenuator.create_dataset("selection", data=np.bytes_("test"))
            if attenuator_transmission is not None:
                attenuator.create_dataset(
                    "attenuator_transmission",
                    data=float(attenuator_transmission),
                )


class TestDirectFlux(unittest.TestCase):
    def test_compute_direct_flux_uses_full_detector_monitor_and_attenuator(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "direct_beam.nxs"
            _write_direct_beam_file(
                path,
                data=np.full((2, 2), 10.0),
                monitor=20.0,
                attenuator_transmission=0.25,
            )

            self.assertAlmostEqual(compute_direct_flux(path), 8.0)

    def test_compute_direct_flux_can_integrate_an_roi(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "direct_beam.nxs"
            _write_direct_beam_file(
                path,
                data=np.arange(1.0, 17.0).reshape(4, 4),
                monitor=2.0,
                attenuator_transmission=0.5,
            )

            self.assertAlmostEqual(compute_direct_flux(path, roi=(1, 3, 1, 3)), 34.0)

    def test_compute_direct_flux_treats_missing_attenuator_as_open_beam(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "direct_beam.nxs"
            _write_direct_beam_file(path, data=np.ones((2, 2)), monitor=2.0)

            self.assertAlmostEqual(compute_direct_flux(path), 2.0)

    def test_compute_direct_flux_rejects_incomplete_attenuator(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "direct_beam.nxs"
            _write_direct_beam_file(
                path,
                data=np.ones((2, 2)),
                monitor=2.0,
                create_attenuator=True,
            )

            with self.assertRaisesRegex(ValueError, "Missing attenuator transmission"):
                compute_direct_flux(path)

    def test_compute_direct_flux_accepts_raw_data_entry(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "direct_beam.nxs"
            _write_direct_beam_file(
                path,
                data=np.ones((2, 2)),
                monitor=2.0,
                attenuator_transmission=0.5,
                entry_name="raw_data",
            )

            self.assertAlmostEqual(compute_direct_flux(path), 4.0)


if __name__ == "__main__":
    unittest.main()
