from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from scarlet.reduction import (
    compute_corrected_water_scattering,
    compute_reference_transmissions,
    compute_transmission,
)
from scarlet.workflow.configuration import Aperture, Collimation, Configuration
from scarlet.workflow.reference import write_refs_norm_file, write_refs_sub_file


def _write_transmission_file(
    path: Path,
    *,
    data: np.ndarray,
    monitor_integral: float | None,
    entry_name: str = "entry",
    monitor0_integral: float | None = None,
) -> None:
    with h5py.File(path, "w") as f:
        entry = f.create_group(entry_name)
        entry.attrs["NX_class"] = b"NXentry"
        control = entry.create_group("control")
        control.attrs["NX_class"] = b"NXmonitor"
        if monitor_integral is not None:
            control.create_dataset("integral", data=float(monitor_integral))
        instrument = entry.create_group("instrument")
        instrument.attrs["NX_class"] = b"NXinstrument"
        if monitor0_integral is not None:
            monitor0 = instrument.create_group("monitor0")
            monitor0.attrs["NX_class"] = b"NXmonitor"
            monitor0.create_dataset("integral", data=float(monitor0_integral))
        detector = instrument.create_group("detector0")
        detector.attrs["NX_class"] = b"NXdetector"
        detector.create_dataset("data", data=np.asarray(data, dtype=np.float64))


class TestTransmission(unittest.TestCase):
    def test_compute_transmission_uses_monitor_normalized_roi_sum(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            transmission_file = root / "sample_transmission.nxs"
            empty_beam_file = root / "empty_beam_transmission.nxs"

            sample_data = np.zeros((4, 4), dtype=np.float64)
            sample_data[1:3, 1:3] = 40.0
            empty_beam_data = np.zeros((4, 4), dtype=np.float64)
            empty_beam_data[1:3, 1:3] = 100.0

            _write_transmission_file(transmission_file, data=sample_data, monitor_integral=20.0)
            _write_transmission_file(empty_beam_file, data=empty_beam_data, monitor_integral=10.0)

            transmission = compute_transmission(transmission_file, empty_beam_file, (1, 3, 1, 3))

            self.assertAlmostEqual(transmission, 0.2)

    def test_compute_transmission_accepts_raw_data_entry(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            transmission_file = root / "sample_transmission.nxs"
            empty_beam_file = root / "empty_beam_transmission.nxs"

            sample_data = np.ones((3, 3), dtype=np.float64)
            empty_beam_data = np.full((3, 3), 2.0, dtype=np.float64)

            _write_transmission_file(
                transmission_file,
                data=sample_data,
                monitor_integral=3.0,
                entry_name="raw_data",
            )
            _write_transmission_file(
                empty_beam_file,
                data=empty_beam_data,
                monitor_integral=3.0,
                entry_name="raw_data",
            )

            transmission = compute_transmission(transmission_file, empty_beam_file, (0, 3, 0, 3))

            self.assertAlmostEqual(transmission, 0.5)

    def test_compute_transmission_requires_control_integral(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            transmission_file = root / "sample_transmission.nxs"
            empty_beam_file = root / "empty_beam_transmission.nxs"

            sample_data = np.ones((3, 3), dtype=np.float64)
            empty_beam_data = np.ones((3, 3), dtype=np.float64)

            _write_transmission_file(
                transmission_file,
                data=sample_data,
                monitor_integral=None,
                monitor0_integral=10.0,
            )
            _write_transmission_file(
                empty_beam_file,
                data=empty_beam_data,
                monitor_integral=10.0,
            )

            with self.assertRaisesRegex(ValueError, "Missing monitor integral"):
                compute_transmission(transmission_file, empty_beam_file, (0, 3, 0, 3))

    def test_compute_corrected_water_scattering_uses_refs_norm_references(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            refs_norm = root / "refs_norm.nxs"
            water_scattering = root / "water_scattering.nxs"
            water_transmission = root / "water_transmission.nxs"
            empty_beam_transmission = root / "empty_beam_transmission.nxs"
            empty_beam_scattering = root / "empty_beam_scattering.nxs"
            empty_cell_scattering = root / "empty_cell_scattering.nxs"
            dark = root / "dark.nxs"

            _write_transmission_file(
                water_scattering,
                data=np.full((4, 4), 10.0, dtype=np.float64),
                monitor_integral=1.0,
            )
            _write_transmission_file(
                water_transmission,
                data=np.full((4, 4), 5.0, dtype=np.float64),
                monitor_integral=1.0,
            )
            _write_transmission_file(
                empty_beam_transmission,
                data=np.full((4, 4), 9.0, dtype=np.float64),
                monitor_integral=1.0,
            )
            _write_transmission_file(
                empty_beam_scattering,
                data=np.full((4, 4), 2.0, dtype=np.float64),
                monitor_integral=1.0,
            )
            _write_transmission_file(
                empty_cell_scattering,
                data=np.full((4, 4), 5.0, dtype=np.float64),
                monitor_integral=1.0,
            )
            _write_transmission_file(
                dark,
                data=np.full((4, 4), 1.0, dtype=np.float64),
                monitor_integral=1.0,
            )

            configuration = Configuration(
                wavelength=6.0,
                sample_detector_distance=4.2,
                config_id="config_1",
                collimation=Collimation(
                    aperture1=Aperture(type="slit", x_gap=0.002, y_gap=0.003),
                    aperture2=Aperture(type="pinhole", diameter=0.004),
                    collimation_distance=1.5,
                    last_aperture_to_sample_distance=0.5,
                ),
            )
            write_refs_norm_file(
                refs_norm,
                configuration,
                water_scattering=water_scattering,
                water_transmission=water_transmission,
                dark=dark,
                empty_beam_transmission=empty_beam_transmission,
                empty_beam_scattering=empty_beam_scattering,
                empty_cell_scattering=empty_cell_scattering,
                transmission_roi_detector=0,
                transmission_roi=(1, 3, 1, 3),
            )

            corrected = compute_corrected_water_scattering(refs_norm)

            np.testing.assert_allclose(corrected, np.full((4, 4), 6.5, dtype=np.float64))

    def test_compute_reference_transmissions_updates_refs_sub_entries(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            refs_sub = root / "refs_sub.nxs"
            empty_beam_transmission = root / "empty_beam_transmission.nxs"
            empty_beam_scattering = root / "empty_beam_scattering.nxs"
            empty_cell_transmission = root / "empty_cell_transmission.nxs"
            empty_cell_scattering = root / "empty_cell_scattering.nxs"
            dark = root / "dark.nxs"

            _write_transmission_file(
                empty_beam_transmission,
                data=np.full((4, 4), 9.0, dtype=np.float64),
                monitor_integral=1.0,
            )
            _write_transmission_file(
                empty_beam_scattering,
                data=np.full((4, 4), 2.0, dtype=np.float64),
                monitor_integral=1.0,
            )
            _write_transmission_file(
                empty_cell_transmission,
                data=np.full((4, 4), 5.0, dtype=np.float64),
                monitor_integral=1.0,
            )
            _write_transmission_file(
                empty_cell_scattering,
                data=np.full((4, 4), 7.0, dtype=np.float64),
                monitor_integral=1.0,
            )
            _write_transmission_file(
                dark,
                data=np.full((4, 4), 1.0, dtype=np.float64),
                monitor_integral=1.0,
            )

            configuration = Configuration(
                wavelength=6.0,
                sample_detector_distance=4.2,
                config_id="config_1",
                collimation=Collimation(
                    aperture1=Aperture(type="slit", x_gap=0.002, y_gap=0.003),
                    aperture2=Aperture(type="pinhole", diameter=0.004),
                    collimation_distance=1.5,
                    last_aperture_to_sample_distance=0.5,
                ),
            )
            write_refs_sub_file(
                refs_sub,
                configuration,
                empty_beam_transmission=empty_beam_transmission,
                dark=dark,
                empty_beam_scattering=empty_beam_scattering,
                empty_cell_transmission=empty_cell_transmission,
                empty_cell_scattering=empty_cell_scattering,
                transmission_roi_detector=0,
                transmission_roi=(1, 3, 1, 3),
            )

            updated = compute_reference_transmissions(refs_sub)

            self.assertAlmostEqual(updated["empty_beam_transmission"], 1.0)
            self.assertAlmostEqual(updated["empty_beam_scattering"], 1.0)
            self.assertAlmostEqual(updated["empty_cell_transmission"], 0.5)
            self.assertAlmostEqual(updated["empty_cell_scattering"], 0.5)
            with h5py.File(refs_sub, "r") as f:
                self.assertAlmostEqual(float(f["/entry/references/empty_beam_transmission/entry/sample/transmission"][()]), 1.0)
                self.assertAlmostEqual(float(f["/entry/references/empty_beam_scattering/entry/sample/transmission"][()]), 1.0)
                self.assertAlmostEqual(float(f["/entry/references/empty_cell_transmission/entry/sample/transmission"][()]), 0.5)
                self.assertAlmostEqual(float(f["/entry/references/empty_cell_scattering/entry/sample/transmission"][()]), 0.5)

    def test_compute_reference_transmissions_updates_refs_norm_water_entries(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            refs_norm = root / "refs_norm.nxs"
            water_scattering = root / "water_scattering.nxs"
            water_transmission = root / "water_transmission.nxs"
            empty_beam_transmission = root / "empty_beam_transmission.nxs"
            dark = root / "dark.nxs"

            _write_transmission_file(
                water_scattering,
                data=np.full((4, 4), 10.0, dtype=np.float64),
                monitor_integral=1.0,
            )
            _write_transmission_file(
                water_transmission,
                data=np.full((4, 4), 5.0, dtype=np.float64),
                monitor_integral=1.0,
            )
            _write_transmission_file(
                empty_beam_transmission,
                data=np.full((4, 4), 9.0, dtype=np.float64),
                monitor_integral=1.0,
            )
            _write_transmission_file(
                dark,
                data=np.full((4, 4), 1.0, dtype=np.float64),
                monitor_integral=1.0,
            )

            configuration = Configuration(
                wavelength=6.0,
                sample_detector_distance=4.2,
                config_id="config_1",
                collimation=Collimation(
                    aperture1=Aperture(type="slit", x_gap=0.002, y_gap=0.003),
                    aperture2=Aperture(type="pinhole", diameter=0.004),
                    collimation_distance=1.5,
                    last_aperture_to_sample_distance=0.5,
                ),
            )
            write_refs_norm_file(
                refs_norm,
                configuration,
                water_scattering=water_scattering,
                water_transmission=water_transmission,
                dark=dark,
                empty_beam_transmission=empty_beam_transmission,
                transmission_roi_detector=0,
                transmission_roi=(1, 3, 1, 3),
            )

            updated = compute_reference_transmissions(refs_norm)

            self.assertAlmostEqual(updated["empty_beam_transmission"], 1.0)
            self.assertAlmostEqual(updated["water_transmission"], 0.5)
            self.assertAlmostEqual(updated["water_scattering"], 0.5)
            with h5py.File(refs_norm, "r") as f:
                self.assertAlmostEqual(float(f["/entry/references/empty_beam_transmission/entry/sample/transmission"][()]), 1.0)
                self.assertAlmostEqual(float(f["/entry/references/water_transmission/entry/sample/transmission"][()]), 0.5)
                self.assertAlmostEqual(float(f["/entry/references/water_scattering/entry/sample/transmission"][()]), 0.5)


if __name__ == "__main__":
    unittest.main()
