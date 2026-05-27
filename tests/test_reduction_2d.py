from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from scarlet.cli import main
from scarlet.reduction import reduce_2d
from scarlet.workflow.configuration import (
    Aperture,
    Collimation,
    Configuration,
    write_refs_norm_file,
    write_refs_sub_file,
)


def _write_synthetic_raw(
    path: Path,
    value: float | np.ndarray | tuple[float | np.ndarray, ...],
    *,
    sample_name: str,
    monitor: float = 1.0,
    beam_center_x: float = 0.5,
    beam_center_y: float = 0.5,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, tuple):
        detector_values = value
    else:
        detector_values = (value,)
    with h5py.File(path, "w") as f:
        entry = f.create_group("entry")
        entry.attrs["NX_class"] = np.bytes_("NXentry")
        entry.create_dataset("definition", data=np.bytes_("NXsas_raw"))
        entry.create_dataset("schema_version", data=np.bytes_("1.3"))

        sample = entry.create_group("sample")
        sample.attrs["NX_class"] = np.bytes_("NXsample")
        sample.create_dataset("name", data=np.bytes_(sample_name))

        control = entry.create_group("control")
        control.attrs["NX_class"] = np.bytes_("NXmonitor")
        control.create_dataset("integral", data=float(monitor))
        control.create_dataset("count_time", data=float(monitor))

        instrument = entry.create_group("instrument")
        instrument.attrs["NX_class"] = np.bytes_("NXinstrument")
        monochromator = instrument.create_group("monochromator")
        monochromator.attrs["NX_class"] = np.bytes_("NXmonochromator")
        monochromator.create_dataset("wavelength", data=6.0)
        for detector_index, detector_value in enumerate(detector_values):
            detector = instrument.create_group(f"detector{detector_index}")
            detector.attrs["NX_class"] = np.bytes_("NXdetector")
            if np.isscalar(detector_value):
                data = np.full((2, 2), float(detector_value), dtype=np.float64)
            else:
                data = np.asarray(detector_value, dtype=np.float64)
            detector.create_dataset("data", data=data)
            detector.create_dataset("distance", data=1.0)
            detector.create_dataset("x_pixel_size", data=0.01)
            detector.create_dataset("y_pixel_size", data=0.02)
            detector.create_dataset("beam_center_x", data=beam_center_x)
            detector.create_dataset("beam_center_y", data=beam_center_y)


def _configuration() -> Configuration:
    return Configuration(
        wavelength=6.0,
        sample_detector_distance=1.0,
        collimation=Collimation(
            aperture1=Aperture(type="slit", x_gap=0.01, y_gap=0.01),
            aperture2=Aperture(type="slit", x_gap=0.005, y_gap=0.005),
            collimation_distance=1.0,
            last_aperture_to_sample_distance=0.1,
        ),
        config_id="config_1",
    )


class TestReduction2D(unittest.TestCase):
    def test_first_deterministic_2d_reduction_with_water_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            refs = root / "refs"
            refs.mkdir()

            dark = raw / "dark.nxs"
            empty_beam_t = raw / "empty_beam_t.nxs"
            empty_beam_s = raw / "empty_beam_s.nxs"
            empty_cell_s = raw / "empty_cell_s.nxs"
            empty_cell_t = raw / "empty_cell_t.nxs"
            sample_s = raw / "sample_s.nxs"
            sample_t = raw / "sample_t.nxs"
            water_s = raw / "water_s.nxs"
            water_t = raw / "water_t.nxs"

            _write_synthetic_raw(dark, (1.0, 0.5), sample_name="Cd")
            _write_synthetic_raw(empty_beam_t, (10.0, 12.0), sample_name="empty_beam")
            _write_synthetic_raw(empty_beam_s, (2.0, 1.0), sample_name="empty_beam")
            _write_synthetic_raw(empty_cell_s, (5.0, 4.0), sample_name="empty_cell")
            _write_synthetic_raw(empty_cell_t, (7.0, 8.0), sample_name="empty_cell")
            _write_synthetic_raw(sample_s, (20.0, 18.0), sample_name="sample")
            _write_synthetic_raw(sample_t, (5.0, 6.0), sample_name="sample")
            _write_synthetic_raw(water_s, (50.0, 40.0), sample_name="water")
            _write_synthetic_raw(water_t, (10.0, 11.0), sample_name="water")

            refs_sub = refs / "refs_sub_config_1.nxs"
            refs_norm = refs / "refs_norm_config_1.nxs"
            write_refs_sub_file(
                refs_sub,
                _configuration(),
                dark=dark,
                empty_beam_transmission=empty_beam_t,
                empty_beam_scattering=empty_beam_s,
                empty_cell_transmission=empty_cell_t,
                empty_cell_scattering=empty_cell_s,
                transmission_roi=(0, 2, 0, 2),
            )
            write_refs_norm_file(
                refs_norm,
                _configuration(),
                dark=dark,
                empty_beam_transmission=empty_beam_t,
                empty_beam_scattering=empty_beam_s,
                empty_cell_transmission=empty_cell_t,
                empty_cell_scattering=empty_cell_s,
                water_scattering=water_s,
                water_transmission=water_t,
                transmission_roi=(0, 2, 0, 2),
            )

            output = root / "reduced.nxs"
            result = reduce_2d(
                sample_s,
                refs_sub,
                sample_transmission=sample_t,
                refs_norm=refs_norm,
                output_path=output,
                azimuthal_bins=1,
            )

            t_sample = (5.0 - 1.0) / (10.0 - 1.0)
            sample_corrected = (20.0 - 1.0 - (2.0 - 1.0)) - t_sample * ((5.0 - 1.0) - (2.0 - 1.0))
            water_corrected = (50.0 - 1.0 - (2.0 - 1.0)) - 1.0 * ((5.0 - 1.0) - (2.0 - 1.0))
            expected = sample_corrected / water_corrected
            sample_corrected_1 = (18.0 - 0.5 - (1.0 - 0.5)) - t_sample * ((4.0 - 0.5) - (1.0 - 0.5))
            water_corrected_1 = (40.0 - 0.5 - (1.0 - 0.5)) - 1.0 * ((4.0 - 0.5) - (1.0 - 0.5))
            expected_1 = sample_corrected_1 / water_corrected_1

            self.assertAlmostEqual(result.sample_transmission.value, t_sample)
            self.assertAlmostEqual(result.water_transmission.value, 1.0)
            self.assertEqual(result.detector_indices, [0, 1])
            np.testing.assert_allclose(result.sample_corrected, np.full((2, 2), sample_corrected))
            np.testing.assert_allclose(result.water_corrected, np.full((2, 2), water_corrected))
            np.testing.assert_allclose(result.intensity, np.full((2, 2), expected))
            np.testing.assert_allclose(
                result.detector_results[1].sample_corrected,
                np.full((2, 2), sample_corrected_1),
            )
            np.testing.assert_allclose(
                result.detector_results[1].water_corrected,
                np.full((2, 2), water_corrected_1),
            )
            np.testing.assert_allclose(
                result.detector_results[1].intensity,
                np.full((2, 2), expected_1),
            )

            with h5py.File(output, "r") as f:
                self.assertEqual(f["/processed_data/definition"][()].decode(), "SCARLET_azimuthal_iq")
                np.testing.assert_allclose(f["/processed_data/data/I"][()], np.array([expected]))
                np.testing.assert_allclose(f["/processed_data/data1/I"][()], np.array([expected_1]))
                np.testing.assert_array_equal(f["/processed_data/data/n_pixels"][()], np.array([4]))
                np.testing.assert_array_equal(f["/processed_data/data1/n_pixels"][()], np.array([4]))
                self.assertEqual(list(f["/processed_data/data"].attrs["axes"]), [b"Q"])
                np.testing.assert_allclose(
                    f["/processed_data/detector0/Qx"][()],
                    np.array([-0.00523592, 0.00523592]),
                    atol=1e-6,
                )
                np.testing.assert_allclose(
                    f["/processed_data/detector0/Qy"][()],
                    np.array([-0.01047089, 0.01047089]),
                    atol=1e-6,
                )
                self.assertEqual(f["/processed_data/data/Q"].attrs["units"], b"1/angstrom")
                self.assertEqual(f["/processed_data/detector0/Qx"].attrs["units"], b"1/angstrom")
                self.assertEqual(f["/processed_data/detector0/Qy"].attrs["units"], b"1/angstrom")
                np.testing.assert_allclose(f["/processed_data/detector0/I_2d"][()], np.full((2, 2), expected))
                self.assertAlmostEqual(float(f["/processed_data/reduction/sample_transmission/value"][()]), t_sample)
                np.testing.assert_array_equal(f["/processed_data/reduction/detector_indices"][()], np.array([0, 1]))

    def test_reduce_2d_cli_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            refs = root / "refs"
            refs.mkdir()

            dark = raw / "dark.nxs"
            empty_beam_t = raw / "empty_beam_t.nxs"
            sample_s = raw / "sample_s.nxs"
            sample_t = raw / "sample_t.nxs"

            _write_synthetic_raw(dark, 0.0, sample_name="Cd")
            _write_synthetic_raw(empty_beam_t, 10.0, sample_name="empty_beam")
            _write_synthetic_raw(sample_s, 4.0, sample_name="sample")
            _write_synthetic_raw(sample_t, 5.0, sample_name="sample")

            refs_sub = refs / "refs_sub_config_1.nxs"
            write_refs_sub_file(
                refs_sub,
                _configuration(),
                dark=dark,
                empty_beam_transmission=empty_beam_t,
                transmission_roi=(0, 2, 0, 2),
            )

            output = root / "cli_reduced.nxs"
            status = main([
                "reduce-2d",
                str(sample_s),
                str(refs_sub),
                str(output),
                "--sample-transmission",
                str(sample_t),
            ])
            self.assertEqual(status, 0)
            self.assertTrue(output.exists())

    def test_q_axes_use_center_of_mass_from_empty_beam_transmission_on_transmission_detector(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            refs = root / "refs"
            refs.mkdir()

            dark = raw / "dark.nxs"
            empty_beam_t = raw / "empty_beam_t.nxs"
            sample_s = raw / "sample_s.nxs"

            transmission_spot = np.array([[1.0, 3.0], [1.0, 3.0]], dtype=np.float64)
            _write_synthetic_raw(dark, 0.0, sample_name="Cd", beam_center_x=0.0, beam_center_y=0.0)
            _write_synthetic_raw(
                empty_beam_t,
                transmission_spot,
                sample_name="empty_beam",
                beam_center_x=0.0,
                beam_center_y=0.0,
            )
            _write_synthetic_raw(sample_s, 4.0, sample_name="sample", beam_center_x=0.0, beam_center_y=0.0)

            refs_sub = refs / "refs_sub_config_1.nxs"
            write_refs_sub_file(
                refs_sub,
                _configuration(),
                dark=dark,
                empty_beam_transmission=empty_beam_t,
                transmission_roi=(0, 2, 0, 2),
            )

            output = root / "reduced_q.nxs"
            reduce_2d(
                sample_s,
                refs_sub,
                output_path=output,
                azimuthal_bins=2,
            )

            expected_qx = np.array([-0.00785376, 0.00261799])
            expected_qy = np.array([-0.01047145, 0.01047145])

            with h5py.File(output, "r") as f:
                np.testing.assert_allclose(f["/processed_data/detector0/Qx"][()], expected_qx, atol=1e-6)
                np.testing.assert_allclose(f["/processed_data/detector0/Qy"][()], expected_qy, atol=1e-6)
                self.assertEqual(
                    f["/processed_data/beam_center_detector0/method"][()].decode(),
                    "center_of_mass_on_empty_beam_transmission_roi",
                )
                self.assertAlmostEqual(float(f["/processed_data/beam_center_detector0/x"][()]), 0.75)
                self.assertAlmostEqual(float(f["/processed_data/beam_center_detector0/y"][()]), 0.5)


if __name__ == "__main__":
    unittest.main()
