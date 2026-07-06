from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from scarlet.io import (
    get_roi,
    read_deadtime_value,
    read_detector,
    read_detector_data,
    read_detector_deadtime,
    read_detector_error,
    read_detector_pixel_size,
    read_empty_beam_transmission_source_file,
)
from scarlet.reduction.correction import correct_detector_dead_time


def _write_test_raw_file(
    path: Path,
    *,
    dataset_name: str | None,
    value: float | None,
    image_data: np.ndarray | None = None,
    monitor: float = 100.0,
    count_time: float | None = None,
    include_pixel_size: bool = False,
    deadtime_corrected: bool = False,
) -> None:
    with h5py.File(path, "w") as f:
        entry = f.create_group("raw_data")
        entry.attrs["NX_class"] = b"NXentry"
        control = entry.create_group("control")
        control.create_dataset("integral", data=float(monitor))
        if count_time is not None:
            control.create_dataset("count_time", data=float(count_time))
        instrument = entry.create_group("instrument")
        detector = instrument.create_group("detector0")
        detector.attrs["NX_class"] = b"NXdetector"
        if image_data is None:
            image_data = np.arange(12, dtype=np.float64).reshape(3, 4)
        detector.create_dataset("data", data=np.asarray(image_data, dtype=np.float64))
        if include_pixel_size:
            detector.create_dataset("x_pixel_size", data=0.001)
            detector.create_dataset("y_pixel_size", data=0.002)
        detector.create_dataset("deadtime_corrected", data=bool(deadtime_corrected))
        if dataset_name is not None:
            detector.create_dataset(dataset_name, data=(float("nan") if value is None else float(value)))


def _write_test_refs_file(path: Path, *, definition: str, source_file: str | None) -> None:
    with h5py.File(path, "w") as f:
        entry = f.create_group("entry")
        entry.attrs["NX_class"] = b"NXentry"
        entry.create_dataset("definition", data=np.bytes_(definition))
        meta = entry.create_group("meta")
        if source_file is not None:
            meta.create_dataset("empty_beam_transmission_source_file", data=np.bytes_(source_file))
        transmission_roi = entry.create_group("transmission_roi")
        transmission_roi.create_dataset("detector", data=np.bytes_("detector0"))
        transmission_roi.create_dataset("x0", data=1)
        transmission_roi.create_dataset("x1", data=5)
        transmission_roi.create_dataset("y0", data=2)
        transmission_roi.create_dataset("y1", data=7)


class TestNexusReader(unittest.TestCase):
    def test_read_detector_returns_dataarray_or_clear_import_error(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            file_path = Path(td) / "raw.nxs"
            _write_test_raw_file(file_path, dataset_name=None, value=None, include_pixel_size=True)

            if importlib.util.find_spec("scipp") is None:
                with self.assertRaisesRegex(ImportError, "scipp is required"):
                    read_detector(file_path, 0)
                return

            detector = read_detector(file_path, 0)
            self.assertEqual(tuple(detector.dims), ("y", "x"))
            np.testing.assert_allclose(detector.values, np.arange(12, dtype=np.float64).reshape(3, 4))
            np.testing.assert_allclose(detector.variances, np.arange(12, dtype=np.float64).reshape(3, 4))
            np.testing.assert_allclose(detector.coords["x"].values, np.arange(4, dtype=np.float64))
            np.testing.assert_allclose(detector.coords["y"].values, np.arange(3, dtype=np.float64))
            self.assertNotIn("x_pixel_size", detector.coords)
            self.assertNotIn("y_pixel_size", detector.coords)

    def test_read_detector_deadtime_reads_dead_time_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            file_path = Path(td) / "raw.nxs"
            _write_test_raw_file(file_path, dataset_name="dead_time", value=1.2e-6)

            deadtime = read_detector_deadtime(file_path, 0)

            self.assertAlmostEqual(deadtime, 1.2e-6)

    def test_read_deadtime_value_reads_dead_time_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            file_path = Path(td) / "raw.nxs"
            _write_test_raw_file(file_path, dataset_name="dead_time", value=1.2e-6)

            deadtime = read_deadtime_value(file_path, 0)

            self.assertAlmostEqual(deadtime, 1.2e-6)

    def test_read_detector_pixel_size_reads_file_metadata_without_exposing_it_as_coords(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            file_path = Path(td) / "raw.nxs"
            _write_test_raw_file(file_path, dataset_name=None, value=None, include_pixel_size=True)

            pixel_size = read_detector_pixel_size(file_path, 0)

            self.assertEqual(pixel_size, (0.001, 0.002))

    def test_read_detector_deadtime_reads_deadtime_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            file_path = Path(td) / "raw.nxs"
            _write_test_raw_file(file_path, dataset_name="deadtime", value=2.5e-6)

            deadtime = read_detector_deadtime(file_path, 0)

            self.assertAlmostEqual(deadtime, 2.5e-6)

    def test_read_detector_deadtime_returns_zero_when_missing_or_nan(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            missing_file = Path(td) / "missing.nxs"
            nan_file = Path(td) / "nan.nxs"
            _write_test_raw_file(missing_file, dataset_name=None, value=None)
            _write_test_raw_file(nan_file, dataset_name="dead_time", value=None)

            self.assertEqual(read_detector_deadtime(missing_file, 0), 0.0)
            self.assertEqual(read_detector_deadtime(nan_file, 0), 0.0)

    def test_read_detector_data_can_correct_deadtime_before_monitor_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            file_path = Path(td) / "raw.nxs"
            image = np.array([[50.0, 100.0], [25.0, 40.0]], dtype=np.float64)
            _write_test_raw_file(
                file_path,
                dataset_name="dead_time",
                value=1.0e-2,
                image_data=image,
                monitor=10.0,
                count_time=10.0,
            )

            corrected = read_detector_data(
                file_path,
                0,
                correct_deadtime=True,
                normalize_by_monitor=True,
            )

            expected = correct_detector_dead_time(image, acq_time=10.0, deadtime=1.0e-2) / 10.0
            np.testing.assert_allclose(corrected, expected)

    def test_read_detector_error_uses_corrected_data_when_deadtime_is_applied(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            file_path = Path(td) / "raw.nxs"
            image = np.array([[50.0, 100.0], [25.0, 40.0]], dtype=np.float64)
            _write_test_raw_file(
                file_path,
                dataset_name="dead_time",
                value=1.0e-2,
                image_data=image,
                monitor=10.0,
                count_time=10.0,
            )

            error = read_detector_error(
                file_path,
                0,
                correct_deadtime=True,
                normalize_by_monitor=True,
            )

            expected_data = correct_detector_dead_time(image, acq_time=10.0, deadtime=1.0e-2)
            np.testing.assert_allclose(error, np.sqrt(expected_data / 10.0) / 10.0)

    def test_read_detector_data_skips_deadtime_recorrection_when_file_is_already_corrected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            file_path = Path(td) / "raw.nxs"
            image = np.array([[50.0, 100.0], [25.0, 40.0]], dtype=np.float64)
            corrected = correct_detector_dead_time(image, acq_time=10.0, deadtime=1.0e-2)
            _write_test_raw_file(
                file_path,
                dataset_name="dead_time",
                value=1.0e-2,
                image_data=corrected,
                monitor=10.0,
                count_time=10.0,
                deadtime_corrected=True,
            )

            loaded = read_detector_data(
                file_path,
                0,
                correct_deadtime=True,
                normalize_by_monitor=True,
            )

            np.testing.assert_allclose(loaded, corrected / 10.0)

    def test_read_detector_error_uses_count_time_for_already_corrected_rate_data(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            file_path = Path(td) / "raw.nxs"
            image = np.array([[50.0, 100.0], [25.0, 40.0]], dtype=np.float64)
            corrected = correct_detector_dead_time(image, acq_time=10.0, deadtime=1.0e-2)
            _write_test_raw_file(
                file_path,
                dataset_name="dead_time",
                value=1.0e-2,
                image_data=corrected,
                monitor=10.0,
                count_time=10.0,
                deadtime_corrected=True,
            )

            error = read_detector_error(
                file_path,
                0,
                correct_deadtime=True,
                normalize_by_monitor=True,
            )

            np.testing.assert_allclose(error, np.sqrt(corrected / 10.0) / 10.0)

    def test_read_detector_data_requires_count_time_for_non_zero_deadtime_correction(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            file_path = Path(td) / "raw.nxs"
            _write_test_raw_file(file_path, dataset_name="dead_time", value=1.0e-2, count_time=None)

            with self.assertRaisesRegex(ValueError, "count_time"):
                read_detector_data(file_path, 0, correct_deadtime=True)

    def test_read_detector_accepts_deadtime_correction_option(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            file_path = Path(td) / "raw.nxs"
            image = np.array([[50.0, 100.0], [25.0, 40.0]], dtype=np.float64)
            _write_test_raw_file(
                file_path,
                dataset_name="dead_time",
                value=1.0e-2,
                image_data=image,
                monitor=10.0,
                count_time=10.0,
            )

            if importlib.util.find_spec("scipp") is None:
                with self.assertRaisesRegex(ImportError, "scipp is required"):
                    read_detector(file_path, 0, correct_deadtime=True, normalize_by_monitor=True)
                return

            detector = read_detector(file_path, 0, correct_deadtime=True, normalize_by_monitor=True)
            expected = correct_detector_dead_time(image, acq_time=10.0, deadtime=1.0e-2) / 10.0
            np.testing.assert_allclose(detector.values, expected)

    def test_read_empty_beam_transmission_source_file_reads_refs_bundle_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            refs_file = Path(td) / "refs_sub.nxs"
            source_file = Path(td) / "empty_beam_transmission.nxs"
            _write_test_refs_file(
                refs_file,
                definition="SCARLET_refs_sub",
                source_file=str(source_file),
            )

            returned = read_empty_beam_transmission_source_file(refs_file)

            self.assertEqual(returned, source_file.resolve())

    def test_read_empty_beam_transmission_source_file_accepts_refs_norm(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            refs_file = Path(td) / "refs_norm.nxs"
            source_file = Path(td) / "empty_beam_transmission.nxs"
            _write_test_refs_file(
                refs_file,
                definition="SCARLET_refs_norm",
                source_file=str(source_file),
            )

            returned = read_empty_beam_transmission_source_file(refs_file)

            self.assertEqual(returned, source_file.resolve())

    def test_read_empty_beam_transmission_source_file_rejects_non_refs_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            file_path = Path(td) / "raw.nxs"
            _write_test_refs_file(
                file_path,
                definition="NXsas_raw",
                source_file=str(Path(td) / "empty_beam_transmission.nxs"),
            )

            with self.assertRaisesRegex(ValueError, "Unsupported refs bundle definition"):
                read_empty_beam_transmission_source_file(file_path)

    def test_get_roi_reads_transmission_roi_from_refs_sub(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            refs_file = Path(td) / "refs_sub.nxs"
            _write_test_refs_file(
                refs_file,
                definition="SCARLET_refs_sub",
                source_file=str(Path(td) / "empty_beam_transmission.nxs"),
            )

            roi, detector = get_roi(refs_file)

            self.assertEqual(roi, [1, 5, 2, 7])
            self.assertEqual(detector, 0)

    def test_get_roi_accepts_refs_norm(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            refs_file = Path(td) / "refs_norm.nxs"
            _write_test_refs_file(
                refs_file,
                definition="SCARLET_refs_norm",
                source_file=str(Path(td) / "empty_beam_transmission.nxs"),
            )

            roi, detector = get_roi(refs_file)

            self.assertEqual(roi, [1, 5, 2, 7])
            self.assertEqual(detector, 0)

    def test_get_roi_rejects_non_refs_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            file_path = Path(td) / "raw.nxs"
            _write_test_refs_file(
                file_path,
                definition="NXsas_raw",
                source_file=str(Path(td) / "empty_beam_transmission.nxs"),
            )

            with self.assertRaisesRegex(ValueError, "Unsupported refs bundle definition"):
                get_roi(file_path)

    def test_get_roi_accepts_integer_detector_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            refs_file = Path(td) / "refs_sub.nxs"
            _write_test_refs_file(
                refs_file,
                definition="SCARLET_refs_sub",
                source_file=str(Path(td) / "empty_beam_transmission.nxs"),
            )
            with h5py.File(refs_file, "r+") as f:
                del f["/entry/transmission_roi/detector"]
                f["/entry/transmission_roi"].create_dataset("detector", data=2)

            roi, detector = get_roi(refs_file)

            self.assertEqual(roi, [1, 5, 2, 7])
            self.assertEqual(detector, 2)


if __name__ == "__main__":
    unittest.main()
