from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from scarlet.io import get_roi, read_detector_deadtime, read_empty_beam_transmission_source_file


def _write_test_raw_file(path: Path, *, dataset_name: str | None, value: float | None) -> None:
    with h5py.File(path, "w") as f:
        entry = f.create_group("raw_data")
        entry.attrs["NX_class"] = b"NXentry"
        control = entry.create_group("control")
        control.create_dataset("integral", data=100.0)
        instrument = entry.create_group("instrument")
        detector = instrument.create_group("detector0")
        detector.attrs["NX_class"] = b"NXdetector"
        detector.create_dataset("data", data=np.arange(12, dtype=np.float64).reshape(3, 4))
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
    def test_read_detector_deadtime_reads_dead_time_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            file_path = Path(td) / "raw.nxs"
            _write_test_raw_file(file_path, dataset_name="dead_time", value=1.2e-6)

            deadtime = read_detector_deadtime(file_path, 0)

            self.assertAlmostEqual(deadtime or 0.0, 1.2e-6)

    def test_read_detector_deadtime_reads_deadtime_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            file_path = Path(td) / "raw.nxs"
            _write_test_raw_file(file_path, dataset_name="deadtime", value=2.5e-6)

            deadtime = read_detector_deadtime(file_path, 0)

            self.assertAlmostEqual(deadtime or 0.0, 2.5e-6)

    def test_read_detector_deadtime_returns_none_when_missing_or_nan(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            missing_file = Path(td) / "missing.nxs"
            nan_file = Path(td) / "nan.nxs"
            _write_test_raw_file(missing_file, dataset_name=None, value=None)
            _write_test_raw_file(nan_file, dataset_name="dead_time", value=None)

            self.assertIsNone(read_detector_deadtime(missing_file, 0))
            self.assertIsNone(read_detector_deadtime(nan_file, 0))

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
