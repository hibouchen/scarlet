import tempfile
from pathlib import Path
import unittest

import h5py
import numpy as np

from scarlet.reduction.stiching import load_segment_from_nexus


def _write_processed_curve_file(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        processed = handle.create_group("processed")
        processed.attrs["NX_class"] = b"NXentry"

        data0 = processed.create_group("data0")
        data0.attrs["NX_class"] = b"NXdata"
        data0.create_dataset("q", data=np.asarray([0.05, 0.06], dtype=np.float64))
        data0.create_dataset("data", data=np.asarray([5.0, 6.0], dtype=np.float64))
        data0.create_dataset("errors", data=np.asarray([0.5, 0.6], dtype=np.float64))
        data0.create_dataset("q_error", data=np.asarray([0.005, 0.006], dtype=np.float64))

        data2 = processed.create_group("data2")
        data2.attrs["NX_class"] = b"NXdata"
        data2.create_dataset("q", data=np.asarray([0.2, 0.1, 0.3], dtype=np.float64))
        data2.create_dataset("data", data=np.asarray([20.0, 10.0, 30.0], dtype=np.float64))
        data2.create_dataset("errors", data=np.asarray([2.0, 1.0, 3.0], dtype=np.float64))
        data2.create_dataset("q_error", data=np.asarray([0.02, 0.01, 0.03], dtype=np.float64))


class TestStichingNexus(unittest.TestCase):
    def test_load_segment_from_nexus_uses_unknown_config_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            file_path = Path(td) / "sample_without_config.nxs"
            _write_processed_curve_file(file_path)

            segments = load_segment_from_nexus(file_path)

            self.assertEqual([segment.name for segment in segments], ["unknown_detector0", "unknown_detector2"])
            self.assertEqual([segment.config_id for segment in segments], ["unknown", "unknown"])
            self.assertEqual([segment.detector_id for segment in segments], ["detector0", "detector2"])
            self.assertEqual([segment.path for segment in segments], [str(file_path), str(file_path)])
            np.testing.assert_allclose(segments[0].curve.q, np.asarray([0.05, 0.06], dtype=np.float64))
            np.testing.assert_allclose(segments[0].curve.i, np.asarray([5.0, 6.0], dtype=np.float64))
            np.testing.assert_allclose(segments[0].curve.di, np.asarray([0.5, 0.6], dtype=np.float64))
            np.testing.assert_allclose(segments[0].curve.dq, np.asarray([0.005, 0.006], dtype=np.float64))
            np.testing.assert_allclose(segments[1].curve.q, np.asarray([0.1, 0.2, 0.3], dtype=np.float64))
            np.testing.assert_allclose(segments[1].curve.i, np.asarray([10.0, 20.0, 30.0], dtype=np.float64))
            np.testing.assert_allclose(segments[1].curve.di, np.asarray([1.0, 2.0, 3.0], dtype=np.float64))
            np.testing.assert_allclose(segments[1].curve.dq, np.asarray([0.01, 0.02, 0.03], dtype=np.float64))

    def test_load_segment_from_nexus_accepts_explicit_config_id(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            file_path = Path(td) / "sample_without_config.nxs"
            _write_processed_curve_file(file_path)

            segments = load_segment_from_nexus(file_path, config_id="config_9")

            self.assertEqual([segment.name for segment in segments], ["c9d0", "c9d2"])
            self.assertEqual([segment.config_id for segment in segments], ["config_9", "config_9"])

    def test_load_segment_from_nexus_returns_empty_list_when_processed_entry_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            file_path = Path(td) / "sample_without_config.nxs"
            with h5py.File(file_path, "w") as handle:
                entry = handle.create_group("entry")
                entry.attrs["NX_class"] = b"NXentry"

            segments = load_segment_from_nexus(file_path)

            self.assertEqual(segments, [])


if __name__ == "__main__":
    unittest.main()
