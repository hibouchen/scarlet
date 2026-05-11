from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from scarlet.cli import main
from scarlet.reduction import azimuthal_average


def _write_reduced_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        entry = f.create_group("processed_data")
        entry.attrs["NX_class"] = np.bytes_("NXentry")
        entry.create_dataset("definition", data=np.bytes_("SCARLET_reduced_2d"))

        for index, base in enumerate((0.0, 40.0)):
            data = entry.create_group(f"data{index}")
            data.attrs["NX_class"] = np.bytes_("NXdata")
            data.attrs["signal"] = np.bytes_("I")
            data.attrs["axes"] = np.asarray([np.bytes_("Qy"), np.bytes_("Qx")])
            data.create_dataset(
                "I",
                data=np.array([[10.0 + base, 20.0 + base], [30.0 + base, 40.0 + base]], dtype=np.float64),
            )
            qx = data.create_dataset("Qx", data=np.array([0.0, 1.0], dtype=np.float64))
            qy = data.create_dataset("Qy", data=np.array([0.0, 1.0], dtype=np.float64))
            qx.attrs["units"] = np.bytes_("1/angstrom")
            qy.attrs["units"] = np.bytes_("1/angstrom")

        entry["data"] = entry["data0"]


class TestAzimuthalAverage(unittest.TestCase):
    def test_azimuthal_average_merges_selected_detectors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reduced = Path(tmp) / "reduced.nxs"
            _write_reduced_file(reduced)

            result = azimuthal_average(
                reduced,
                detector_indices=[0, 1],
                n_bins=2,
                q_min=0.0,
                q_max=1.5,
            )

            np.testing.assert_allclose(result.q, np.array([0.375, 1.125]))
            np.testing.assert_allclose(result.intensity, np.array([30.0, 50.0]))
            np.testing.assert_array_equal(result.n_pixels, np.array([2, 6]))
            self.assertEqual(result.detector_indices, [0, 1])

    def test_azimuthal_average_cli_writes_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reduced = Path(tmp) / "reduced.nxs"
            output = Path(tmp) / "iq.csv"
            _write_reduced_file(reduced)

            status = main([
                "azimuthal-average",
                str(reduced),
                str(output),
                "--bins",
                "2",
                "--q-min",
                "0.0",
                "--q-max",
                "1.5",
                "--overwrite",
            ])
            self.assertEqual(status, 0)
            self.assertTrue(output.exists())

            with output.open(newline="", encoding="utf-8") as fh:
                rows = list(csv.reader(fh))

            self.assertEqual(rows[0], ["q_A^-1", "I", "n_pixels"])
            self.assertEqual(rows[1][2], "2")
            self.assertEqual(rows[2][2], "6")


if __name__ == "__main__":
    unittest.main()
