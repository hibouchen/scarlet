from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from scarlet.cli import main


class TestCli(unittest.TestCase):
    def test_schema_list_command(self) -> None:
        self.assertEqual(main(["schema", "list"]), 0)

    def test_convert_list_command(self) -> None:
        self.assertEqual(main(["convert", "list"]), 0)

    def test_unknown_converter_returns_usage_error(self) -> None:
        status = main(["convert", "unknown", "input.nxs", "output.nxs"])
        self.assertEqual(status, 2)

    def test_azimuthal_average_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reduced = Path(tmp) / "reduced.nxs"
            output = Path(tmp) / "iq.csv"
            with h5py.File(reduced, "w") as f:
                entry = f.create_group("processed_data")
                entry.attrs["NX_class"] = np.bytes_("NXentry")
                data = entry.create_group("data0")
                data.attrs["NX_class"] = np.bytes_("NXdata")
                data.attrs["signal"] = np.bytes_("I")
                data.attrs["axes"] = np.asarray([np.bytes_("Qy"), np.bytes_("Qx")])
                data.create_dataset("I", data=np.array([[1.0, 2.0], [3.0, 4.0]]))
                data.create_dataset("Qx", data=np.array([0.0, 1.0]))
                data.create_dataset("Qy", data=np.array([0.0, 1.0]))
                entry["data"] = entry["data0"]

            status = main([
                "azimuthal-average",
                str(reduced),
                str(output),
                "--bins",
                "2",
                "--q-min",
                "0",
                "--q-max",
                "1.5",
                "--overwrite",
            ])
            self.assertEqual(status, 0)
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
