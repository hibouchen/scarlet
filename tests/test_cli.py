from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

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
                data.attrs["axes"] = np.asarray([np.bytes_("Q")])
                data.create_dataset("I", data=np.array([1.0, 2.0]))
                data.create_dataset("Q", data=np.array([0.5, 1.5]))
                data.create_dataset("Q_edges", data=np.array([0.0, 1.0, 2.0]))
                data.create_dataset("n_pixels", data=np.array([3, 5]))
                entry["data"] = entry["data0"]

            status = main([
                "azimuthal-average",
                str(reduced),
                str(output),
                "--overwrite",
            ])
            self.assertEqual(status, 0)
            self.assertTrue(output.exists())

    def test_nxsas_gui_command_dispatches_viewer(self) -> None:
        with mock.patch("scarlet.gui.run_nxsas_viewer") as run_nxsas_viewer:
            status = main(["nxsas-gui", "tests/data/sam/raw_data"])

        self.assertEqual(status, 0)
        run_nxsas_viewer.assert_called_once_with(Path("tests/data/sam/raw_data"))

    def test_viewer_command_dispatches_silx_viewer(self) -> None:
        with mock.patch("scarlet.gui.run_viewer", return_value=0) as run_viewer:
            status = main(["viewer", "tests/data/sam/raw_data", "--instrument", "sam"])

        self.assertEqual(status, 0)
        run_viewer.assert_called_once_with(Path("tests/data/sam/raw_data"), instrument="sam")


if __name__ == "__main__":
    unittest.main()
