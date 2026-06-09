from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np

from scarlet.gui.scarlet_viewer import build_mask_source, build_parser, default_mask_output_path, main


def _write_test_nxsas_file(path: Path) -> None:
    with h5py.File(path, "w") as f:
        entry = f.create_group("raw_data")
        entry.attrs["NX_class"] = b"NXentry"
        entry.create_dataset("definition", data=np.bytes_("NXsas_raw"))
        instrument = entry.create_group("instrument")
        instrument.attrs["NX_class"] = b"NXinstrument"
        monochromator = instrument.create_group("monochromator")
        monochromator.create_dataset("wavelength", data=6.0)

        detector0 = instrument.create_group("detector0")
        detector0.attrs["NX_class"] = b"NXdetector"
        detector0.create_dataset("data", data=np.arange(12, dtype=np.float64).reshape(3, 4))
        detector0_transformations = detector0.create_group("transformations")
        detector0_transformations.create_dataset("translation", data=np.array([0.0, 0.0, 4.2], dtype=float))

        detector1 = instrument.create_group("detector1")
        detector1.attrs["NX_class"] = b"NXdetector"
        detector1.create_dataset("data", data=np.arange(20, dtype=np.float64).reshape(4, 5))
        detector1_transformations = detector1.create_group("transformations")
        detector1_transformations.create_dataset("translation", data=np.array([0.0, 0.0, 5.0], dtype=float))

        collimation = instrument.create_group("collimation")
        elements = collimation.create_group("elements")
        aperture1 = elements.create_group("aperture1")
        aperture1.attrs["NX_class"] = b"NXslit"
        aperture1.create_dataset("x_gap", data=0.002)
        aperture1.create_dataset("y_gap", data=0.003)
        aperture1_transformations = aperture1.create_group("transformations")
        aperture1_transformations.create_dataset("translation", data=np.array([0.0, 0.0, -2.0], dtype=float))
        aperture2 = elements.create_group("aperture2")
        aperture2.attrs["NX_class"] = b"NXpinhole"
        aperture2.create_dataset("diameter", data=0.004)
        aperture2_transformations = aperture2.create_group("transformations")
        aperture2_transformations.create_dataset("translation", data=np.array([0.0, 0.0, -0.5], dtype=float))


def _write_test_mask_file(path: Path) -> None:
    with h5py.File(path, "w") as f:
        entry = f.create_group("entry")
        entry.attrs["NX_class"] = b"NXentry"
        entry.create_dataset("definition", data=np.bytes_("SCARLET_masks"))
        configuration = entry.create_group("configuration")
        configuration.attrs["NX_class"] = b"NXcollection"
        configuration.create_dataset("wavelength", data=6.0)
        configuration.create_dataset("sample_detector_distance", data=4.2)
        collimation = configuration.create_group("collimation")
        collimation.attrs["NX_class"] = b"NXcollection"
        collimation.create_dataset("collimation_distance", data=1.5)
        collimation.create_dataset("last_aperture_to_sample_distance", data=0.5)
        aperture1 = collimation.create_group("aperture1")
        aperture1.attrs["NX_class"] = b"NXslit"
        aperture1.create_dataset("x_gap", data=0.002)
        aperture1.create_dataset("y_gap", data=0.003)
        aperture2 = collimation.create_group("aperture2")
        aperture2.attrs["NX_class"] = b"NXpinhole"
        aperture2.create_dataset("diameter", data=0.004)
        mask = entry.create_group("mask")
        mask.attrs["NX_class"] = b"NXcollection"
        mask.create_dataset("mask_detector0", data=np.arange(12, dtype=np.uint8).reshape(3, 4) % 2)
        mask.create_dataset("mask_detector1", data=np.arange(20, dtype=np.uint8).reshape(4, 5) % 2)


class TestGuiScarletViewer(unittest.TestCase):
    def test_build_parser_defaults_to_sansllb(self) -> None:
        args = build_parser().parse_args([])

        self.assertIsNone(args.directory)
        self.assertEqual(args.instrument, "sansllb")

    def test_main_dispatches_run_viewer(self) -> None:
        with mock.patch("scarlet.gui.scarlet_viewer.run_viewer", return_value=0) as run_viewer:
            status = main(["tests/data/sam/raw_data", "--instrument", "sam"])

        self.assertEqual(status, 0)
        run_viewer.assert_called_once_with(Path("tests/data/sam/raw_data"), instrument="sam")

    def test_build_mask_source_from_nxsas_view_uses_original_source_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            source_file = Path(td) / "input_raw.hdf"
            view_file = Path(td) / "converted.nxs"
            source_file.write_bytes(b"raw")
            _write_test_nxsas_file(view_file)

            source = build_mask_source(source_file, view_file)

            self.assertEqual(source.file_path, source_file.resolve())
            self.assertEqual(source.entry_path, "/raw_data")
            self.assertEqual(sorted(source.detector_data), [0, 1])
            self.assertEqual(source.detector_data[0].shape, (3, 4))
            self.assertAlmostEqual(source.configuration.wavelength, 6.0)

    def test_build_mask_source_from_mask_view_reads_mask_detector_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            mask_file = Path(td) / "existing_masks.nxs"
            _write_test_mask_file(mask_file)

            source = build_mask_source(mask_file, mask_file)

            self.assertEqual(source.file_path, mask_file.resolve())
            self.assertEqual(source.entry_path, "/entry")
            self.assertEqual(sorted(source.detector_data), [0, 1])
            np.testing.assert_array_equal(
                source.detector_data[0],
                np.arange(12, dtype=np.uint8).reshape(3, 4) % 2,
            )

    def test_default_mask_output_path_appends_masks_suffix(self) -> None:
        path = default_mask_output_path("/tmp/sample_raw.nxs")

        self.assertEqual(path, Path("/tmp/sample_raw_masks.nxs"))


if __name__ == "__main__":
    unittest.main()
