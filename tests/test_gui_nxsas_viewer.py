from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np

from scarlet.gui import format_nexus_summary, list_nexus_files, prepare_view_file, read_nexus_dataset, scan_nexus_file


def _write_test_nxsas_file(path: Path, *, definition: str = "NXsas_raw") -> None:
    with h5py.File(path, "w") as f:
        entry = f.create_group("raw_data")
        entry.attrs["NX_class"] = b"NXentry"
        entry.create_dataset("definition", data=np.bytes_(definition))
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

        sample = entry.create_group("sample")
        sample.attrs["NX_class"] = b"NXsample"
        sample.create_dataset("name", data=np.bytes_("sample-a"))


def _write_test_nxsas_file_with_array_sample_name(path: Path) -> None:
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

        sample = entry.create_group("sample")
        sample.attrs["NX_class"] = b"NXsample"
        sample.create_dataset("name", data=np.asarray([np.bytes_("S1_P_PB_25_2mm")]))


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


def _write_minimal_raw_input_file(path: Path) -> None:
    with h5py.File(path, "w") as f:
        entry = f.create_group("entry0")
        entry.attrs["NX_class"] = b"NXentry"
        entry.create_dataset("title", data=np.bytes_("raw-input"))


class TestGuiNxsasViewerHelpers(unittest.TestCase):
    def test_list_nexus_files_filters_supported_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            expected = [directory / "raw_a.h5", directory / "mask_b.nxs", directory / "raw_c.HDF5"]
            _write_test_nxsas_file(expected[0])
            _write_test_mask_file(expected[1])
            _write_test_nxsas_file(expected[2])
            (directory / "notes.txt").write_text("ignore", encoding="utf-8")

            files = list_nexus_files(directory)

            self.assertEqual(files, sorted(path.resolve() for path in expected))

    def test_list_nexus_files_ignores_refs_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            visible = directory / "mask.nxs"
            ignored_sub = directory / "refs_sub.nxs"
            ignored_norm = directory / "refs_norm.nxs"
            _write_test_mask_file(visible)
            _write_test_nxsas_file(ignored_sub, definition="SCARLET_refs_sub")
            _write_test_nxsas_file(ignored_norm, definition="SCARLET_refs_norm")

            files = list_nexus_files(directory)

            self.assertEqual(files, [visible.resolve()])

    def test_scan_nexus_file_collects_entries_and_detector_images(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            file_path = Path(td) / "raw.nxs"
            _write_test_nxsas_file(file_path)

            summary = scan_nexus_file(file_path)

            self.assertEqual(summary.file_path, file_path.resolve())
            self.assertEqual(summary.definition, "NXsas_raw")
            self.assertEqual(summary.sample_name, "sample-a")
            self.assertAlmostEqual(summary.detector0_distance_m, 4.2)
            self.assertAlmostEqual(summary.collimation_distance_m, 1.5)
            self.assertAlmostEqual(summary.wavelength_a, 6.0)
            self.assertEqual(summary.entry_paths, ["/raw_data"])
            self.assertEqual(
                summary.detector_paths,
                [
                    "/raw_data/instrument/detector0/data",
                    "/raw_data/instrument/detector1/data",
                ],
            )
            self.assertIn("/raw_data/instrument/detector0/data", summary.image_dataset_paths)
            self.assertIn("/raw_data/sample/name", {node.path for node in summary.nodes})

    def test_scan_mask_file_collects_mask_datasets_as_detector_tabs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            file_path = Path(td) / "masks.nxs"
            _write_test_mask_file(file_path)

            summary = scan_nexus_file(file_path)

            self.assertEqual(summary.definition, "SCARLET_masks")
            self.assertIsNone(summary.sample_name)
            self.assertAlmostEqual(summary.detector0_distance_m, 4.2)
            self.assertAlmostEqual(summary.collimation_distance_m, 1.5)
            self.assertAlmostEqual(summary.wavelength_a, 6.0)
            self.assertEqual(
                summary.detector_paths,
                [
                    "/entry/mask/mask_detector0",
                    "/entry/mask/mask_detector1",
                ],
            )
            self.assertIn("/entry/mask/mask_detector0", summary.image_dataset_paths)

    def test_read_nexus_dataset_and_format_summary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            file_path = Path(td) / "raw.nxs"
            _write_test_nxsas_file(file_path)

            dataset = read_nexus_dataset(file_path, "/raw_data/instrument/detector1/data")
            summary_text = format_nexus_summary(scan_nexus_file(file_path))

            np.testing.assert_array_equal(dataset, np.arange(20, dtype=np.float64).reshape(4, 5))
            self.assertTrue(summary_text.startswith("raw.nxs\nsample-a\n"))
            self.assertIn("distance= 4.2 m ; collimation= 1.5 m ; wavelength= 6 A", summary_text)
            self.assertIn("Detector images: 2", summary_text)
            self.assertIn("/raw_data", summary_text)
            self.assertIn("Definition: NXsas_raw", summary_text)

    def test_format_summary_for_mask_has_empty_second_line(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            file_path = Path(td) / "masks.nxs"
            _write_test_mask_file(file_path)

            summary_text = format_nexus_summary(scan_nexus_file(file_path))

            self.assertTrue(summary_text.startswith("masks.nxs\n\n"))
            self.assertIn("distance= 4.2 m ; collimation= 1.5 m ; wavelength= 6 A", summary_text)

    def test_scan_nexus_file_decodes_array_sample_name(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            file_path = Path(td) / "raw_array_name.nxs"
            _write_test_nxsas_file_with_array_sample_name(file_path)

            summary = scan_nexus_file(file_path)
            summary_text = format_nexus_summary(summary)

            self.assertEqual(summary.sample_name, "S1_P_PB_25_2mm")
            self.assertTrue(summary_text.startswith("raw_array_name.nxs\nS1_P_PB_25_2mm\n"))

    def test_prepare_view_file_keeps_direct_nxsas_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            file_path = Path(td) / "raw.nxs"
            _write_test_nxsas_file(file_path)

            prepared = prepare_view_file(file_path, apparatus="sam", temp_dir=Path(td) / "tmp")

            self.assertEqual(prepared.source_file, file_path.resolve())
            self.assertEqual(prepared.view_file, file_path.resolve())
            self.assertFalse(prepared.converted)
            self.assertIsNone(prepared.apparatus)

    def test_prepare_view_file_converts_raw_input_with_selected_apparatus(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            source_file = Path(td) / "raw_input.nxs"
            temp_dir = Path(td) / "tmp"
            _write_minimal_raw_input_file(source_file)

            def _fake_convert(apparatus: str, input_path: Path, output_path: Path, **kwargs) -> None:
                self.assertEqual(apparatus, "sansllb")
                self.assertEqual(Path(input_path).resolve(), source_file.resolve())
                self.assertTrue(kwargs["overwrite"])
                _write_test_nxsas_file(Path(output_path))

            with mock.patch("scarlet.io.converters.convert_to_scarlet_nxsas_raw", side_effect=_fake_convert):
                prepared = prepare_view_file(source_file, apparatus="sansllb", temp_dir=temp_dir)

            self.assertEqual(prepared.source_file, source_file.resolve())
            self.assertTrue(prepared.converted)
            self.assertEqual(prepared.apparatus, "sansllb")
            self.assertTrue(prepared.view_file.exists())
            self.assertEqual(scan_nexus_file(prepared.view_file).definition, "NXsas_raw")


if __name__ == "__main__":
    unittest.main()
