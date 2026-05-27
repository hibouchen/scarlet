from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from scarlet.gui import load_mask_source, write_mask_bundle
from scarlet.validation.schema_loader import load_schema
from scarlet.validation.schema_validator import validate_nexus_file


def _write_test_raw_nexus(path: Path) -> None:
    with h5py.File(path, "w") as f:
        entry = f.create_group("raw_data")
        entry.attrs["NX_class"] = b"NXentry"
        sample = entry.create_group("sample")
        sample.attrs["NX_class"] = b"NXsample"
        sample.create_dataset("name", data=np.bytes_("sample"))

        instrument = entry.create_group("instrument")
        instrument.attrs["NX_class"] = b"NXinstrument"
        mono = instrument.create_group("monochromator")
        mono.create_dataset("wavelength", data=6.0)

        for detector_index, beam_center in ((0, (1.5, 2.5)), (1, (0.5, 1.5))):
            detector = instrument.create_group(f"detector{detector_index}")
            detector.attrs["NX_class"] = b"NXdetector"
            detector.create_dataset(
                "data",
                data=np.arange(12, dtype=np.float64).reshape(3, 4) + detector_index,
            )
            detector.create_dataset("beam_center_x", data=float(beam_center[0]))
            detector.create_dataset("beam_center_y", data=float(beam_center[1]))
            transformations = detector.create_group("transformations")
            transformations.create_dataset("translation", data=np.array([0.0, 0.0, 4.2], dtype=float))

        col = instrument.create_group("collimation")
        elements = col.create_group("elements")

        ap1 = elements.create_group("ap1")
        ap1.attrs["NX_class"] = b"NXslit"
        ap1.create_dataset("x_gap", data=0.002)
        ap1.create_dataset("y_gap", data=0.003)
        ap1_tr = ap1.create_group("transformations")
        ap1_tr.create_dataset("translation", data=np.array([0.0, 0.0, -2.0], dtype=float))

        ap2 = elements.create_group("ap2")
        ap2.attrs["NX_class"] = b"NXpinhole"
        ap2.create_dataset("diameter", data=0.004)
        ap2_tr = ap2.create_group("transformations")
        ap2_tr.create_dataset("translation", data=np.array([0.0, 0.0, -0.5], dtype=float))


class TestGuiMaskEditor(unittest.TestCase):
    def test_load_mask_source_reads_detector_images(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            source_path = Path(td) / "raw.nxs"
            _write_test_raw_nexus(source_path)

            source = load_mask_source(source_path)

            self.assertEqual(source.file_path, source_path.resolve())
            self.assertEqual(source.entry_path, "/raw_data")
            self.assertEqual(source.detector_indices, [0, 1])
            self.assertEqual(source.detector_data[0].shape, (3, 4))
            self.assertEqual(source.detector_data[1].shape, (3, 4))
            self.assertAlmostEqual(source.configuration.wavelength, 6.0)
            self.assertAlmostEqual(float(source.configuration.sample_detector_distance), 4.2)

    def test_write_mask_bundle_stores_masks_and_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            source_path = Path(td) / "raw.nxs"
            out_path = Path(td) / "masks.nxs"
            _write_test_raw_nexus(source_path)

            source = load_mask_source(source_path)
            masks = {
                0: np.array(
                    [[0, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 1]],
                    dtype=np.uint8,
                ),
                1: np.array(
                    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
                    dtype=np.uint8,
                ),
            }

            write_mask_bundle(out_path, source, masks)

            report = validate_nexus_file(out_path, load_schema("scarlet_masks_v1.0.yaml"))
            self.assertTrue(report.ok, report.format_lines())

            with h5py.File(out_path, "r") as f:
                self.assertEqual(f["/entry/definition"][()].decode(), "SCARLET_masks")
                np.testing.assert_array_equal(f["/entry/mask/mask_detector0"][()], masks[0])
                np.testing.assert_array_equal(f["/entry/mask/mask_detector1"][()], masks[1])
                self.assertEqual(f["/entry/meta/source_file"][()].decode(), str(source_path.resolve()))
                self.assertEqual(f["/entry/meta/source_entry_path"][()].decode(), "/raw_data")
                self.assertEqual(f["/entry/meta/mask_convention"][()].decode(), "1=masked, 0=valid")
                self.assertAlmostEqual(float(f["/entry/configuration/wavelength"][()]), 6.0)
                self.assertAlmostEqual(float(f["/entry/configuration/sample_detector_distance"][()]), 4.2)


if __name__ == "__main__":
    unittest.main()
