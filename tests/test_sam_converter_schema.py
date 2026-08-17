from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

try:
    import h5py  # noqa: F401
    import numpy as np
except Exception:  # pragma: no cover
    h5py = None  # type: ignore[assignment]


from scarlet.io.converters.sam import convert_sam_to_scarlet_nxsas_raw
from scarlet.reduction import correct_detector_dead_time
from scarlet.validation.schema_loader import load_schema
from scarlet.validation.schema_validator import validate_nexus_file


@unittest.skipIf(h5py is None, "h5py not available")
class TestSamConverterSchema(unittest.TestCase):
    def test_sam_converter_deadtime_corrects_detector_data(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            input_path = Path(td) / "sam_input.nxs"
            output_path = Path(td) / "sam_output.h5"
            raw_counts = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)

            with h5py.File(input_path, "w") as fin:
                entry = fin.create_group("entry0")
                entry.attrs["NX_class"] = b"NXentry"
                entry.create_dataset("experiment_identifier", data=np.bytes_("EB"))
                entry.create_dataset("title", data=np.bytes_("sample"))
                entry.create_dataset("time", data=10.0)

                monitor = entry.create_group("monitor")
                monitor.attrs["NX_class"] = b"NXmonitor"
                monitor.create_dataset("mode", data=np.bytes_("time"))
                monitor.create_dataset("preset", data=10.0)
                monitor.create_dataset("integral", data=1234.0)

                data1 = entry.create_group("data1")
                data1.attrs["NX_class"] = b"NXdata"
                data1.create_dataset("detector_data", data=raw_counts)

                instrument = entry.create_group("instrument")
                instrument.attrs["NX_class"] = b"NXinstrument"

                beam = instrument.create_group("Beam")
                beam.create_dataset("sample_ap_x_or_diam", data=8.0)
                beam["sample_ap_x_or_diam"].attrs["units"] = np.bytes_("mm")
                beam.create_dataset("sample_ap_y", data=4.0)
                beam["sample_ap_y"].attrs["units"] = np.bytes_("mm")

                selector = instrument.create_group("Selector")
                selector.create_dataset("wavelength", data=6.0)

                distance = instrument.create_group("Distance")
                distance.create_dataset("S2_Sample", data=2.5)

                detector = instrument.create_group("detector")
                detector.create_dataset("pixel_size_x", data=5.0)
                detector.create_dataset("pixel_size_y", data=5.0)
                detector.create_dataset("dead_time", data=1.0e-3)

                collimation = instrument.create_group("collimation")
                collimation.create_dataset("position", data=1.0)

                slits = instrument.create_group("VirtualSlitAxis")
                for idx in range(1, 5):
                    slits.create_dataset(f"s{idx}w_actual_width", data=10.0)
                    slits.create_dataset(f"s{idx}h_actual_width", data=10.0)

            convert_sam_to_scarlet_nxsas_raw(input_path, output_path, overwrite=True)

            expected = correct_detector_dead_time(raw_counts, acq_time=10.0, deadtime=1.0e-3)
            expected_wavelength_error = 0.1 * 6.0

            with h5py.File(output_path, "r") as fout:
                np.testing.assert_allclose(fout["/raw_data/instrument/detector0/data"][()], expected)
                np.testing.assert_allclose(fout["/raw_data/data0/counts"][()], expected)
                self.assertTrue(bool(fout["/raw_data/instrument/detector0/deadtime_corrected"][()]))
                self.assertEqual(fout["/raw_data/sample/name"][()].decode(), "EB")
                self.assertEqual(fout["/raw_data/instrument/collimation/elements/slit4"].attrs["NX_class"], b"NXslit")
                self.assertEqual(float(fout["/raw_data/instrument/collimation/elements/slit4/x_gap"][()]), 0.008)
                self.assertEqual(float(fout["/raw_data/instrument/collimation/elements/slit4/y_gap"][()]), 0.004)
                self.assertEqual(
                    float(fout["/raw_data/instrument/monochromator/wavelength_error"][()]),
                    expected_wavelength_error,
                )

    def test_sam_converter_maps_zero_sample_ap_y_to_slit4_pinhole(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            input_path = Path(td) / "sam_input_data_group.nxs"
            output_path = Path(td) / "sam_output_data_group.h5"
            raw_counts = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)

            with h5py.File(input_path, "w") as fin:
                entry = fin.create_group("entry0")
                entry.attrs["NX_class"] = b"NXentry"
                entry.create_dataset("title", data=np.bytes_("sample"))
                entry.create_dataset("time", data=10.0)

                monitor = entry.create_group("monitor")
                monitor.attrs["NX_class"] = b"NXmonitor"
                monitor.create_dataset("mode", data=np.bytes_("time"))
                monitor.create_dataset("preset", data=10.0)
                monitor.create_dataset("integral", data=1234.0)

                data = entry.create_group("data")
                data.attrs["NX_class"] = b"NXdata"
                data.create_dataset("detector_data1", data=raw_counts[..., np.newaxis])

                instrument = entry.create_group("SAM")
                instrument.attrs["NX_class"] = b"NXinstrument"

                beam = instrument.create_group("Beam")
                beam.create_dataset("sample_ap_x_or_diam", data=7.0)
                beam["sample_ap_x_or_diam"].attrs["units"] = np.bytes_("mm")
                beam.create_dataset("sample_ap_y", data=0.0)
                beam["sample_ap_y"].attrs["units"] = np.bytes_("mm")

                selector = instrument.create_group("Selector")
                selector.create_dataset("wavelength", data=6.0)

                distance = instrument.create_group("Distance")
                distance.create_dataset("S2_Sample", data=2.5)

                detector = instrument.create_group("detector")
                detector.create_dataset("pixel_size_x", data=5.0)
                detector.create_dataset("pixel_size_y", data=5.0)
                detector.create_dataset("dead_time", data=0.0)

                collimation = instrument.create_group("collimation")
                collimation.create_dataset("position", data=1.0)

                slits = instrument.create_group("VirtualSlitAxis")
                for idx in range(1, 5):
                    slits.create_dataset(f"s{idx}w_actual_width", data=10.0)
                    slits.create_dataset(f"s{idx}h_actual_width", data=10.0)

            convert_sam_to_scarlet_nxsas_raw(input_path, output_path, overwrite=True)

            expected = correct_detector_dead_time(raw_counts, acq_time=10.0, deadtime=0.0)
            expected_wavelength_error = 0.1 * 6.0

            with h5py.File(output_path, "r") as fout:
                np.testing.assert_allclose(fout["/raw_data/instrument/detector0/data"][()], expected)
                self.assertEqual(fout["/raw_data/sample/name"][()].decode(), "sample")
                self.assertEqual(fout["/raw_data/instrument/collimation/elements/slit4"].attrs["NX_class"], b"NXpinhole")
                self.assertEqual(float(fout["/raw_data/instrument/collimation/elements/slit4/diameter"][()]), 0.007)
                self.assertEqual(fout["/raw_data/instrument/collimation/aperture2"].attrs["NX_class"], b"NXpinhole")
                self.assertEqual(float(fout["/raw_data/instrument/collimation/aperture2/diameter"][()]), 0.007)
                self.assertEqual(
                    float(fout["/raw_data/instrument/monochromator/wavelength_error"][()]),
                    expected_wavelength_error,
                )

    def test_sam_sample_validates_with_schema(self) -> None:
        root = Path(__file__).resolve().parent.parent
        raw_data = root / "data" / "SAM" / "raw"
        sample = raw_data / "011826.nxs"
        if not sample.exists():
            self.skipTest(f"Missing test input file: {sample}")

        processed = Path(os.environ.get("SCARLET_TEST_OUTPUT_DIR", root / "data" / "SAM" / "processed"))
        processed.mkdir(parents=True, exist_ok=True)
        out = processed / "011826_scarlet_nxsas_raw.h5"

        convert_sam_to_scarlet_nxsas_raw(sample, out, overwrite=True)

        schema = load_schema("scarlet_nxsas_raw_v1.3_mono.yaml")
        report = validate_nexus_file(out, schema)
        self.assertTrue(report.ok, "\n".join(report.format_lines()))

        with h5py.File(out, "r") as fout:
            wavelength = float(fout["/raw_data/instrument/monochromator/wavelength"][()])
            self.assertEqual(float(fout["/raw_data/instrument/monochromator/wavelength_error"][()]), 0.1 * wavelength)
            self.assertEqual(fout["/raw_data/sample/name"][()].decode(), "EB")
            self.assertEqual(fout["/raw_data/control/mode"][()].decode(), "timer")
            self.assertEqual(float(fout["/raw_data/control/preset"][()]), 30.0)
            self.assertEqual(float(fout["/raw_data/control/integral"][()]), 296271.0)
            self.assertEqual(float(fout["/raw_data/control/count_time"][()]), 30.0)
            self.assertIn("/raw_data/instrument/collimation/aperture1", fout)
            self.assertIn("/raw_data/instrument/collimation/aperture2", fout)
            self.assertEqual(fout["/raw_data/instrument/collimation/elements/slit4"].attrs["NX_class"], b"NXpinhole")
            self.assertEqual(float(fout["/raw_data/instrument/collimation/elements/slit4/diameter"][()]), 0.007)
            got = [x.decode() if isinstance(x, (bytes, bytearray)) else str(x) for x in fout["/raw_data/instrument/collimation/element_order"][()]]
            self.assertEqual(got, ["guide1", "slit1", "guide2", "slit2", "guide3", "slit3", "slit4"])
            self.assertEqual(fout["/raw_data/instrument/collimation/elements/guide1/state"][()].decode(), "in")
            self.assertEqual(fout["/raw_data/instrument/collimation/elements/guide2/state"][()].decode(), "in")
            self.assertEqual(fout["/raw_data/instrument/collimation/elements/guide3/state"][()].decode(), "in")
