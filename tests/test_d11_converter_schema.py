from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

try:
    import h5py  # noqa: F401
    import numpy as np
except Exception:  # pragma: no cover
    h5py = None  # type: ignore[assignment]

from scarlet.io.converters.d11 import convert_d11_to_scarlet_nxsas_raw
from scarlet.validation.schema_loader import load_schema
from scarlet.validation.schema_validator import validate_nexus_file


@unittest.skipIf(h5py is None, "h5py not available")
class TestD11ConverterSchema(unittest.TestCase):
    def test_d11_converter_maps_required_fields(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            input_path = Path(td) / "d11_input.nxs"
            output_path = Path(td) / "d11_output.h5"
            raw_counts = np.arange(12, dtype=np.int32).reshape(3, 4, 1)

            with h5py.File(input_path, "w") as fin:
                entry = fin.create_group("entry0")
                entry.attrs["NX_class"] = b"NXentry"
                entry.create_dataset("sample_description", data=np.bytes_("Acb75 73.1D"))
                duration = entry.create_dataset("duration", data=300.0)
                duration.attrs["units"] = np.bytes_("Sec")

                monitor1 = entry.create_group("monitor1")
                monitor1.attrs["NX_class"] = b"NXmonitor"
                monitor1.create_dataset("mode", data=np.bytes_("monitor"))
                monitor1.create_dataset("preset", data=5000.0)
                monitor1.create_dataset("integral", data=3057222.0)

                instrument = entry.create_group("D11")
                instrument.attrs["NX_class"] = b"NXinstrument"

                beam = instrument.create_group("Beam")
                beam.create_dataset("center_x", data=1.5)
                beam.create_dataset("center_y", data=2.5)
                beam.create_dataset("sample_ap_x_or_diam", data=7.0)
                beam["sample_ap_x_or_diam"].attrs["units"] = np.bytes_("mm")
                beam.create_dataset("sample_ap_y", data=10.0)
                beam["sample_ap_y"].attrs["units"] = np.bytes_("mm")

                selector = instrument.create_group("selector")
                selector.create_dataset("wavelength", data=6.0)
                selector.create_dataset("wavelength_res", data=9.0)
                selector.create_dataset("rotation_speed", data=21521.0)

                collimation = instrument.create_group("collimation")
                collimation.create_dataset("actual_position", data=8.0)
                collimation.create_dataset("guide_exit_cross_section_width", data=45.0)
                collimation.create_dataset("guide_exit_cross_section_height", data=50.0)

                detector = instrument.create_group("detector")
                data = detector.create_dataset("data", data=raw_counts)
                data.attrs["target"] = np.bytes_("/entry0/D11/detector/data")
                det_actual = detector.create_dataset("det_actual", data=8.003)
                det_actual.attrs["units"] = np.bytes_("m")

            report = convert_d11_to_scarlet_nxsas_raw(input_path, output_path, overwrite=True)
            self.assertIn("Using D11 monitor1", "\n".join(report.notes))
            self.assertFalse(any("dead" in warning.lower() for warning in report.warnings))

            schema = load_schema("scarlet_nxsas_raw_v1.3_mono.yaml")
            validation = validate_nexus_file(output_path, schema)
            self.assertTrue(validation.ok, "\n".join(validation.format_lines()))

            with h5py.File(output_path, "r") as fout:
                expected_counts = raw_counts[..., 0].astype(np.float64)
                np.testing.assert_allclose(fout["/raw_data/instrument/detector0/data"][()], expected_counts)
                np.testing.assert_allclose(fout["/raw_data/data0/counts"][()], expected_counts)
                self.assertEqual(fout["/raw_data/sample/name"][()].decode(), "Acb75 73.1D")
                self.assertEqual(fout["/raw_data/control/mode"][()].decode(), "monitor")
                self.assertEqual(float(fout["/raw_data/control/preset"][()]), 5000.0)
                self.assertEqual(float(fout["/raw_data/control/integral"][()]), 3057222.0)
                self.assertEqual(float(fout["/raw_data/control/count_time"][()]), 300.0)
                self.assertEqual(float(fout["/raw_data/instrument/monitor1/integral"][()]), 3057222.0)
                self.assertEqual(float(fout["/raw_data/instrument/monochromator/wavelength"][()]), 6.0)
                self.assertEqual(float(fout["/raw_data/instrument/monochromator/wavelength_error"][()]), 0.54)
                self.assertEqual(float(fout["/raw_data/instrument/collimation/collimation_distance"][()]), 8.0)
                self.assertEqual(float(fout["/raw_data/instrument/collimation/aperture2/x_gap"][()]), 0.007)
                self.assertEqual(float(fout["/raw_data/instrument/collimation/aperture2/y_gap"][()]), 0.010)
                self.assertEqual(float(fout["/raw_data/instrument/detector0/x_pixel_size"][()]), 0.005)
                self.assertEqual(float(fout["/raw_data/instrument/detector0/y_pixel_size"][()]), 0.005)
                self.assertEqual(float(fout["/raw_data/instrument/detector0/dead_time"][()]), 0.0)
                self.assertEqual(float(fout["/raw_data/instrument/detector0/beam_center_x"][()]), 1.5)
                self.assertEqual(float(fout["/raw_data/instrument/detector0/beam_center_y"][()]), 2.5)
                self.assertEqual(
                    float(fout["/raw_data/instrument/detector0/transformations/translation"][2]),
                    8.003,
                )

