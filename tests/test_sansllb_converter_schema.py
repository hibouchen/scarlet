from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

try:
    import h5py  # noqa: F401
    import numpy
except Exception:  # pragma: no cover
    h5py = None  # type: ignore[assignment]


from scarlet.io.converters.sansllb import convert_sansllb_to_scarlet_nxsas_raw
from scarlet.reduction import correct_detector_dead_time
from scarlet.validation.schema_loader import load_schema
from scarlet.validation.schema_validator import validate_nexus_file


@unittest.skipIf(h5py is None, "h5py/numpy not available")
class TestSansLlbConverterSchema(unittest.TestCase):
    @staticmethod
    def _write_minimal_sansllb_input(
        input_path: Path,
        *,
        sample_mask_shape: str = "square",
        sample_mask_size: float = 8.0,
        sample_mask_size_y: float | None = 8.0,
    ) -> None:
        raw0 = numpy.array([[10.0, 20.0], [30.0, 40.0]], dtype=numpy.float64)
        with h5py.File(input_path, "w") as fin:
            entry = fin.create_group("entry0")
            entry.attrs["NX_class"] = b"NXentry"

            sample = entry.create_group("sample")
            sample.attrs["NX_class"] = b"NXsample"
            sample.create_dataset("name", data=numpy.bytes_("sample"))

            control = entry.create_group("control")
            control.attrs["NX_class"] = b"NXmonitor"
            control.create_dataset("count_time", data=10.0)

            monitor2 = entry.create_group("monitor2")
            monitor2.attrs["NX_class"] = b"NXmonitor"
            monitor2.create_dataset("integral", data=500.0)

            instrument = entry.create_group("SANS-LLB")
            instrument.attrs["NX_class"] = b"NXinstrument"

            source = instrument.create_group("source")
            incident_wavelength = source.create_dataset("incident_wavelength", data=0.6)
            incident_wavelength.attrs["units"] = b"nm"

            aperture = instrument.create_group("aperture")
            x_gap = aperture.create_dataset("x_gap", data=0.01)
            x_gap.attrs["units"] = b"m"
            y_gap = aperture.create_dataset("y_gap", data=0.01)
            y_gap.attrs["units"] = b"m"

            sample_mask = instrument.create_group("sample_mask")
            sample_mask.create_dataset("shape", data=numpy.array([numpy.bytes_(sample_mask_shape)]))
            size = sample_mask.create_dataset("size", data=numpy.array([sample_mask_size], dtype=numpy.float64))
            size.attrs["units"] = b"mm"
            if sample_mask_size_y is not None:
                size_y = sample_mask.create_dataset(
                    "size_y", data=numpy.array([sample_mask_size_y], dtype=numpy.float64)
                )
                size_y.attrs["units"] = b"mm"

            collimator = instrument.create_group("collimator")
            collimator_length = collimator.create_dataset("length", data=1.0)
            collimator_length.attrs["units"] = b"m"
            collimator_distance = collimator.create_dataset("distance", data=0.2)
            collimator_distance.attrs["units"] = b"m"
            for slit_name, distance in (("slit0", 1.2), ("slit1", 0.2)):
                slit = collimator.create_group(slit_name)
                slit_x_gap = slit.create_dataset("x_gap", data=0.01)
                slit_x_gap.attrs["units"] = b"m"
                slit_y_gap = slit.create_dataset("y_gap", data=0.01)
                slit_y_gap.attrs["units"] = b"m"
                slit_distance = slit.create_dataset("distance", data=distance)
                slit_distance.attrs["units"] = b"m"

            central_detector = instrument.create_group("central_detector")
            central_detector.attrs["NX_class"] = b"NXdetector"
            central_detector.create_dataset("data", data=raw0)
            central_detector.create_dataset("x_pixel_size", data=0.005)
            central_detector.create_dataset("y_pixel_size", data=0.005)
            central_detector.create_dataset("beam_center_x", data=0.5)
            central_detector.create_dataset("beam_center_y", data=0.5)
            central_detector.create_dataset("dead_time", data=1.0e-3)

    def test_sansllb_converter_maps_square_sample_mask_to_aperture2_slit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            input_path = Path(td) / "sansllb_square_mask.hdf"
            output_path = Path(td) / "sansllb_square_mask_out.h5"
            self._write_minimal_sansllb_input(input_path, sample_mask_shape="square", sample_mask_size=8.0, sample_mask_size_y=6.0)

            convert_sansllb_to_scarlet_nxsas_raw(input_path, output_path, overwrite=True)

            with h5py.File(output_path, "r") as fout:
                ap2 = fout["/raw_data/instrument/collimation/aperture2"]
                self.assertEqual(ap2.attrs["NX_class"], b"NXslit")
                self.assertAlmostEqual(float(ap2["x_gap"][()]), 0.008)
                self.assertAlmostEqual(float(ap2["y_gap"][()]), 0.006)
                self.assertEqual(ap2["x_gap"].attrs["units"], b"m")
                self.assertEqual(ap2["y_gap"].attrs["units"], b"m")

    def test_sansllb_converter_maps_circle_sample_mask_to_aperture2_pinhole(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            input_path = Path(td) / "sansllb_circle_mask.hdf"
            output_path = Path(td) / "sansllb_circle_mask_out.h5"
            self._write_minimal_sansllb_input(input_path, sample_mask_shape="circle", sample_mask_size=12.0, sample_mask_size_y=12.0)

            convert_sansllb_to_scarlet_nxsas_raw(input_path, output_path, overwrite=True)

            with h5py.File(output_path, "r") as fout:
                ap2 = fout["/raw_data/instrument/collimation/aperture2"]
                self.assertEqual(ap2.attrs["NX_class"], b"NXpinhole")
                self.assertAlmostEqual(float(ap2["diameter"][()]), 0.012)
                self.assertEqual(ap2["diameter"].attrs["units"], b"m")

    def test_sansllb_converter_uses_collimator_length_for_collimation_distance(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            input_path = Path(td) / "sansllb_collimation_length.hdf"
            output_path = Path(td) / "sansllb_collimation_length_out.h5"
            self._write_minimal_sansllb_input(input_path)

            with h5py.File(input_path, "a") as fin:
                length = fin["/entry0/SANS-LLB/collimator/length"]
                length[...] = 8_000.0
                length.attrs["units"] = b"mm"

            convert_sansllb_to_scarlet_nxsas_raw(input_path, output_path, overwrite=True)

            with h5py.File(output_path, "r") as fout:
                self.assertAlmostEqual(float(fout["/raw_data/instrument/collimation/collimation_distance"][()]), 8.0)
                self.assertEqual(fout["/raw_data/instrument/collimation/collimation_distance"].attrs["units"], b"m")

    def test_sansllb_converter_deadtime_corrects_all_detector_views(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            input_path = Path(td) / "sansllb_input.hdf"
            output_path = Path(td) / "sansllb_output.h5"
            raw0 = numpy.array([[10.0, 20.0], [30.0, 40.0]], dtype=numpy.float64)
            raw1 = numpy.array([[5.0, 15.0], [25.0, 35.0]], dtype=numpy.float64)

            with h5py.File(input_path, "w") as fin:
                entry = fin.create_group("entry0")
                entry.attrs["NX_class"] = b"NXentry"

                sample = entry.create_group("sample")
                sample.attrs["NX_class"] = b"NXsample"
                sample.create_dataset("name", data=numpy.bytes_("sample"))

                control = entry.create_group("control")
                control.attrs["NX_class"] = b"NXmonitor"
                control.create_dataset("count_time", data=10.0)

                monitor2 = entry.create_group("monitor2")
                monitor2.attrs["NX_class"] = b"NXmonitor"
                monitor2.create_dataset("integral", data=500.0)

                instrument = entry.create_group("SANS-LLB")
                instrument.attrs["NX_class"] = b"NXinstrument"

                source = instrument.create_group("source")
                incident_wavelength = source.create_dataset("incident_wavelength", data=0.6)
                incident_wavelength.attrs["units"] = b"nm"

                aperture = instrument.create_group("aperture")
                x_gap = aperture.create_dataset("x_gap", data=0.01)
                x_gap.attrs["units"] = b"m"
                y_gap = aperture.create_dataset("y_gap", data=0.01)
                y_gap.attrs["units"] = b"m"

                sample_mask = instrument.create_group("sample_mask")
                sample_mask.create_dataset("shape", data=numpy.array([numpy.bytes_("square")]))
                size = sample_mask.create_dataset("size", data=numpy.array([8.0], dtype=numpy.float64))
                size.attrs["units"] = b"mm"
                size_y = sample_mask.create_dataset("size_y", data=numpy.array([8.0], dtype=numpy.float64))
                size_y.attrs["units"] = b"mm"

                collimator = instrument.create_group("collimator")
                collimator_length = collimator.create_dataset("length", data=1.0)
                collimator_length.attrs["units"] = b"m"
                collimator_distance = collimator.create_dataset("distance", data=0.2)
                collimator_distance.attrs["units"] = b"m"
                for slit_name, distance in (("slit0", 1.2), ("slit1", 0.2)):
                    slit = collimator.create_group(slit_name)
                    slit_x_gap = slit.create_dataset("x_gap", data=0.01)
                    slit_x_gap.attrs["units"] = b"m"
                    slit_y_gap = slit.create_dataset("y_gap", data=0.01)
                    slit_y_gap.attrs["units"] = b"m"
                    slit_distance = slit.create_dataset("distance", data=distance)
                    slit_distance.attrs["units"] = b"m"

                central_detector = instrument.create_group("central_detector")
                central_detector.attrs["NX_class"] = b"NXdetector"
                central_detector.create_dataset("data", data=raw0)
                central_detector.create_dataset("x_pixel_size", data=0.005)
                central_detector.create_dataset("y_pixel_size", data=0.005)
                central_detector.create_dataset("beam_center_x", data=0.5)
                central_detector.create_dataset("beam_center_y", data=0.5)
                central_detector.create_dataset("dead_time", data=1.0e-3)

                left_detector = instrument.create_group("left_detector")
                left_detector.attrs["NX_class"] = b"NXdetector"
                left_detector.create_dataset("data", data=raw1)
                left_detector.create_dataset("x_pixel_size", data=0.005)
                left_detector.create_dataset("y_pixel_size", data=0.005)
                left_detector.create_dataset("beam_center_x", data=0.5)
                left_detector.create_dataset("beam_center_y", data=0.5)
                left_detector.create_dataset("deadtime", data=2.0e-3)

            convert_sansllb_to_scarlet_nxsas_raw(input_path, output_path, overwrite=True)

            expected0 = correct_detector_dead_time(raw0, acq_time=10.0, deadtime=1.0e-3)
            expected1 = correct_detector_dead_time(raw1, acq_time=10.0, deadtime=2.0e-3)

            with h5py.File(output_path, "r") as fout:
                numpy.testing.assert_allclose(fout["/raw_data/instrument/detector0/data"][()], expected0)
                numpy.testing.assert_allclose(fout["/raw_data/data0/counts"][()], expected0)
                numpy.testing.assert_allclose(fout["/raw_data/instrument/detector1/data"][()], expected1)
                numpy.testing.assert_allclose(fout["/raw_data/data1/counts"][()], expected1)
                self.assertTrue(bool(fout["/raw_data/instrument/detector0/deadtime_corrected"][()]))
                self.assertTrue(bool(fout["/raw_data/instrument/detector1/deadtime_corrected"][()]))

    def test_sansllb_converter_accepts_vector_count_time_and_empty_monitor_modes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            input_path = Path(td) / "sansllb_input_vector_count_time.hdf"
            output_path = Path(td) / "sansllb_output_vector_count_time.h5"
            raw0 = numpy.array([[10.0, 20.0], [30.0, 40.0]], dtype=numpy.float64)
            count_times = numpy.array([2.0, 3.5, 4.5], dtype=numpy.float64)

            with h5py.File(input_path, "w") as fin:
                entry = fin.create_group("entry0")
                entry.attrs["NX_class"] = b"NXentry"

                sample = entry.create_group("sample")
                sample.attrs["NX_class"] = b"NXsample"
                sample.create_dataset("name", data=numpy.bytes_("sample"))

                control = entry.create_group("control")
                control.attrs["NX_class"] = b"NXmonitor"
                control.create_dataset("count_time", data=count_times)
                control.create_dataset("mode", data=numpy.array([numpy.bytes_("")]))

                for monitor_name, integral in (
                    ("monitor0", numpy.array([100.0, 101.0, 102.0], dtype=numpy.float64)),
                    ("monitor1", numpy.array([200.0, 201.0, 202.0], dtype=numpy.float64)),
                    ("monitor2", numpy.array([300.0, 301.0, 302.0], dtype=numpy.float64)),
                ):
                    monitor = entry.create_group(monitor_name)
                    monitor.attrs["NX_class"] = b"NXmonitor"
                    monitor.create_dataset("mode", data=numpy.array([numpy.bytes_("")]))
                    monitor.create_dataset("integral", data=integral)

                instrument = entry.create_group("SANS-LLB")
                instrument.attrs["NX_class"] = b"NXinstrument"

                source = instrument.create_group("source")
                incident_wavelength = source.create_dataset("incident_wavelength", data=0.6)
                incident_wavelength.attrs["units"] = b"nm"

                aperture = instrument.create_group("aperture")
                x_gap = aperture.create_dataset("x_gap", data=0.01)
                x_gap.attrs["units"] = b"m"
                y_gap = aperture.create_dataset("y_gap", data=0.01)
                y_gap.attrs["units"] = b"m"

                sample_mask = instrument.create_group("sample_mask")
                sample_mask.create_dataset("shape", data=numpy.array([numpy.bytes_("square")]))
                size = sample_mask.create_dataset("size", data=numpy.array([8.0], dtype=numpy.float64))
                size.attrs["units"] = b"mm"
                size_y = sample_mask.create_dataset("size_y", data=numpy.array([8.0], dtype=numpy.float64))
                size_y.attrs["units"] = b"mm"

                collimator = instrument.create_group("collimator")
                collimator_length = collimator.create_dataset("length", data=1.0)
                collimator_length.attrs["units"] = b"m"
                collimator_distance = collimator.create_dataset("distance", data=0.2)
                collimator_distance.attrs["units"] = b"m"
                for slit_name, distance in (("slit0", 1.2), ("slit1", 0.2)):
                    slit = collimator.create_group(slit_name)
                    slit_x_gap = slit.create_dataset("x_gap", data=0.01)
                    slit_x_gap.attrs["units"] = b"m"
                    slit_y_gap = slit.create_dataset("y_gap", data=0.01)
                    slit_y_gap.attrs["units"] = b"m"
                    slit_distance = slit.create_dataset("distance", data=distance)
                    slit_distance.attrs["units"] = b"m"

                central_detector = instrument.create_group("central_detector")
                central_detector.attrs["NX_class"] = b"NXdetector"
                central_detector.create_dataset("data", data=raw0)
                central_detector.create_dataset("x_pixel_size", data=0.005)
                central_detector.create_dataset("y_pixel_size", data=0.005)
                central_detector.create_dataset("beam_center_x", data=0.5)
                central_detector.create_dataset("beam_center_y", data=0.5)
                central_detector.create_dataset("dead_time", data=1.0e-3)

            convert_sansllb_to_scarlet_nxsas_raw(input_path, output_path, overwrite=True)

            expected0 = correct_detector_dead_time(raw0, acq_time=float(count_times.sum()), deadtime=1.0e-3)
            schema = load_schema("scarlet_nxsas_raw_v1.3_mono.yaml")
            report = validate_nexus_file(output_path, schema)

            self.assertTrue(report.ok, "\n".join(report.format_lines()))
            with h5py.File(output_path, "r") as fout:
                numpy.testing.assert_allclose(fout["/raw_data/instrument/detector0/data"][()], expected0)
                self.assertAlmostEqual(float(fout["/raw_data/control/preset"][()]), 903.0)
                self.assertAlmostEqual(float(fout["/raw_data/control/integral"][()]), 903.0)
                self.assertAlmostEqual(float(fout["/raw_data/control/count_time"][()]), float(count_times.sum()))
                self.assertEqual(fout["/raw_data/instrument/monitor0/mode"][()].decode(), "monitor")
                self.assertEqual(fout["/raw_data/instrument/monitor1/mode"][()].decode(), "monitor")
                self.assertEqual(fout["/raw_data/instrument/monitor2/mode"][()].decode(), "monitor")
                self.assertAlmostEqual(float(fout["/raw_data/instrument/monitor0/integral"][()]), 303.0)
                self.assertAlmostEqual(float(fout["/raw_data/instrument/monitor1/integral"][()]), 603.0)
                self.assertAlmostEqual(float(fout["/raw_data/instrument/monitor2/integral"][()]), 903.0)

    def test_sansllb_sample_validates_with_schema(self) -> None:
        raw_data = Path(__file__).resolve().parent / "data" / "sansllb" / "raw_data"
        sample = raw_data / "sans-llb2025n002339.hdf"
        if not sample.exists():
            self.skipTest(f"Missing test input file: {sample}")

        # Write output to a persistent location for manual inspection.
        # Override with SCARLET_TEST_OUTPUT_DIR=/path/to/dir if desired.
        root = Path(__file__).resolve().parent.parent
        processed = Path(os.environ.get("SCARLET_TEST_OUTPUT_DIR", root / "data" / "SANSLLB" / "processed"))
        processed.mkdir(parents=True, exist_ok=True)
        out = processed / "sans-llb2025n002339_scarlet_nxsas_raw.h5"

        convert_sansllb_to_scarlet_nxsas_raw(sample, out, overwrite=True)

        schema = load_schema("scarlet_nxsas_raw_v1.3_mono.yaml")
        report = validate_nexus_file(out, schema)
        self.assertTrue(report.ok, "\n".join(report.format_lines()))

        with h5py.File(sample, "r") as fin, h5py.File(out, "r") as fout:
            entry = "/entry0" if "/entry0" in fin else ("/entry" if "/entry" in fin else "/entry1")
            preset_expected = float(fin[f"{entry}/monitor2/integral"][()].reshape(()))
            wavelength_expected = float(fin[f"{entry}/SANS-LLB/velocity_selector/wavelength"][()].reshape(())) * 10.0
            self.assertEqual(fout["/raw_data/control/mode"][()].decode(), "monitor")
            self.assertEqual(float(fout["/raw_data/control/preset"][()]), preset_expected)
            self.assertEqual(float(fout["/raw_data/control/integral"][()]), preset_expected)
            self.assertAlmostEqual(float(fout["/raw_data/instrument/monochromator/wavelength"][()]), wavelength_expected)
            self.assertIn("/raw_data/instrument/collimation/aperture1", fout)
            self.assertIn("/raw_data/instrument/collimation/aperture2", fout)
            self.assertEqual(float(fout["/raw_data/instrument/detector0/x_pixel_size"][()]), 0.005)
            self.assertEqual(fout["/raw_data/instrument/detector0/x_pixel_size"].attrs["units"], b"m")
            self.assertEqual(fout["/raw_data/instrument/detector0/beam_center_x"].shape, ())
            self.assertEqual(fout["/raw_data/instrument/detector0/beam_center_y"].shape, ())
            for detector_name, data_view, detector_index in (
                ("left_detector", "left_data", 1),
                ("bottom_detector", "bottom_data", 2),
            ):
                data_group = fin[f"{entry}/{data_view}"]
                for axis in ("x", "y"):
                    coord = numpy.asarray(data_group[axis][()], dtype=float).reshape(-1)
                    pos = numpy.asarray(data_group[f"pos{axis}"][()], dtype=float).reshape(-1)
                    slope, intercept = numpy.polyfit(coord, pos, 1)
                    expected = float(-intercept / slope)
                    actual = float(fout[f"/raw_data/instrument/detector{detector_index}/beam_center_{axis}"][()])
                    self.assertAlmostEqual(actual, expected)
            self.assertEqual(fout["/raw_data/instrument/collimation/collimation_distance"].attrs["units"], b"m")
            self.assertEqual(fout["/raw_data/instrument/collimation/aperture2"].attrs["NX_class"], b"NXslit")
            expected_ap2_gap = 0.01
            sample_mask_path = f"{entry}/SANS-LLB/sample_mask"
            if sample_mask_path in fin:
                size_ds = fin[f"{sample_mask_path}/size"]
                expected_ap2_gap = float(size_ds[()].reshape(-1)[0])
                units = size_ds.attrs.get("units")
                if isinstance(units, (bytes, bytearray)):
                    units = units.decode(errors="replace")
                if str(units).lower() == "mm":
                    expected_ap2_gap /= 1000.0
            self.assertEqual(float(fout["/raw_data/instrument/collimation/aperture2/x_gap"][()]), expected_ap2_gap)
            self.assertEqual(fout["/raw_data/instrument/collimation/aperture2/x_gap"].attrs["units"], b"m")
            self.assertEqual(float(fout["/raw_data/instrument/collimation/aperture2/y_gap"][()]), expected_ap2_gap)

            expected_monitors = sorted(
                key for key in fin[entry].keys() if key.startswith("monitor") and isinstance(fin[f"{entry}/{key}"], h5py.Group)
            )
            got_monitors = sorted(key for key in fout["/raw_data/instrument"].keys() if key.startswith("monitor"))
            self.assertEqual(got_monitors, expected_monitors)

            for name in expected_monitors:
                if "integral" in fin[f"{entry}/{name}"]:
                    integral_expected = float(fin[f"{entry}/{name}/integral"][()].reshape(()))
                    self.assertEqual(float(fout[f"/raw_data/instrument/{name}/integral"][()]), integral_expected)

        # Collimation order: slit0, guide0, slit1, guide1, ...
        with h5py.File(sample, "r") as fin:
            entry = "/entry0" if "/entry0" in fin else ("/entry" if "/entry" in fin else "/entry1")
            inst = None
            for k, obj in fin[entry].items():
                if isinstance(obj, h5py.Group) and obj.attrs.get("NX_class", b"") == b"NXinstrument":
                    inst = f"{entry}/{k}"
                    break
            self.assertIsNotNone(inst, "No NXinstrument group found in input file")
            col = fin[f"{inst}/collimator"]

            def idx(prefix: str, name: str) -> int | None:
                if not name.startswith(prefix):
                    return None
                tail = name[len(prefix) :]
                return int(tail) if tail.isdigit() else None

            slit_idxs = sorted(i for i in (idx("slit", k) for k in col.keys()) if i is not None)
            guide_idxs = sorted(i for i in (idx("guide", k) for k in col.keys()) if i is not None)

            max_guide = guide_idxs[-1] if guide_idxs else None
            if max_guide is not None:
                slit_idxs = [i for i in slit_idxs if i <= max_guide + 1]

            expected: list[str] = []
            if max_guide is not None:
                for i in range(max_guide + 1):
                    if i in slit_idxs and f"slit{i}" in col:
                        expected.append(f"slit{i}")
                    if i in guide_idxs and f"guide{i}" in col:
                        expected.append(f"guide{i}")
                last_slit = max_guide + 1
                if last_slit in slit_idxs and f"slit{last_slit}" in col:
                    expected.append(f"slit{last_slit}")
            else:
                max_i = max(slit_idxs[-1] if slit_idxs else -1, guide_idxs[-1] if guide_idxs else -1)
                for i in range(max_i + 1):
                    if i in slit_idxs and f"slit{i}" in col:
                        expected.append(f"slit{i}")
                    if i in guide_idxs and f"guide{i}" in col:
                        expected.append(f"guide{i}")

        with h5py.File(out, "r") as fout:
            got = [x.decode() if isinstance(x, (bytes, bytearray)) else str(x) for x in fout["/raw_data/instrument/collimation/element_order"][()]]

        self.assertEqual(got, expected)

        with h5py.File(sample, "r") as fin, h5py.File(out, "r") as fout:
            entry = "/entry0" if "/entry0" in fin else ("/entry" if "/entry" in fin else "/entry1")
            inst = None
            for k, obj in fin[entry].items():
                if isinstance(obj, h5py.Group) and obj.attrs.get("NX_class", b"") == b"NXinstrument":
                    inst = f"{entry}/{k}"
                    break
            self.assertIsNotNone(inst, "No NXinstrument group found in input file")
            col = fin[f"{inst}/collimator"]

            for name in expected:
                if not name.startswith("guide"):
                    continue
                gg_in = col[name]
                if "selection" not in gg_in:
                    continue
                sel = gg_in["selection"][()]
                if hasattr(sel, "size") and getattr(sel, "size") == 1:
                    sel = sel.reshape(()).item() if hasattr(sel, "reshape") else sel[0]
                if isinstance(sel, (bytes, bytearray)):
                    sel_s = sel.decode(errors="replace")
                else:
                    sel_s = str(sel)

                state = fout[f"/raw_data/instrument/collimation/elements/{name}/state"][()]
                if isinstance(state, (bytes, bytearray)):
                    state_s = state.decode(errors="replace")
                else:
                    state_s = str(state)

                self.assertNotIn(
                    "selection",
                    fout[f"/raw_data/instrument/collimation/elements/{name}"],
                )

                if sel_s == "ft":
                    self.assertEqual(state_s, "out")
                elif sel_s == "ng":
                    self.assertEqual(state_s, "in")
