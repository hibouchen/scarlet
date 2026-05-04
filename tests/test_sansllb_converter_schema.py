from __future__ import annotations

import os
from pathlib import Path
import unittest

try:
    import h5py  # noqa: F401
    import numpy  # noqa: F401
except Exception:  # pragma: no cover
    h5py = None  # type: ignore[assignment]


from scarlet.io.converters.sansllb import convert_sansllb_to_scarlet_nxsas_raw
from scarlet.validation.schema_loader import load_schema
from scarlet.validation.schema_validator import validate_nexus_file


@unittest.skipIf(h5py is None, "h5py/numpy not available")
class TestSansLlbConverterSchema(unittest.TestCase):
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
            self.assertEqual(fout["/entry/control/mode"][()].decode(), "monitor")
            self.assertEqual(float(fout["/entry/control/preset"][()]), preset_expected)
            self.assertEqual(float(fout["/entry/control/integral"][()]), preset_expected)
            self.assertIn("/entry/instrument/collimation/aperture1", fout)
            self.assertIn("/entry/instrument/collimation/aperture2", fout)
            self.assertEqual(float(fout["/entry/instrument/detector0/x_pixel_size"][()]), 0.005)
            self.assertEqual(fout["/entry/instrument/detector0/x_pixel_size"].attrs["units"], b"m")
            self.assertEqual(fout["/entry/instrument/collimation/collimation_distance"].attrs["units"], b"m")
            self.assertEqual(fout["/entry/instrument/collimation/aperture2"].attrs["NX_class"], b"NXslit")
            self.assertEqual(float(fout["/entry/instrument/collimation/aperture2/x_gap"][()]), 0.01)
            self.assertEqual(fout["/entry/instrument/collimation/aperture2/x_gap"].attrs["units"], b"m")
            self.assertEqual(float(fout["/entry/instrument/collimation/aperture2/y_gap"][()]), 0.01)

            expected_monitors = sorted(
                key for key in fin[entry].keys() if key.startswith("monitor") and isinstance(fin[f"{entry}/{key}"], h5py.Group)
            )
            got_monitors = sorted(key for key in fout["/entry/instrument"].keys() if key.startswith("monitor"))
            self.assertEqual(got_monitors, expected_monitors)

            for name in expected_monitors:
                if "integral" in fin[f"{entry}/{name}"]:
                    integral_expected = float(fin[f"{entry}/{name}/integral"][()].reshape(()))
                    self.assertEqual(float(fout[f"/entry/instrument/{name}/integral"][()]), integral_expected)

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
            got = [x.decode() if isinstance(x, (bytes, bytearray)) else str(x) for x in fout["/entry/instrument/collimation/element_order"][()]]

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

                state = fout[f"/entry/instrument/collimation/elements/{name}/state"][()]
                if isinstance(state, (bytes, bytearray)):
                    state_s = state.decode(errors="replace")
                else:
                    state_s = str(state)

                self.assertNotIn(
                    "selection",
                    fout[f"/entry/instrument/collimation/elements/{name}"],
                )

                if sel_s == "ft":
                    self.assertEqual(state_s, "out")
                elif sel_s == "ng":
                    self.assertEqual(state_s, "in")
