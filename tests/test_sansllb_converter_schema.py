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

        schema = load_schema("scarlet_nxsas_raw_v1.0.yaml")
        report = validate_nexus_file(out, schema)
        self.assertTrue(report.ok, "\n".join(report.format_lines()))

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
            max_i = max(slit_idxs[-1] if slit_idxs else -1, guide_idxs[-1] if guide_idxs else -1)

            expected: list[str] = []
            for i in range(max_i + 1):
                if f"slit{i}" in col:
                    expected.append(f"slit{i}")
                if f"guide{i}" in col:
                    expected.append(f"guide{i}")

        with h5py.File(out, "r") as fout:
            got = [x.decode() if isinstance(x, (bytes, bytearray)) else str(x) for x in fout["/entry/instrument/collimation/element_order"][()]]

        self.assertEqual(got, expected)
