from __future__ import annotations

import os
from pathlib import Path
import unittest

try:
    import h5py  # noqa: F401
except Exception:  # pragma: no cover
    h5py = None  # type: ignore[assignment]


from scarlet.io.converters.sam import convert_sam_to_scarlet_nxsas_raw
from scarlet.validation.schema_loader import load_schema
from scarlet.validation.schema_validator import validate_nexus_file


@unittest.skipIf(h5py is None, "h5py not available")
class TestSamConverterSchema(unittest.TestCase):
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
            self.assertEqual(fout["/raw_data/control/mode"][()].decode(), "timer")
            self.assertEqual(float(fout["/raw_data/control/preset"][()]), 30.0)
            self.assertEqual(float(fout["/raw_data/control/integral"][()]), 296271.0)
            self.assertEqual(float(fout["/raw_data/control/count_time"][()]), 30.0)
            self.assertIn("/raw_data/instrument/collimation/aperture1", fout)
            self.assertIn("/raw_data/instrument/collimation/aperture2", fout)
            got = [x.decode() if isinstance(x, (bytes, bytearray)) else str(x) for x in fout["/raw_data/instrument/collimation/element_order"][()]]
            self.assertEqual(got, ["slit1", "guide1", "slit2", "guide2", "slit3", "guide3", "slit4"])
            self.assertEqual(fout["/raw_data/instrument/collimation/elements/guide1/state"][()].decode(), "in")
            self.assertEqual(fout["/raw_data/instrument/collimation/elements/guide2/state"][()].decode(), "in")
            self.assertEqual(fout["/raw_data/instrument/collimation/elements/guide3/state"][()].decode(), "in")
