from __future__ import annotations

from pathlib import Path
import tempfile
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

        with tempfile.TemporaryDirectory(prefix="scarlet_processed_") as td:
            processed = Path(td) / "processed_data"
            processed.mkdir(parents=True, exist_ok=True)
            out = processed / "sans-llb2025n002339_scarlet_nxsas_raw.h5"

            convert_sansllb_to_scarlet_nxsas_raw(sample, out, overwrite=True)

            schema = load_schema("scarlet_nxsas_raw_v1.0.yaml")
            report = validate_nexus_file(out, schema)
            self.assertTrue(report.ok, "\n".join(report.format_lines()))
