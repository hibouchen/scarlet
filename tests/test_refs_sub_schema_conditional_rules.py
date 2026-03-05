from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import h5py  # noqa: F401
except Exception:  # pragma: no cover
    h5py = None  # type: ignore[assignment]


from scarlet.validation.schema_loader import load_schema
from scarlet.validation.schema_validator import validate_nexus_file


@unittest.skipIf(h5py is None, "h5py not available")
class TestRefsSubSchemaConditionalRules(unittest.TestCase):
    def test_conditional_rules_require_fields_by_nx_class(self) -> None:
        schema = load_schema("scarlet_refs_sub_v1.0.yaml")

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "refs_sub.nxs"

            with h5py.File(p, "w") as f:
                entry = f.create_group("entry")
                entry.attrs["NX_class"] = b"NXentry"

                entry.create_dataset("definition", data=b"SCARLET_refs_sub")
                entry.create_dataset("schema_version", data=b"1.0")
                entry.create_dataset("config_id", data=b"cfg-1")

                cfg = entry.create_group("configuration")
                cfg.create_dataset("wavelength", data=6.0)
                cfg.create_dataset("sample_detector_distance", data=4.0)

                coll = cfg.create_group("collimation")
                coll.create_dataset("collimation_distance", data=6.0)
                coll.create_dataset("last_aperture_to_sample_distance", data=1.0)

                ap1 = coll.create_group("aperture1")
                ap1.attrs["NX_class"] = b"NXslit"
                # Missing x_gap/y_gap on purpose (should be required by conditional_rules)

                ap2 = coll.create_group("aperture2")
                ap2.attrs["NX_class"] = b"NXpinhole"
                ap2.create_dataset("diameter", data=0.01)

                refs = entry.create_group("references")
                ebt = refs.create_group("empty_beam_transmission")
                ebt.create_group("entry")

                troi = entry.create_group("transmission_roi")
                troi.create_dataset("detector", data=b"detector0")
                troi.create_dataset("roi_type", data=b"rectangle")
                troi.create_dataset("x0", data=0)
                troi.create_dataset("x1", data=10)
                troi.create_dataset("y0", data=0)
                troi.create_dataset("y1", data=10)

                meta = entry.create_group("meta")
                meta.create_dataset("created_utc", data=b"2026-03-05T00:00:00Z")
                meta.create_dataset("mask_convention", data=b"1=masked, 0=valid")

            report = validate_nexus_file(p, schema)
            self.assertFalse(report.ok)

            paths = [m.path for m in report.errors]
            self.assertIn("/entry/configuration/collimation/aperture1/x_gap", paths)
            self.assertIn("/entry/configuration/collimation/aperture1/y_gap", paths)

