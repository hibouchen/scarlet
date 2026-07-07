from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import h5py  # noqa: F401
    import numpy as np
except Exception:  # pragma: no cover
    h5py = None  # type: ignore[assignment]


from scarlet.validation.schema_loader import load_schema
from scarlet.validation.schema_validator import validate_nexus_file


@unittest.skipIf(h5py is None, "h5py/numpy not available")
class TestWorkflowContextSchema(unittest.TestCase):
    def test_workflow_context_schema_accepts_current_layout(self) -> None:
        schema = load_schema("scarlet_workflow_context_v1.0.yaml")

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "workflow_context.nxs"

            with h5py.File(p, "w") as f:
                entry = f.create_group("entry")
                entry.attrs["NX_class"] = b"NXentry"
                entry.create_dataset("definition", data=b"SCARLET_workflow_context")
                entry.create_dataset("schema_version", data=b"1.0")

                metadata = entry.create_group("metadata")
                metadata.create_dataset("experiment_id", data=b"experiment")
                metadata.create_dataset("instrument_name", data=b"sansllb")
                metadata.create_dataset("root_dir", data=b"/data/raw")
                metadata.create_dataset("output_dir", data=b"/data/out")
                metadata.create_dataset("created_utc", data=b"2026-07-07T00:00:00Z")

                runs = entry.create_group("runs")
                runs.create_dataset("sample_name", data=np.array([b"sample_a"], dtype="S32"))
                runs.create_dataset("config_id", data=np.array([b"config_1"], dtype="S32"))
                runs.create_dataset("mode", data=np.array([b"scattering"], dtype="S32"))
                runs.create_dataset("entity", data=np.array([b"sample"], dtype="S32"))
                runs.create_dataset("duplicate_index", data=np.array([0], dtype=np.int64))
                runs.create_dataset("file_path", data=np.array([b"/data/out/sample_a.nxs"], dtype="S128"))

                configurations = entry.create_group("configurations")
                config = configurations.create_group("config_1")
                config.create_dataset("wavelength", data=6.0)
                config.create_dataset("sample_detector_distance", data=np.array([4.2, 1.8], dtype=np.float64))
                coll = config.create_group("collimation")
                coll.create_dataset("collimation_distance", data=8.5)
                coll.create_dataset("last_aperture_to_sample_distance", data=2.1)
                ap1 = coll.create_group("aperture1")
                ap1.create_dataset("type", data=b"slit")
                ap1.create_dataset("x_gap", data=0.01)
                ap1.create_dataset("y_gap", data=0.02)
                ap2 = coll.create_group("aperture2")
                ap2.create_dataset("type", data=b"pinhole")
                ap2.create_dataset("diameter", data=0.005)

                beam_centers = entry.create_group("beam_centers")
                bc_cfg = beam_centers.create_group("config_1")
                bc_det0 = bc_cfg.create_group("detector0")
                bc_det0.create_dataset("beam_center_x", data=63.5)
                bc_det0.create_dataset("beam_center_y", data=64.5)

                rois = entry.create_group("rois")
                roi_cfg = rois.create_group("config_1")
                roi_cfg.create_dataset("x0", data=100)
                roi_cfg.create_dataset("x1", data=140)
                roi_cfg.create_dataset("y0", data=95)
                roi_cfg.create_dataset("y1", data=135)
                roi_cfg.create_dataset("detector_number", data=0)

                references = entry.create_group("references")
                dark = references.create_group("dark")
                dark.create_dataset("config_1", data=b"/data/out/dark.nxs")
                empty_beam = references.create_group("empty_beam")
                eb_cfg = empty_beam.create_group("config_1")
                eb_cfg.create_dataset("transmission", data=b"/data/out/empty_beam_transmission.nxs")
                eb_cfg.create_dataset("scattering", data=b"/data/out/empty_beam_scattering.nxs")
                empty_cell = references.create_group("empty_cell")
                ec_cfg = empty_cell.create_group("config_1")
                ec_cfg.create_dataset("transmission", data=b"/data/out/empty_cell_transmission.nxs")
                water = references.create_group("water")
                w_cfg = water.create_group("config_1")
                w_cfg.create_dataset("scattering", data=b"/data/out/water_scattering.nxs")
                mask_files = references.create_group("mask_files")
                mask_files.create_dataset("config_1", data=b"/data/out/config_1_masks.nxs")
                flatfields = references.create_group("flatfields")
                flatfields.create_dataset("config_1", data=b"/data/out/flatfield_config_1.nxs")

                transmissions = entry.create_group("transmissions")
                sample_transmissions = transmissions.create_group("sample")
                sample_transmissions.create_dataset("sample_name", data=np.array([b"sample_a"], dtype="S32"))
                sample_transmissions.create_dataset("config_id", data=np.array([b"config_1"], dtype="S32"))
                sample_transmissions.create_dataset("value", data=np.array([0.91], dtype=np.float64))
                empty_cell_transmissions = transmissions.create_group("empty_cell")
                empty_cell_transmissions.create_dataset("config_id", data=np.array([b"config_1"], dtype="S32"))
                empty_cell_transmissions.create_dataset("value", data=np.array([0.95], dtype=np.float64))

                sample_thicknesses = entry.create_group("sample_thicknesses")
                sample_thicknesses_sample = sample_thicknesses.create_group("sample")
                sample_thicknesses_sample.create_dataset("sample_name", data=np.array([b"sample_a"], dtype="S32"))
                sample_thicknesses_sample.create_dataset("config_id", data=np.array([b"config_1"], dtype="S32"))
                sample_thicknesses_sample.create_dataset("value", data=np.array([0.002], dtype=np.float64))

                masks = entry.create_group("masks")
                mask_cfg = masks.create_group("config_1")
                mask_cfg.create_dataset("detector0", data=np.zeros((4, 4), dtype=np.uint8))

                flatfield_sources = entry.create_group("flatfield_sources")
                flatfield_sources.create_dataset("config_2", data=b"config_1")

                stale_flatfields = entry.create_group("stale_flatfields")
                stale_flatfields.create_dataset("config_3", data=np.bool_(True))

                artifacts = entry.create_group("artifacts")
                artifacts.create_dataset("name", data=np.array([b"sample_a.nxs"], dtype="S64"))
                artifacts.create_dataset("path", data=np.array([b"/data/out/sample_a.nxs"], dtype="S128"))
                artifacts.create_dataset("kind", data=np.array([b"nexus"], dtype="S16"))
                artifacts.create_dataset("created_utc", data=np.array([b"2026-07-07T00:00:00Z"], dtype="S32"))

                logs = entry.create_group("logs")
                logs.create_dataset("level", data=np.array([b"INFO"], dtype="S16"))
                logs.create_dataset("message", data=np.array([b"created workflow"], dtype="S64"))
                logs.create_dataset("where", data=np.array([b"test"], dtype="S32"))
                logs.create_dataset("when_utc", data=np.array([b"2026-07-07T00:00:00Z"], dtype="S32"))
                logs.create_dataset("meta_json", data=np.array([b"{}"], dtype="S16"))

                issues = entry.create_group("issues")
                issues.create_dataset("level", data=np.array([b"WARN"], dtype="S16"))
                issues.create_dataset("message", data=np.array([b"example warning"], dtype="S64"))
                issues.create_dataset("where", data=np.array([b"test"], dtype="S32"))
                issues.create_dataset("key", data=np.array([b"config_1"], dtype="S32"))
                issues.create_dataset("when_utc", data=np.array([b"2026-07-07T00:00:00Z"], dtype="S32"))
                issues.create_dataset("meta_json", data=np.array([b"{}"], dtype="S16"))

                timings = entry.create_group("timings")
                timings.create_dataset("initialize", data=0.42)

                store = entry.create_group("store")
                store.create_dataset("converted_data_dir", data=b"{\"__type__\": \"path\", \"value\": \"/data/out\"}")

            report = validate_nexus_file(p, schema)
            self.assertTrue(report.ok, "\n".join(report.format_lines()))

    def test_workflow_context_schema_requires_duplicate_index(self) -> None:
        schema = load_schema("scarlet_workflow_context_v1.0.yaml")

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "workflow_context_missing_duplicate_index.nxs"

            with h5py.File(p, "w") as f:
                entry = f.create_group("entry")
                entry.attrs["NX_class"] = b"NXentry"
                entry.create_dataset("definition", data=b"SCARLET_workflow_context")
                entry.create_dataset("schema_version", data=b"1.0")

                metadata = entry.create_group("metadata")
                metadata.create_dataset("experiment_id", data=b"experiment")
                metadata.create_dataset("instrument_name", data=b"sansllb")
                metadata.create_dataset("root_dir", data=b"/data/raw")
                metadata.create_dataset("output_dir", data=b"/data/out")
                metadata.create_dataset("created_utc", data=b"2026-07-07T00:00:00Z")

                runs = entry.create_group("runs")
                runs.create_dataset("sample_name", data=np.array([b"sample_a"], dtype="S32"))
                runs.create_dataset("config_id", data=np.array([b"config_1"], dtype="S32"))
                runs.create_dataset("mode", data=np.array([b"scattering"], dtype="S32"))
                runs.create_dataset("entity", data=np.array([b"sample"], dtype="S32"))
                runs.create_dataset("file_path", data=np.array([b"/data/out/sample_a.nxs"], dtype="S128"))

                entry.create_group("configurations")
                entry.create_group("transmissions")
                entry.create_group("masks")
                entry.create_group("artifacts")
                entry["artifacts"].create_dataset("name", data=np.array([], dtype="S1"))
                entry["artifacts"].create_dataset("path", data=np.array([], dtype="S1"))
                entry["artifacts"].create_dataset("kind", data=np.array([], dtype="S1"))
                entry["artifacts"].create_dataset("created_utc", data=np.array([], dtype="S1"))
                entry.create_group("logs")
                entry["logs"].create_dataset("level", data=np.array([], dtype="S1"))
                entry["logs"].create_dataset("message", data=np.array([], dtype="S1"))
                entry["logs"].create_dataset("where", data=np.array([], dtype="S1"))
                entry["logs"].create_dataset("when_utc", data=np.array([], dtype="S1"))
                entry["logs"].create_dataset("meta_json", data=np.array([], dtype="S1"))
                entry.create_group("issues")
                entry["issues"].create_dataset("level", data=np.array([], dtype="S1"))
                entry["issues"].create_dataset("message", data=np.array([], dtype="S1"))
                entry["issues"].create_dataset("where", data=np.array([], dtype="S1"))
                entry["issues"].create_dataset("key", data=np.array([], dtype="S1"))
                entry["issues"].create_dataset("when_utc", data=np.array([], dtype="S1"))
                entry["issues"].create_dataset("meta_json", data=np.array([], dtype="S1"))
                entry.create_group("timings")
                entry.create_group("store")

            report = validate_nexus_file(p, schema)
            self.assertFalse(report.ok)
            self.assertIn("/entry/runs/duplicate_index", [message.path for message in report.errors])
