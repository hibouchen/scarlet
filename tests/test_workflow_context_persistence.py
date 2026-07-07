from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scarlet.validation.schema_loader import load_schema
from scarlet.validation.schema_validator import validate_nexus_file
from scarlet.workflow.configuration import Aperture, Collimation, Configuration
from scarlet.workflow.context import RunKey, WorkflowContext


class TestWorkflowContextPersistence(unittest.TestCase):
    def test_save_and_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw_dir = root / "raw"
            out_dir = root / "out"
            raw_dir.mkdir()
            out_dir.mkdir()

            sample_path = out_dir / "sample_a.nxs"
            dark_a_path = out_dir / "dark_a.nxs"
            dark_b_path = out_dir / "dark_b.nxs"
            empty_beam_path = out_dir / "empty_beam_transmission.nxs"
            mask_bundle_path = out_dir / "config_1_masks.nxs"
            flatfield_path = out_dir / "flatfield_config_1.nxs"

            ctx = WorkflowContext(
                experiment_id="exp-123",
                instrument_name="sansllb",
                root_dir=raw_dir,
                output_dir=out_dir,
            )
            ctx.configurations["config_1"] = Configuration(
                wavelength=6.0,
                sample_detector_distance=[4.2, 1.8],
                collimation=Collimation(
                    aperture1=Aperture(type="slit", x_gap=0.01, y_gap=0.02),
                    aperture2=Aperture(type="pinhole", diameter=0.005),
                    collimation_distance=8.5,
                    last_aperture_to_sample_distance=2.1,
                ),
                config_id="config_1",
                notes="main config",
            )
            ctx.add_run(
                RunKey(config_id="config_1", entity="sample", mode="scattering", sample_name="sample_a"),
                sample_path,
            )
            ctx.add_run(
                RunKey(config_id="config_1", entity="dark", mode="scattering", sample_name="Cd"),
                dark_a_path,
            )
            ctx.add_run(
                RunKey(config_id="config_1", entity="dark", mode="scattering", sample_name="Cd"),
                dark_b_path,
            )
            ctx.add_run(
                RunKey(config_id="config_1", entity="empty_beam", mode="transmission", sample_name="EmptyBeam"),
                empty_beam_path,
            )
            ctx.set_beam_center("config_1", 0, (63.5, 64.5))
            ctx.set_roi("config_1", (100, 140, 95, 135))
            ctx.set_transmission("sample_a", "config_1", 0.91)
            ctx.set_empty_cell_transmission("config_1", 0.95)
            ctx.set_sample_thickness("sample_a", "config_1", 0.002)
            ctx.set_mask("config_1", 0, np.zeros((4, 4), dtype=np.uint8))
            ctx.mask_files["config_1"] = mask_bundle_path.resolve()
            ctx.set_flatfield("config_1", flatfield_path)
            ctx.set_flatfield_source("config_2", "config_1")
            ctx.stale_flatfields.add("config_3")
            ctx.info("workflow created", where="test_save", step="setup")
            ctx.warn("reference missing", where="test_save", key="config_2", entity="water")
            ctx.timings["initialize"] = 0.42
            ctx.add_artifact("sample_a.nxs", sample_path, kind="nexus")
            ctx.set("converted_data_dir", out_dir)
            ctx.set("roi_cache", {"config_1": (100, 140, 95, 135)})

            workflow_path = root / "workflow_context.nxs"
            saved_path = ctx.save(workflow_path)

            self.assertEqual(saved_path, workflow_path.resolve())

            schema = load_schema("scarlet_workflow_context_v1.0.yaml")
            report = validate_nexus_file(saved_path, schema)
            self.assertTrue(report.ok, "\n".join(report.format_lines()))

            loaded = WorkflowContext.load(saved_path)

            self.assertEqual(loaded.experiment_id, "exp-123")
            self.assertEqual(loaded.instrument_name, "sansllb")
            self.assertEqual(loaded.root_dir, raw_dir.resolve())
            self.assertEqual(loaded.output_dir, out_dir.resolve())
            self.assertEqual(len(loaded.runs), 4)
            self.assertEqual(sorted(key.duplicate_index for key in loaded.runs if key.entity == "dark"), [0, 1])
            self.assertEqual(loaded.get_empty_beam("config_1", "transmission"), empty_beam_path.resolve())
            self.assertEqual(loaded.get_beam_center("config_1", 0), (63.5, 64.5))
            self.assertEqual(loaded.get_roi("config_1"), (100, 140, 95, 135))
            self.assertEqual(loaded.get_transmission("sample_a", "config_1"), 0.91)
            self.assertEqual(loaded.get_empty_cell_transmission("config_1"), 0.95)
            self.assertEqual(loaded.get_sample_thickness("sample_a", "config_1"), 0.002)
            self.assertEqual(loaded.get_mask_file("config_1"), mask_bundle_path.resolve())
            self.assertEqual(loaded.get_flatfield("config_1"), flatfield_path.resolve())
            self.assertEqual(loaded.get_flatfield_source("config_2"), "config_1")
            self.assertIn("config_3", loaded.stale_flatfields)
            self.assertTrue(np.array_equal(loaded.get_mask("config_1", 0), np.zeros((4, 4), dtype=np.uint8)))
            self.assertEqual(loaded.timings["initialize"], 0.42)
            self.assertEqual(loaded.get("converted_data_dir"), out_dir.resolve())
            self.assertEqual(loaded.get("roi_cache"), {"config_1": (100, 140, 95, 135)})
            self.assertTrue(any(artifact.kind == "workflow_context" for artifact in loaded.artifacts))
            self.assertTrue(any(log.message == "workflow created" for log in loaded.logs))
            self.assertTrue(any(issue.message == "reference missing" for issue in loaded.issues))
