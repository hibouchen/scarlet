from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from scarlet.workflow.configuration import configuration_from_nexus


def _ds(group: h5py.Group, name: str, value) -> None:
    if isinstance(value, str):
        group.create_dataset(name, data=np.bytes_(value))
    else:
        group.create_dataset(name, data=value)


class TestConfigurationFromNexus(unittest.TestCase):
    def test_prefers_explicit_instrument_collimation_metadata_over_element_inference(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "inst_explicit_collimation.nxs"
            with h5py.File(path, "w") as f:
                entry = f.create_group("raw_data")
                instrument = entry.create_group("instrument")

                mono = instrument.create_group("monochromator")
                _ds(mono, "wavelength", 5.0)

                det0 = instrument.create_group("detector0")
                tr0 = det0.create_group("transformations")
                _ds(tr0, "translation", np.array([0.0, 0.0, 8.0], dtype=float))

                col = instrument.create_group("collimation")
                _ds(col, "collimation_distance", 8.0)
                _ds(col, "last_aperture_to_sample_distance", 1.6)

                ap1 = col.create_group("aperture1")
                ap1.attrs["NX_class"] = b"NXslit"
                _ds(ap1, "x_gap", 0.06)
                _ds(ap1, "y_gap", 0.06)

                ap2 = col.create_group("aperture2")
                ap2.attrs["NX_class"] = b"NXslit"
                _ds(ap2, "x_gap", 0.008)
                _ds(ap2, "y_gap", 0.008)

                elems = col.create_group("elements")
                for name, z, xgap in (("slit0", -2.9333333, 0.02375), ("slit1", -1.6, 0.01456)):
                    slit = elems.create_group(name)
                    slit.attrs["NX_class"] = b"NXslit"
                    _ds(slit, "x_gap", xgap)
                    _ds(slit, "y_gap", xgap)
                    tr = slit.create_group("transformations")
                    _ds(tr, "translation", np.array([0.0, 0.0, z], dtype=float))

            cfg, issues = configuration_from_nexus(path, entry_path="/raw_data")

            self.assertEqual(issues, [])
            self.assertIsNotNone(cfg.collimation)
            assert cfg.collimation is not None
            self.assertAlmostEqual(cfg.collimation.collimation_distance, 8.0)
            self.assertAlmostEqual(cfg.collimation.last_aperture_to_sample_distance, 1.6)
            self.assertAlmostEqual(cfg.collimation.aperture1.x_gap or 0.0, 0.06)
            self.assertAlmostEqual(cfg.collimation.aperture2.x_gap or 0.0, 0.008)


if __name__ == "__main__":
    unittest.main()
