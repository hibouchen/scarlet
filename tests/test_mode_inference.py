from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from scarlet.io.mode_inference import guess_measurement_mode_from_nexus_image


class TestModeInference(unittest.TestCase):
    def test_guess_measurement_mode_from_nexus_image_returns_unknown_for_non_2d_detector_data(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            file_path = Path(td) / "non_2d_detector.nxs"
            with h5py.File(file_path, "w") as handle:
                entry = handle.create_group("raw_data")
                entry.attrs["NX_class"] = b"NXentry"
                instrument = entry.create_group("instrument")
                instrument.attrs["NX_class"] = b"NXinstrument"
                detector = instrument.create_group("detector0")
                detector.attrs["NX_class"] = b"NXdetector"
                detector.create_dataset("data", data=np.ones((1, 4, 4), dtype=np.float64))

            guess = guess_measurement_mode_from_nexus_image(file_path)

            self.assertEqual(guess.mode, "unknown")
            self.assertEqual(guess.confidence, 0.0)
            self.assertEqual(guess.scores, {"transmission": 0.0, "scattering": 0.0})
            self.assertEqual(len(guess.reasons), 1)
            self.assertIn("not 2D", guess.reasons[0])

