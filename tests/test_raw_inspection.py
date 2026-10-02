from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from scarlet.io.raw_inspection import inspect_raw_run


def _write_sansllb_run(
    path: Path,
    *,
    beam_stop_x_mm: float | None,
    beam_stop_y_mm: float | None,
    image: np.ndarray,
    attenuator_selection: str = "0",
) -> None:
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry0")
        entry.attrs["NX_class"] = np.bytes_("NXentry")
        sample = entry.create_group("sample")
        sample.create_dataset("name", data=np.bytes_("sample_a"))

        instrument = entry.create_group("SANS-LLB")
        instrument.attrs["NX_class"] = np.bytes_("NXinstrument")
        detector = instrument.create_group("central_detector")
        detector.attrs["NX_class"] = np.bytes_("NXdetector")
        detector.create_dataset("data", data=image)
        distance = detector.create_dataset("distance", data=8000.0)
        distance.attrs["units"] = np.bytes_("mm")

        attenuator = instrument.create_group("attenuator")
        attenuator.create_dataset("selection", data=np.bytes_(attenuator_selection))

        if beam_stop_x_mm is not None and beam_stop_y_mm is not None:
            beam_stop = instrument.create_group("beam_stop")
            for name, value in (
                ("x", beam_stop_x_mm),
                ("y", beam_stop_y_mm),
                ("size", 70.0),
            ):
                dataset = beam_stop.create_dataset(name, data=value)
                dataset.attrs["units"] = np.bytes_("mm")


class TestSansllbRawInspection(unittest.TestCase):
    def test_parked_beam_stop_identifies_transmission_despite_scattering_like_image(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "transmission.hdf"
            image = np.zeros((31, 31), dtype=np.float64)
            image[15, 20] = 100.0
            _write_sansllb_run(
                path,
                beam_stop_x_mm=10.0,
                beam_stop_y_mm=-95.0,
                image=image,
            )

            metadata = inspect_raw_run("sansllb", path)

            self.assertEqual(metadata.mode, "transmission")

    def test_centered_beam_stop_identifies_scattering_despite_transmission_like_image(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "scattering.hdf"
            image = np.zeros((31, 31), dtype=np.float64)
            image[15, 15] = 100.0
            _write_sansllb_run(
                path,
                beam_stop_x_mm=7.0,
                beam_stop_y_mm=-3.0,
                image=image,
            )

            metadata = inspect_raw_run("sansllb", path)

            self.assertEqual(metadata.mode, "scattering")

    def test_inserted_attenuator_identifies_transmission_when_beam_stop_metadata_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "transmission_without_beam_stop.hdf"
            image = np.zeros((31, 31), dtype=np.float64)
            image[15, 20] = 100.0
            _write_sansllb_run(
                path,
                beam_stop_x_mm=None,
                beam_stop_y_mm=None,
                image=image,
                attenuator_selection="2",
            )

            metadata = inspect_raw_run("sansllb", path)

            self.assertEqual(metadata.mode, "transmission")


if __name__ == "__main__":
    unittest.main()
