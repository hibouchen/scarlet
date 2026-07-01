from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

raise unittest.SkipTest("scarlet.workflow.reference has been removed for now")

from scarlet.reduction import normalize_by_solid_angle
from scarlet.workflow.configuration import Aperture, Collimation, Configuration
from scarlet.workflow.pipeline import (
    ReductionInputs,
    ReductionPipeline,
    ReductionState,
    as_reduction_step,
    check_inputs,
    check_ref_sub_file,
    compute_transmission_step,
    subtract_references_step,
)
from scarlet.workflow.reference import write_refs_sub_file


def _write_raw_file(
    path: Path,
    *,
    data: np.ndarray,
    monitor_integral: float = 1.0,
    beam_center: tuple[float, float] | None = None,
    detector_distance: float | None = None,
    extra_detectors: dict[int, np.ndarray] | None = None,
    extra_beam_centers: dict[int, tuple[float, float]] | None = None,
    extra_detector_distances: dict[int, float] | None = None,
) -> None:
    with h5py.File(path, "w") as f:
        entry = f.create_group("raw_data")
        entry.attrs["NX_class"] = b"NXentry"
        control = entry.create_group("control")
        control.attrs["NX_class"] = b"NXmonitor"
        control.create_dataset("integral", data=float(monitor_integral))
        instrument = entry.create_group("instrument")
        instrument.attrs["NX_class"] = b"NXinstrument"
        detector = instrument.create_group("detector0")
        detector.attrs["NX_class"] = b"NXdetector"
        data_array = np.asarray(data, dtype=np.float64)
        detector.create_dataset("data", data=data_array)
        detector.create_dataset("x_pixel_size", data=0.001)
        detector.create_dataset("y_pixel_size", data=0.001)
        if beam_center is None:
            beam_center = ((data_array.shape[1] - 1) / 2.0, (data_array.shape[0] - 1) / 2.0)
        detector.create_dataset("beam_center_x", data=float(beam_center[0]))
        detector.create_dataset("beam_center_y", data=float(beam_center[1]))
        if detector_distance is not None:
            transformations = detector.create_group("transformations")
            transformations.create_dataset("translation", data=np.array([0.0, 0.0, float(detector_distance)]))
        for detector_number, detector_data in sorted((extra_detectors or {}).items()):
            extra_detector = instrument.create_group(f"detector{int(detector_number)}")
            extra_detector.attrs["NX_class"] = b"NXdetector"
            extra_array = np.asarray(detector_data, dtype=np.float64)
            extra_detector.create_dataset("data", data=extra_array)
            extra_detector.create_dataset("x_pixel_size", data=0.001)
            extra_detector.create_dataset("y_pixel_size", data=0.001)
            extra_center = None if extra_beam_centers is None else extra_beam_centers.get(int(detector_number))
            if extra_center is None:
                extra_center = ((extra_array.shape[1] - 1) / 2.0, (extra_array.shape[0] - 1) / 2.0)
            extra_detector.create_dataset("beam_center_x", data=float(extra_center[0]))
            extra_detector.create_dataset("beam_center_y", data=float(extra_center[1]))
            extra_distance = None if extra_detector_distances is None else extra_detector_distances.get(int(detector_number))
            if extra_distance is not None:
                transformations = extra_detector.create_group("transformations")
                transformations.create_dataset("translation", data=np.array([0.0, 0.0, float(extra_distance)]))


def _configuration() -> Configuration:
    return Configuration(
        wavelength=6.0,
        sample_detector_distance=[4.2],
        config_id="cfg",
        collimation=Collimation(
            aperture1=Aperture(type="slit", x_gap=0.002, y_gap=0.003),
            aperture2=Aperture(type="pinhole", diameter=0.004),
            collimation_distance=1.5,
            last_aperture_to_sample_distance=0.5,
        ),
    )


class TestPipelineBis(unittest.TestCase):
    def test_default_pipeline_uses_decorated_step_names(self) -> None:
        pipeline = ReductionPipeline.default()

        self.assertEqual(
            pipeline.step_names,
            (
                "check inputs",
                "check reference subtraction file",
                "check reference normalization file",
                "compute transmission",
                "subtract references",
                "flatfield correction",
                "azimuthal averaging",
            ),
        )
        self.assertEqual(as_reduction_step(check_ref_sub_file).name, "check reference subtraction file")

    def test_compute_transmission_step_replaces_frozen_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sample_transmission = root / "sample_transmission.nxs"
            empty_beam_transmission = root / "empty_beam_transmission.nxs"
            refs_sub = root / "refs_sub.nxs"

            _write_raw_file(sample_transmission, data=np.full((4, 4), 5.0, dtype=np.float64))
            _write_raw_file(empty_beam_transmission, data=np.full((4, 4), 10.0, dtype=np.float64))

            write_refs_sub_file(
                refs_sub,
                _configuration(),
                empty_beam_transmission=empty_beam_transmission,
                transmission_roi_detector=0,
                transmission_roi=(1, 3, 1, 3),
            )

            state = ReductionState(
                inputs=ReductionInputs(
                    sample_file_scattering=str(sample_transmission),
                    sample_file_transmission=str(sample_transmission),
                    ref_sub_file=str(refs_sub),
                )
            )

            updated = compute_transmission_step(state)

            self.assertAlmostEqual(updated.inputs.sample_transmission or 0.0, 0.5)

    def test_check_inputs_initializes_sample_detectors_and_falls_back_to_scattering_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sample_scattering = root / "sample_scattering.nxs"
            _write_raw_file(sample_scattering, data=np.full((4, 4), 8.0, dtype=np.float64), monitor_integral=4.0)

            state = ReductionState(
                inputs=ReductionInputs(
                    sample_file_scattering=str(sample_scattering),
                    sample_file_transmission="",
                )
            )

            updated = check_inputs(state)

            self.assertEqual(updated.inputs.sample_file_transmission, str(sample_scattering))
            self.assertEqual(sorted(updated.detectors), [0])
            np.testing.assert_allclose(updated.detectors[0].data, np.full((4, 4), 2.0, dtype=np.float64))
            np.testing.assert_allclose(updated.detectors[0].data_error, np.full((4, 4), np.sqrt(8.0) / 4.0))
            self.assertIsNone(updated.detectors[0].mask)
            np.testing.assert_allclose(updated.data, updated.detectors[0].data)

    def test_check_ref_sub_file_merges_reference_masks_into_state_detectors(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sample_scattering = root / "sample_scattering.nxs"
            empty_beam_transmission = root / "empty_beam_transmission.nxs"
            refs_sub = root / "refs_sub.nxs"
            sample_data = np.arange(16, dtype=np.float64).reshape(4, 4)

            _write_raw_file(sample_scattering, data=sample_data, monitor_integral=2.0)
            _write_raw_file(empty_beam_transmission, data=np.full((4, 4), 10.0, dtype=np.float64))

            write_refs_sub_file(
                refs_sub,
                _configuration(),
                empty_beam_transmission=empty_beam_transmission,
                transmission_roi_detector=0,
                transmission_roi=(1, 3, 1, 3),
                masks={0: np.array([[0, 1, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.uint8)},
            )

            state = check_inputs(
                ReductionState(
                    inputs=ReductionInputs(
                        sample_file_scattering=str(sample_scattering),
                        sample_file_transmission="",
                        ref_sub_file=str(refs_sub),
                    )
                )
            )
            updated = check_ref_sub_file(state)

            expected_mask = np.array([[0, 1, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.uint8)
            np.testing.assert_array_equal(updated.detectors[0].mask, expected_mask)
            np.testing.assert_allclose(updated.detectors[0].data, sample_data / 2.0)

    def test_without_water_correction_pipeline_subtracts_refs_sub_references(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sample_scattering = root / "sample_scattering.nxs"
            sample_transmission = root / "sample_transmission.nxs"
            empty_beam_transmission = root / "empty_beam_transmission.nxs"
            dark = root / "dark.nxs"
            empty_cell_transmission = root / "empty_cell_transmission.nxs"
            empty_cell_scattering = root / "empty_cell_scattering.nxs"
            refs_sub = root / "refs_sub.nxs"

            _write_raw_file(
                sample_scattering,
                data=np.full((4, 4), 10.0, dtype=np.float64),
                beam_center=(1.5, 1.5),
                detector_distance=2.1,
            )
            _write_raw_file(sample_transmission, data=np.full((4, 4), 5.0, dtype=np.float64))
            _write_raw_file(empty_beam_transmission, data=np.full((4, 4), 10.0, dtype=np.float64))
            _write_raw_file(dark, data=np.full((4, 4), 1.0, dtype=np.float64))
            _write_raw_file(empty_cell_transmission, data=np.full((4, 4), 5.0, dtype=np.float64))
            _write_raw_file(empty_cell_scattering, data=np.full((4, 4), 4.0, dtype=np.float64))

            write_refs_sub_file(
                refs_sub,
                _configuration(),
                empty_beam_transmission=empty_beam_transmission,
                dark=dark,
                empty_cell_transmission=empty_cell_transmission,
                empty_cell_scattering=empty_cell_scattering,
                transmission_roi_detector=0,
                transmission_roi=(1, 3, 1, 3),
                beam_centers={0: (0.25, 2.25)},
            )

            pipeline = ReductionPipeline(
                steps=(
                    as_reduction_step(check_inputs),
                    as_reduction_step(check_ref_sub_file),
                    as_reduction_step(compute_transmission_step),
                    as_reduction_step(subtract_references_step),
                )
            )

            state = pipeline.run(
                ReductionInputs(
                    sample_file_scattering=str(sample_scattering),
                    sample_file_transmission=str(sample_transmission),
                    ref_sub_file=str(refs_sub),
                )
            )

            expected = normalize_by_solid_angle(
                np.full((4, 4), 12.0, dtype=np.float64),
                detector_distance=4.2,
                beam_center=(0.25, 2.25),
                pixel_size=(0.001, 0.001),
            )
            solid_angle_correction = normalize_by_solid_angle(
                np.ones((4, 4), dtype=np.float64),
                detector_distance=4.2,
                beam_center=(0.25, 2.25),
                pixel_size=(0.001, 0.001),
            )
            np.testing.assert_allclose(state.data, expected)
            np.testing.assert_allclose(
                state.data_error,
                np.full((4, 4), np.sqrt(56.0), dtype=np.float64) * solid_angle_correction,
            )
            self.assertEqual(sorted(state.detectors), [0])
            self.assertEqual(
                state.reductions_steps,
                [
                    "check inputs",
                    "check reference subtraction file",
                    "compute transmission",
                    "subtract references",
                ],
            )
            self.assertAlmostEqual(state.inputs.sample_transmission or 0.0, 0.5)

    def test_only_azimuthal_averaging_pipeline_skips_transmission_and_ref_norm(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sample_scattering = root / "sample_scattering.nxs"
            empty_beam_transmission = root / "empty_beam_transmission.nxs"
            refs_sub = root / "refs_sub.nxs"

            sample_data = np.zeros((4, 4), dtype=np.float64)
            sample_data[1:3, 1:3] = 10.0
            _write_raw_file(sample_scattering, data=sample_data, monitor_integral=2.0, detector_distance=2.1)
            _write_raw_file(empty_beam_transmission, data=np.full((4, 4), 10.0, dtype=np.float64))

            write_refs_sub_file(
                refs_sub,
                _configuration(),
                empty_beam_transmission=empty_beam_transmission,
                transmission_roi_detector=0,
                transmission_roi=(1, 3, 1, 3),
                beam_centers={0: (1.5, 1.5)},
                masks={0: np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8)},
            )

            state = ReductionPipeline.only_azimuthal_averaging().run(
                ReductionInputs(
                    sample_file_scattering=str(sample_scattering),
                    sample_file_transmission="",
                    ref_sub_file=str(refs_sub),
                    ref_norm_file="",
                )
            )

            self.assertEqual(state.reductions_steps, ["check inputs", "check reference subtraction file", "azimuthal averaging"])
            self.assertEqual(sorted(state.detectors), [0])
            self.assertIsNotNone(state.x)
            self.assertEqual(state.data.ndim, 1)
            self.assertTrue(np.any(np.isfinite(state.data)))

    def test_without_water_correction_pipeline_subtracts_references_for_all_sample_detectors(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sample_scattering = root / "sample_scattering.nxs"
            sample_transmission = root / "sample_transmission.nxs"
            empty_beam_transmission = root / "empty_beam_transmission.nxs"
            dark = root / "dark.nxs"
            empty_cell_transmission = root / "empty_cell_transmission.nxs"
            empty_cell_scattering = root / "empty_cell_scattering.nxs"
            refs_sub = root / "refs_sub.nxs"

            _write_raw_file(
                sample_scattering,
                data=np.full((4, 4), 10.0, dtype=np.float64),
                extra_detectors={1: np.full((4, 4), 8.0, dtype=np.float64)},
                beam_center=(1.5, 1.5),
                detector_distance=2.1,
                extra_beam_centers={1: (1.5, 1.5)},
                extra_detector_distances={1: 2.1},
            )
            _write_raw_file(sample_transmission, data=np.full((4, 4), 5.0, dtype=np.float64))
            _write_raw_file(
                empty_beam_transmission,
                data=np.full((4, 4), 10.0, dtype=np.float64),
                extra_detectors={1: np.full((4, 4), 10.0, dtype=np.float64)},
            )
            _write_raw_file(
                dark,
                data=np.full((4, 4), 1.0, dtype=np.float64),
                extra_detectors={1: np.full((4, 4), 1.0, dtype=np.float64)},
            )
            _write_raw_file(empty_cell_transmission, data=np.full((4, 4), 5.0, dtype=np.float64))
            _write_raw_file(
                empty_cell_scattering,
                data=np.full((4, 4), 4.0, dtype=np.float64),
                extra_detectors={1: np.full((4, 4), 3.0, dtype=np.float64)},
            )

            configuration = Configuration(
                wavelength=6.0,
                sample_detector_distance=[4.2, 3.1],
                config_id="cfg",
                collimation=Collimation(
                    aperture1=Aperture(type="slit", x_gap=0.002, y_gap=0.003),
                    aperture2=Aperture(type="pinhole", diameter=0.004),
                    collimation_distance=1.5,
                    last_aperture_to_sample_distance=0.5,
                ),
            )
            write_refs_sub_file(
                refs_sub,
                configuration,
                empty_beam_transmission=empty_beam_transmission,
                dark=dark,
                empty_cell_transmission=empty_cell_transmission,
                empty_cell_scattering=empty_cell_scattering,
                transmission_roi_detector=0,
                transmission_roi=(1, 3, 1, 3),
                beam_centers={0: (0.25, 2.25), 1: (1.75, 0.5)},
            )

            pipeline = ReductionPipeline(
                steps=(
                    as_reduction_step(check_inputs),
                    as_reduction_step(check_ref_sub_file),
                    as_reduction_step(compute_transmission_step),
                    as_reduction_step(subtract_references_step),
                )
            )

            state = pipeline.run(
                ReductionInputs(
                    sample_file_scattering=str(sample_scattering),
                    sample_file_transmission=str(sample_transmission),
                    ref_sub_file=str(refs_sub),
                )
            )

            self.assertEqual(sorted(state.detectors), [0, 1])
            expected0 = normalize_by_solid_angle(
                np.full((4, 4), 12.0, dtype=np.float64),
                detector_distance=4.2,
                beam_center=(0.25, 2.25),
                pixel_size=(0.001, 0.001),
            )
            expected1 = normalize_by_solid_angle(
                np.full((4, 4), 10.0, dtype=np.float64),
                detector_distance=3.1,
                beam_center=(1.75, 0.5),
                pixel_size=(0.001, 0.001),
            )
            np.testing.assert_allclose(state.detectors[0].data, expected0)
            np.testing.assert_allclose(state.detectors[1].data, expected1)
            self.assertIsNotNone(state.detectors[0].data_error)
            self.assertIsNotNone(state.detectors[1].data_error)
            np.testing.assert_allclose(state.data, expected0)


if __name__ == "__main__":
    unittest.main()
