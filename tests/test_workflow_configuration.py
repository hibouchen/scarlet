from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping, Optional

import h5py
import numpy as np

from scarlet.validation.schema_loader import load_schema
from scarlet.validation.schema_validator import validate_nexus_file
from scarlet.workflow.configuration import (
    Aperture,
    Collimation,
    ConfigTolerance,
    Configuration,
    compare_configurations,
    configuration_from_nexus,
    write_refs_sub_file,
)


def _ds(g: h5py.Group, name: str, value: Any) -> None:
    if isinstance(value, str):
        g.create_dataset(name, data=np.bytes_(value))
    else:
        g.create_dataset(name, data=value)


def _mk_aperture(g: h5py.Group, nx_class: str, fields: Mapping[str, Any]) -> None:
    g.attrs["NX_class"] = nx_class
    for k, v in fields.items():
        _ds(g, k, v)


def _write_refs_sub_style_file(
    path: Path,
    *,
    wavelength_a: Optional[float] = 6.0,
    sample_detector_distance_m: Optional[float] = 4.2,
    config_id: Optional[str] = "cfg",
    notes: Optional[str] = "notes",
    collimation_distance_m: Optional[float] = 1.5,
    last_aperture_to_sample_distance_m: Optional[float] = 0.5,
    aperture1: Optional[tuple[str, Mapping[str, Any]]] = ("NXslit", {"x_gap": 0.002, "y_gap": 0.003}),
    aperture2: Optional[tuple[str, Mapping[str, Any]]] = ("NXpinhole", {"diameter": 0.004}),
) -> None:
    with h5py.File(path, "w") as f:
        entry = f.create_group("entry")
        if config_id is not None:
            _ds(entry, "config_id", config_id)

        cfg = entry.create_group("configuration")
        if wavelength_a is not None:
            _ds(cfg, "wavelength", wavelength_a)
        if sample_detector_distance_m is not None:
            _ds(cfg, "sample_detector_distance", sample_detector_distance_m)
        if notes is not None:
            _ds(cfg, "notes", notes)

        if (
            collimation_distance_m is not None
            or last_aperture_to_sample_distance_m is not None
            or aperture1 is not None
            or aperture2 is not None
        ):
            col = cfg.create_group("collimation")
            if collimation_distance_m is not None:
                _ds(col, "collimation_distance", collimation_distance_m)
            if last_aperture_to_sample_distance_m is not None:
                _ds(col, "last_aperture_to_sample_distance", last_aperture_to_sample_distance_m)
            if aperture1 is not None:
                ap1 = col.create_group("aperture1")
                _mk_aperture(ap1, aperture1[0], aperture1[1])
            if aperture2 is not None:
                ap2 = col.create_group("aperture2")
                _mk_aperture(ap2, aperture2[0], aperture2[1])


def _write_instrument_style_file(
    path: Path,
    *,
    wavelength_a: Optional[float] = 6.0,
    translation_xyz_m: Optional[tuple[float, float, float]] = (0.0, 0.0, 4.2),
    upstream_elements: Optional[list[tuple[str, float, str, Mapping[str, Any]]]] = None,
) -> None:
    """
    upstream_elements:
      list of (name, z_m, nx_class, fields) placed under /entry/instrument/collimation/elements.
    """
    if upstream_elements is None:
        upstream_elements = [
            ("ap0", -4.0, "NXslit", {"x_gap": 0.002, "y_gap": 0.003}),
            ("ap1", -2.0, "NXslit", {"x_gap": 0.002, "y_gap": 0.003}),
            ("ap2", -0.5, "NXpinhole", {"diameter": 0.004}),
        ]

    with h5py.File(path, "w") as f:
        entry = f.create_group("entry")
        inst = entry.create_group("instrument")
        if wavelength_a is not None:
            mono = inst.create_group("monochromator")
            _ds(mono, "wavelength", wavelength_a)

        det0 = inst.create_group("detector0")
        if translation_xyz_m is not None:
            tr = det0.create_group("transformations")
            _ds(tr, "translation", np.array(translation_xyz_m, dtype=float))

        col = inst.create_group("collimation")
        elems = col.create_group("elements")
        for name, z_m, nx_class, fields in upstream_elements:
            g = elems.create_group(name)
            _mk_aperture(g, nx_class, fields)
            tr = g.create_group("transformations")
            _ds(tr, "translation", np.array([0.0, 0.0, float(z_m)], dtype=float))


def _write_reference_source_file(path: Path, *, title: str) -> None:
    with h5py.File(path, "w") as f:
        entry = f.create_group("entry")
        entry.attrs["NX_class"] = b"NXentry"
        _ds(entry, "title", title)


class TestConfigurationFromNexus(unittest.TestCase):
    def test_write_refs_sub_file_matches_schema(self) -> None:
        schema = load_schema("scarlet_refs_sub_v1.0.yaml")

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "refs_sub.nxs"
            ebt = Path(td) / "empty_beam_transmission.nxs"
            dark = Path(td) / "dark.nxs"
            _write_reference_source_file(ebt, title="ebt")
            _write_reference_source_file(dark, title="dark")

            configuration = Configuration(
                wavelength=6.0,
                sample_detector_distance=4.2,
                config_id="cfg-1",
                notes="baseline",
                collimation=Collimation(
                    aperture1=Aperture(type="slit", x_gap=0.002, y_gap=0.003),
                    aperture2=Aperture(type="pinhole", diameter=0.004),
                    collimation_distance=1.5,
                    last_aperture_to_sample_distance=0.5,
                ),
            )

            write_refs_sub_file(
                out,
                configuration,
                empty_beam_transmission=ebt,
                dark=dark,
                transmission_roi_detector=0,
                transmission_roi=(1, 10, 2, 11),
                masks={0: np.array([[0, 1], [1, 0]], dtype=np.uint8)},
                attenuation_factor=3.0,
                created_utc="2026-03-30T00:00:00Z",
                scarlet_version="test",
            )

            report = validate_nexus_file(out, schema)
            self.assertTrue(report.ok, report.format_lines())

            cfg, issues = configuration_from_nexus(out)
            self.assertEqual(issues, [])
            self.assertEqual(cfg, configuration)

            with h5py.File(out, "r") as f:
                self.assertEqual(f["/entry/references/empty_beam_transmission/entry/title"][()].decode(), "ebt")
                self.assertEqual(f["/entry/meta/dark_source_file"][()].decode(), str(dark.resolve()))

    def test_refs_sub_style_round_trip_with_collimation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "cfg.h5"
            _write_refs_sub_style_file(p)

            cfg, issues = configuration_from_nexus(p)
            self.assertEqual(issues, [])
            self.assertAlmostEqual(cfg.wavelength, 6.0)
            self.assertAlmostEqual(cfg.sample_detector_distance, 4.2)
            self.assertEqual(cfg.config_id, "cfg")
            self.assertEqual(cfg.notes, "notes")
            self.assertIsNotNone(cfg.collimation)
            assert cfg.collimation is not None
            self.assertEqual(cfg.collimation.aperture1.type, "slit")
            self.assertAlmostEqual(cfg.collimation.aperture1.x_gap or 0.0, 0.002)
            self.assertAlmostEqual(cfg.collimation.aperture1.y_gap or 0.0, 0.003)
            self.assertEqual(cfg.collimation.aperture2.type, "pinhole")
            self.assertAlmostEqual(cfg.collimation.aperture2.diameter or 0.0, 0.004)
            self.assertAlmostEqual(cfg.collimation.collimation_distance, 1.5)
            self.assertAlmostEqual(cfg.collimation.last_aperture_to_sample_distance, 0.5)

    def test_refs_sub_style_missing_parameters_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "missing.h5"
            _write_refs_sub_style_file(
                p,
                wavelength_a=None,
                sample_detector_distance_m=None,
                collimation_distance_m=None,  # triggers "incomplete collimation"
            )

            cfg, issues = configuration_from_nexus(p)
            self.assertTrue(any("configuration/wavelength missing" in s for s in issues))
            self.assertTrue(any("configuration/sample_detector_distance missing" in s for s in issues))
            self.assertTrue(any("incomplete collimation" in s for s in issues))
            self.assertTrue(math.isnan(cfg.wavelength))
            self.assertTrue(math.isnan(cfg.sample_detector_distance))
            self.assertIsNone(cfg.collimation)

    def test_instrument_style_infers_distances_and_collimation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "inst.h5"
            _write_instrument_style_file(p)

            cfg, issues = configuration_from_nexus(p)
            self.assertEqual(issues, [])
            self.assertAlmostEqual(cfg.wavelength, 6.0)
            self.assertAlmostEqual(cfg.sample_detector_distance, 4.2)
            self.assertIsNotNone(cfg.collimation)
            assert cfg.collimation is not None
            # upstream elements are at z=-2.0 and z=-0.5 (two closest to sample), cd=1.5, lad=0.5
            self.assertAlmostEqual(cfg.collimation.collimation_distance, 1.5)
            self.assertAlmostEqual(cfg.collimation.last_aperture_to_sample_distance, 0.5)
            self.assertEqual(cfg.collimation.aperture1.type, "slit")
            self.assertEqual(cfg.collimation.aperture2.type, "pinhole")

    def test_instrument_style_missing_upstream_apertures(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "inst_missing.h5"
            _write_instrument_style_file(
                p,
                upstream_elements=[("only", -1.0, "NXslit", {"x_gap": 0.002, "y_gap": 0.003})],
            )

            cfg, issues = configuration_from_nexus(p)
            self.assertIsNone(cfg.collimation)
            self.assertTrue(any("Could not infer 2 upstream apertures" in s for s in issues))


class TestCompareConfigurations(unittest.TestCase):
    def test_compare_from_files_same_within_tolerance(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p1 = Path(td) / "a.h5"
            p2 = Path(td) / "b.h5"
            _write_refs_sub_style_file(p1, wavelength_a=6.0, sample_detector_distance_m=4.200)
            _write_refs_sub_style_file(p2, wavelength_a=6.05, sample_detector_distance_m=4.205)

            a, _ = configuration_from_nexus(p1)
            b, _ = configuration_from_nexus(p2)
            same, diffs = compare_configurations(a, b)
            self.assertTrue(same)
            self.assertEqual(diffs, [])

    def test_compare_from_files_detects_collimation_variation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p1 = Path(td) / "a.h5"
            p2 = Path(td) / "b.h5"
            _write_refs_sub_style_file(p1, aperture2=("NXpinhole", {"diameter": 0.004}))
            _write_refs_sub_style_file(p2, aperture2=("NXpinhole", {"diameter": 0.020}))

            a, _ = configuration_from_nexus(p1)
            b, _ = configuration_from_nexus(p2)
            same, diffs = compare_configurations(a, b, tol=ConfigTolerance(aperture_m=0.001))
            self.assertFalse(same)
            self.assertTrue(any("aperture2.diameter" in s for s in diffs))

    def test_compare_handles_missing_collimation_when_not_required(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p1 = Path(td) / "a.h5"
            p2 = Path(td) / "b.h5"
            _write_refs_sub_style_file(p1, collimation_distance_m=None, last_aperture_to_sample_distance_m=None, aperture1=None, aperture2=None)
            _write_refs_sub_style_file(p2)

            a, _ = configuration_from_nexus(p1)
            b, _ = configuration_from_nexus(p2)
            same, diffs = compare_configurations(a, b, require_collimation=False)
            self.assertTrue(same)
            self.assertEqual(diffs, [])

    def test_compare_reports_missing_aperture_parameter(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p1 = Path(td) / "a.h5"
            p2 = Path(td) / "b.h5"
            _write_refs_sub_style_file(p1, aperture1=("NXslit", {"x_gap": 0.002, "y_gap": 0.003}))
            _write_refs_sub_style_file(p2, aperture1=("NXslit", {"y_gap": 0.003}))  # x_gap missing

            a, _ = configuration_from_nexus(p1)
            b, _ = configuration_from_nexus(p2)
            same, diffs = compare_configurations(a, b, tol=ConfigTolerance(aperture_m=1e-9))
            self.assertFalse(same)
            self.assertTrue(any("aperture1.x_gap: missing value(s)" in s for s in diffs))
