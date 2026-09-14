from __future__ import annotations

import argparse
from dataclasses import dataclass
import sys
from pathlib import Path

import h5py
import numpy as np

from scarlet.validation.schema_loader import load_schema
from scarlet.validation.schema_validator import ValidationReport, validate_nexus_file


@dataclass(frozen=True)
class _AzimuthalAverageExport:
    detector_indices: list[int]
    q: np.ndarray
    intensity: np.ndarray
    q_edges: np.ndarray
    n_pixels: np.ndarray


def _read_azimuthal_average_entry(
    file_path: Path,
    *,
    processed_entry: str,
    detector_indices: list[int] | None,
    q_min: float | None,
    q_max: float | None,
) -> _AzimuthalAverageExport:
    with h5py.File(file_path, "r") as handle:
        if processed_entry not in handle:
            raise ValueError(f"Missing processed entry: {processed_entry}")
        entry = handle[processed_entry]
        available = sorted(
            int(name[4:])
            for name, group in entry.items()
            if isinstance(group, h5py.Group) and name.startswith("data") and name[4:].isdigit()
        )
        if not available:
            raise ValueError(f"No NXdata groups found under {processed_entry}")
        selected = available if detector_indices is None else detector_indices
        missing = [index for index in selected if index not in available]
        if missing:
            raise ValueError(f"Missing detector data for indices {missing} in {processed_entry}")
        if len(selected) != 1:
            raise ValueError("azimuthal-average export currently supports one detector at a time")

        group = entry[f"data{selected[0]}"]
        q = np.asarray(group["Q"][()], dtype=np.float64)
        intensity = np.asarray(group["I"][()], dtype=np.float64)
        q_edges = np.asarray(group["Q_edges"][()], dtype=np.float64)
        n_pixels = np.asarray(group["n_pixels"][()], dtype=np.int64)

    if q.shape != intensity.shape or q.shape != n_pixels.shape:
        raise ValueError("Reduced azimuthal datasets must share the same 1D shape")

    valid = np.ones(q.shape, dtype=bool)
    if q_min is not None:
        valid &= q >= float(q_min)
    if q_max is not None:
        valid &= q <= float(q_max)
    if not np.any(valid):
        raise ValueError("No azimuthal-average points remain after applying the requested q range")

    return _AzimuthalAverageExport(
        detector_indices=list(selected),
        q=q[valid],
        intensity=intensity[valid],
        q_edges=q_edges,
        n_pixels=n_pixels[valid],
    )


def _write_azimuthal_average_csv(
    file_path: Path,
    result: _AzimuthalAverageExport,
    *,
    overwrite: bool,
) -> None:
    output_path = file_path.resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.column_stack((result.q, result.intensity, result.n_pixels))
    np.savetxt(
        output_path,
        matrix,
        delimiter=",",
        header="Q,I,n_pixels",
        comments="",
    )


def _cmd_schema_list(_: argparse.Namespace) -> int:
    from importlib import resources

    schemas_dir = resources.files("scarlet.schemas")
    schema_names = sorted(
        p.name for p in schemas_dir.iterdir() if p.is_file() and p.suffix in {".yaml", ".yml"}
    )
    for name in schema_names:
        print(name)
    return 0


def _print_validation_report(report: ValidationReport) -> None:
    for msg in report.format_lines():
        print(msg)
    if report.ok:
        print("OK")
    else:
        print(f"FAILED: {len(report.errors)} error(s), {len(report.warnings)} warning(s)")


def _validate_file_for_cli(
    file_path: Path,
    *,
    schema_name: str,
    entry_path: str | None = None,
    strict: bool = False,
) -> int:
    try:
        schema = load_schema(schema_name)
    except FileNotFoundError:
        print(f"Schema not found: {schema_name!r}", file=sys.stderr)
        return 2

    try:
        report = validate_nexus_file(
            file_path=file_path,
            schema=schema,
            entry_path=entry_path,
            strict=strict,
        )
    except OSError as e:
        print(f"Cannot open file {str(file_path)!r}: {e}", file=sys.stderr)
        return 2

    _print_validation_report(report)
    return 0 if report.ok else 1


def _cmd_validate(args: argparse.Namespace) -> int:
    return _validate_file_for_cli(
        Path(args.file),
        schema_name=args.schema,
        entry_path=args.entry,
        strict=args.strict,
    )


def _cmd_convert(args: argparse.Namespace) -> int:
    from scarlet.io.converters import convert_to_scarlet_nxsas_raw, list_apparatus

    apparatus = args.apparatus
    if apparatus == "list":
        for name in list_apparatus():
            print(name)
        return 0

    try:
        report = convert_to_scarlet_nxsas_raw(
            apparatus,
            Path(args.input),
            Path(args.output),
            entry_in=args.entry_in,
            overwrite=args.overwrite,
        )
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    except FileExistsError as e:
        print(str(e), file=sys.stderr)
        return 2
    except OSError as e:
        print(f"Cannot convert file: {e}", file=sys.stderr)
        return 2

    print(f"Converted: {report.input_file} -> {report.output_file}")
    print(f"Input entry: {report.entry_in}")
    for note in report.notes:
        print(f"NOTE: {note}")
    for warning in report.warnings:
        print(f"WARNING: {warning}")

    if args.validate:
        return _validate_file_for_cli(
            Path(args.output),
            schema_name=args.schema,
            strict=args.strict,
        )
    return 0


def _cmd_azimuthal_average(args: argparse.Namespace) -> int:
    detector_indices = None if args.detector is None else list(args.detector)
    try:
        result = _read_azimuthal_average_entry(
            Path(args.input),
            processed_entry=args.processed_entry,
            detector_indices=detector_indices,
            q_min=args.q_min,
            q_max=args.q_max,
        )
        _write_azimuthal_average_csv(
            Path(args.output),
            result,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, ValueError, OSError) as e:
        print(str(e), file=sys.stderr)
        return 2

    print(f"Azimuthal average: {args.output}")
    print(f"Detectors: " + ", ".join(f"detector{i}" for i in result.detector_indices))
    print(f"Bins: {len(result.q)}")
    print(f"Q range: {result.q_edges[0]:.6g} .. {result.q_edges[-1]:.6g} A^-1")
    return 0


def _cmd_viewer(args: argparse.Namespace) -> int:
    from scarlet.gui import run_viewer

    try:
        return int(
            run_viewer(
                None if args.directory is None else Path(args.directory),
                instrument=args.instrument,
            )
        )
    except (FileNotFoundError, NotADirectoryError, ValueError, OSError) as e:
        print(str(e), file=sys.stderr)
        return 2


def _cmd_tutorial(args: argparse.Namespace) -> int:
    from scarlet.tutorial import run

    return run(args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="scarlet", description="SCARLET utilities")
    sub = p.add_subparsers(dest="cmd", required=True)

    schema_p = sub.add_parser("schema", help="Schema utilities")
    schema_sub = schema_p.add_subparsers(dest="schema_cmd", required=True)
    schema_list = schema_sub.add_parser("list", help="List packaged schemas")
    schema_list.set_defaults(func=_cmd_schema_list)

    v = sub.add_parser("validate", help="Validate a NeXus/HDF5 file against a schema")
    v.add_argument("file", help="Path to .nxs/.h5 NeXus file")
    v.add_argument(
        "--schema",
        default="scarlet_nxsas_raw_v1.3_mono.yaml",
        help="Schema YAML filename (packaged) or path on disk",
    )
    v.add_argument(
        "--entry",
        default=None,
        help="Entry group path (default: schema's entry_path, usually /entry)",
    )
    v.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors",
    )
    v.set_defaults(func=_cmd_validate)

    c = sub.add_parser("convert", help="Convert raw instrument files to SCARLET NXsas_raw")
    c.add_argument(
        "apparatus",
        help="Instrument/apparatus name, e.g. 'sam' or 'sansllb'. Use 'list' to list known converters.",
    )
    c.add_argument("input", nargs="?", help="Input NeXus/HDF5 file")
    c.add_argument("output", nargs="?", help="Output SCARLET NXsas_raw file")
    c.add_argument("--entry-in", default=None, help="Input entry path/name to convert")
    c.add_argument("--overwrite", action="store_true", help="Overwrite existing output file")
    c.add_argument("--validate", action="store_true", help="Validate the converted file after writing")
    c.add_argument(
        "--schema",
        default="scarlet_nxsas_raw_v1.3_mono.yaml",
        help="Schema YAML filename or path used when --validate is enabled",
    )
    c.add_argument("--strict", action="store_true", help="Treat validation warnings as errors")
    c.set_defaults(func=_cmd_convert)

    avg = sub.add_parser(
        "azimuthal-average",
        help="Export or merge azimuthal I(Q) curves from a reduced SCARLET file",
    )
    avg.add_argument("input", help="Reduced SCARLET NeXus/HDF5 file containing /processed_data")
    avg.add_argument("output", help="Output CSV file")
    avg.add_argument(
        "--processed-entry",
        default="/processed_data",
        help="NXentry containing reduced detector images (default: /processed_data)",
    )
    avg.add_argument(
        "--detector",
        action="append",
        type=int,
        default=None,
        help="Detector index to include. Repeat to average multiple detectors; default is all reduced detectors.",
    )
    avg.add_argument("--bins", type=int, default=200, help="Number of radial Q bins (default: 200)")
    avg.add_argument("--q-min", type=float, default=None, help="Minimum Q in A^-1 (default: auto)")
    avg.add_argument("--q-max", type=float, default=None, help="Maximum Q in A^-1 (default: auto)")
    avg.add_argument(
        "--q-scale",
        choices=("linear", "log"),
        default="linear",
        help="Q binning scale used for azimuthal averaging (default: linear)",
    )
    avg.add_argument("--overwrite", action="store_true", help="Overwrite existing output CSV")
    avg.set_defaults(func=_cmd_azimuthal_average)

    viewer = sub.add_parser("viewer", help="Open the silx-based SCARLET viewer")
    viewer.add_argument("directory", nargs="?", help="Optional data folder loaded at startup")
    viewer.add_argument(
        "--instrument",
        choices=("sam", "sansllb"),
        default="sansllb",
        help="Instrument used to convert raw files before display (default: sansllb)",
    )
    viewer.set_defaults(func=_cmd_viewer)

    tutorial = sub.add_parser("notebook", help="Open the packaged SCARLET tutorial notebook")
    tutorial.add_argument(
        "destination",
        nargs="?",
        default=None,
        help=(
            "Notebook file or directory to open. Directory paths and paths without "
            "an extension receive tutorial.ipynb. If omitted, ./tutorial.ipynb is "
            "selected from a graphical launcher."
        ),
    )
    tutorial.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the destination notebook with the packaged tutorial.",
    )
    tutorial.add_argument(
        "--no-browser",
        action="store_true",
        help="Start Jupyter without opening a browser window.",
    )
    tutorial.add_argument(
        "--copy-only",
        action="store_true",
        help="Copy or locate the tutorial notebook without starting Jupyter.",
    )
    tutorial.set_defaults(func=_cmd_tutorial)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "convert" and args.apparatus != "list" and (args.input is None or args.output is None):
        parser.error("convert requires INPUT and OUTPUT, except for: scarlet convert list")
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return 130
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", None) or str(e)
        print(
            f"Missing dependency: {missing}. Install project dependencies: `pip install -e .`.",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
