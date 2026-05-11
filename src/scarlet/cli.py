from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scarlet.validation.schema_loader import load_schema
from scarlet.validation.schema_validator import ValidationReport, validate_nexus_file


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


def _print_generated_files(outputs: dict[str, Path]) -> None:
    if not outputs:
        print("No files generated.")
        return
    for config_id, path in sorted(outputs.items()):
        print(f"{config_id}: {path}")


def _validate_generated_files(
    outputs: dict[str, Path],
    *,
    schema_name: str,
    strict: bool,
) -> int:
    status = 0
    for config_id, path in sorted(outputs.items()):
        print(f"\nValidating {config_id}: {path}")
        status = max(
            status,
            _validate_file_for_cli(
                path,
                schema_name=schema_name,
                strict=strict,
            ),
        )
    return status


def _cmd_refs_sub_from_excel(args: argparse.Namespace) -> int:
    from scarlet.workflow.configuration import write_refs_sub_files_from_excel

    try:
        outputs = write_refs_sub_files_from_excel(
            Path(args.excel),
            Path(args.data_dir),
            Path(args.output_dir),
            transmission_roi_detector=args.transmission_roi_detector,
            transmission_roi_half_size=args.transmission_roi_half_size,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, ValueError, OSError) as e:
        print(str(e), file=sys.stderr)
        return 2

    _print_generated_files(outputs)
    if args.validate:
        return _validate_generated_files(
            outputs,
            schema_name=args.schema,
            strict=args.strict,
        )
    return 0


def _cmd_refs_norm_from_excel(args: argparse.Namespace) -> int:
    from scarlet.workflow.configuration import write_refs_norm_files_from_excel

    try:
        outputs = write_refs_norm_files_from_excel(
            Path(args.excel),
            Path(args.data_dir),
            Path(args.output_dir),
            transmission_roi_detector=args.transmission_roi_detector,
            transmission_roi_half_size=args.transmission_roi_half_size,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, ValueError, OSError) as e:
        print(str(e), file=sys.stderr)
        return 2

    _print_generated_files(outputs)
    if args.validate:
        return _validate_generated_files(
            outputs,
            schema_name=args.schema,
            strict=args.strict,
        )
    return 0


def _cmd_reduce_2d(args: argparse.Namespace) -> int:
    from scarlet.reduction import reduce_2d

    try:
        result = reduce_2d(
            Path(args.sample_scattering),
            Path(args.refs_sub),
            sample_transmission=None if args.sample_transmission is None else Path(args.sample_transmission),
            refs_norm=None if args.refs_norm is None else Path(args.refs_norm),
            output_path=Path(args.output),
            detector_index=args.detector,
            normalize_by=args.normalize_by,
            apply_mask=not args.no_mask,
            overwrite=args.overwrite,
            raw_entry=args.raw_entry,
            processed_entry=args.processed_entry,
            refs_entry=args.refs_entry,
        )
    except (FileExistsError, FileNotFoundError, ValueError, OSError) as e:
        print(str(e), file=sys.stderr)
        return 2

    print(f"Reduced 2D image: {args.output}")
    print(f"Sample transmission: {result.sample_transmission.value:.6g}")
    if result.water_transmission is not None:
        print(f"Water transmission: {result.water_transmission.value:.6g}")
    if len(result.detector_indices) == 1:
        print(f"Detector: detector{result.detector_index}")
    else:
        print("Detectors: " + ", ".join(f"detector{i}" for i in result.detector_indices))
    print(f"Normalization: {result.normalize_by}")
    return 0


def _cmd_azimuthal_average(args: argparse.Namespace) -> int:
    from scarlet.reduction import azimuthal_average, write_azimuthal_average_csv

    detector_indices = None if args.detector is None else list(args.detector)
    try:
        result = azimuthal_average(
            Path(args.input),
            detector_indices=detector_indices,
            processed_entry=args.processed_entry,
            n_bins=args.bins,
            q_min=args.q_min,
            q_max=args.q_max,
        )
        write_azimuthal_average_csv(
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


def _add_refs_from_excel_parser(
    parent: argparse.ArgumentParser,
    *,
    default_schema: str,
    command_help: str,
) -> argparse.ArgumentParser:
    p = parent.add_parser("from-excel", help=command_help)
    p.add_argument("excel", help="Run-configuration Excel file generated by SCARLET")
    p.add_argument("data_dir", help="Directory containing the converted NXsas_raw data files")
    p.add_argument("output_dir", help="Output directory for generated reference bundles")
    p.add_argument(
        "--transmission-roi-detector",
        type=int,
        default=0,
        help="Detector index used to estimate the transmission ROI (default: 0)",
    )
    p.add_argument(
        "--transmission-roi-half-size",
        type=int,
        default=1,
        help="Minimum half-size, in pixels, added around the detected direct-beam ROI (default: 1)",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    p.add_argument("--validate", action="store_true", help="Validate generated files after writing")
    p.add_argument(
        "--schema",
        default=default_schema,
        help="Schema YAML filename or path used when --validate is enabled",
    )
    p.add_argument("--strict", action="store_true", help="Treat validation warnings as errors")
    return p


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

    refs_sub = sub.add_parser("refs-sub", help="Generate SCARLET subtraction-reference bundles")
    refs_sub_sub = refs_sub.add_subparsers(dest="refs_sub_cmd", required=True)
    refs_sub_from_excel = _add_refs_from_excel_parser(
        refs_sub_sub,
        default_schema="scarlet_refs_sub_v1.0.yaml",
        command_help="Generate refs_sub files from a run-configuration Excel file",
    )
    refs_sub_from_excel.set_defaults(func=_cmd_refs_sub_from_excel)

    refs_norm = sub.add_parser("refs-norm", help="Generate SCARLET normalization-reference bundles")
    refs_norm_sub = refs_norm.add_subparsers(dest="refs_norm_cmd", required=True)
    refs_norm_from_excel = _add_refs_from_excel_parser(
        refs_norm_sub,
        default_schema="scarlet_refs_norm_v1.0.yaml",
        command_help="Generate refs_norm files from a run-configuration Excel file",
    )
    refs_norm_from_excel.set_defaults(func=_cmd_refs_norm_from_excel)

    r2d = sub.add_parser(
        "reduce-2d",
        help="Run the first deterministic 2D reduction pass",
    )
    r2d.add_argument("sample_scattering", help="Converted SCARLET NXsas_raw sample scattering file")
    r2d.add_argument("refs_sub", help="SCARLET refs_sub bundle for the sample configuration")
    r2d.add_argument("output", help="Output NeXus/HDF5 file. The raw file is copied and /processed_data is added.")
    r2d.add_argument(
        "--sample-transmission",
        default=None,
        help="Converted SCARLET NXsas_raw sample transmission file. If omitted, T_sample=1 is assumed.",
    )
    r2d.add_argument(
        "--refs-norm",
        default=None,
        help="Optional SCARLET refs_norm bundle used for water normalization",
    )
    r2d.add_argument("--detector", type=int, default=None, help="Detector index to reduce (default: all detectors)")
    r2d.add_argument(
        "--normalize-by",
        choices=("monitor", "count_time", "none"),
        default="monitor",
        help="How detector images are normalized before subtraction (default: monitor)",
    )
    r2d.add_argument(
        "--raw-entry",
        default="/raw_data",
        help="Raw-data NXentry in sample files (default: /raw_data; falls back to /entry if absent)",
    )
    r2d.add_argument(
        "--processed-entry",
        default="/processed_data",
        help="NXentry used to store reduced data in the output file (default: /processed_data)",
    )
    r2d.add_argument(
        "--refs-entry",
        default="/entry",
        help="NXentry used by refs_sub/refs_norm bundles (default: /entry)",
    )
    r2d.add_argument("--no-mask", action="store_true", help="Do not apply masks stored in the reference bundles")
    r2d.add_argument("--overwrite", action="store_true", help="Overwrite existing output file or processed entry")
    r2d.set_defaults(func=_cmd_reduce_2d)

    avg = sub.add_parser(
        "azimuthal-average",
        help="Compute an azimuthal I(Q) average from a reduced 2D SCARLET file",
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
    avg.add_argument("--overwrite", action="store_true", help="Overwrite existing output CSV")
    avg.set_defaults(func=_cmd_azimuthal_average)

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
