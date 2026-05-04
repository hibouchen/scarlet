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


def _cmd_validate(args: argparse.Namespace) -> int:
    try:
        schema = load_schema(args.schema)
    except FileNotFoundError:
        print(f"Schema not found: {args.schema!r}", file=sys.stderr)
        return 2

    try:
        report: ValidationReport = validate_nexus_file(
            file_path=Path(args.file),
            schema=schema,
            entry_path=args.entry,
            strict=args.strict,
        )
    except OSError as e:
        print(f"Cannot open file {args.file!r}: {e}", file=sys.stderr)
        return 2

    for msg in report.format_lines():
        print(msg)

    if report.ok:
        print("OK")
        return 0

    print(f"FAILED: {len(report.errors)} error(s), {len(report.warnings)} warning(s)")
    return 1


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

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
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
