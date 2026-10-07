from __future__ import annotations

import argparse
from pathlib import Path
import sys

from PySide6 import QtCore, QtWidgets

from .main_window import ScarletMainWindow


def create_application(argv: list[str] | None = None) -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv if argv is None else argv)
    app.setApplicationName("SCARLET")
    app.setOrganizationName("SCARLET")
    return app


def run_gui(
    *,
    initial_raw_directory: str | Path | None = None,
    initial_workflow: str | Path | None = None,
    initial_instrument: str = "sansllb",
) -> int:
    if initial_raw_directory is not None and initial_workflow is not None:
        raise ValueError("Choose either an initial raw directory or an initial workflow, not both")
    app = create_application()
    window = ScarletMainWindow(
        initial_raw_directory=initial_raw_directory,
        initial_instrument=initial_instrument,
    )
    window.show()
    if initial_workflow is not None:
        workflow_path = Path(initial_workflow)
        QtCore.QTimer.singleShot(0, lambda: window.open_workflow(workflow_path))
    return int(app.exec())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scarlet-gui", description="Open the SCARLET workflow application")
    parser.add_argument("raw_directory", nargs="?", help="Raw acquisition folder to preselect")
    parser.add_argument("--workflow", help="Saved SCARLET workflow to open")
    parser.add_argument(
        "--instrument",
        choices=("sam", "sansllb"),
        default="sansllb",
        help="Initial instrument selection (default: sansllb)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_gui(
        initial_raw_directory=args.raw_directory,
        initial_workflow=args.workflow,
        initial_instrument=args.instrument,
    )


if __name__ == "__main__":
    raise SystemExit(main())

