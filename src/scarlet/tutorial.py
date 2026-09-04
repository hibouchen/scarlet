from __future__ import annotations

import argparse
from importlib import resources
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_TUTORIAL_NAME = "tutorial.ipynb"


def _packaged_tutorial():
    return resources.files("scarlet.notebooks").joinpath(DEFAULT_TUTORIAL_NAME)


def prepare_tutorial_copy(
    destination: Path,
    *,
    overwrite: bool = False,
) -> Path:
    output_path = destination.expanduser().resolve()
    if output_path.is_dir() or output_path.suffix == "":
        output_path = output_path / DEFAULT_TUTORIAL_NAME
    if output_path.exists() and not overwrite:
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with _packaged_tutorial().open("rb") as source, output_path.open("wb") as target:
        shutil.copyfileobj(source, target)
    return output_path


def open_tutorial(
    notebook_path: Path,
    *,
    no_browser: bool = False,
) -> int:
    command = [sys.executable, "-m", "jupyter", "lab", str(notebook_path)]
    if no_browser:
        command.append("--no-browser")
    return int(subprocess.call(command))


class NotebookLauncherDialog:
    def __init__(self) -> None:
        qt = _load_qt()
        self.qt = qt
        self.selected_notebook: Path | None = None

        self.dialog = qt.QDialog()
        self.dialog.setWindowTitle("SCARLET notebook")
        self.dialog.resize(640, 260)

        root = qt.QVBoxLayout(self.dialog)

        tabs = qt.QTabWidget()
        root.addWidget(tabs, 1)

        create_tab = qt.QWidget()
        create_layout = qt.QVBoxLayout(create_tab)
        tabs.addTab(create_tab, "New notebook")

        folder_layout = qt.QHBoxLayout()
        folder_layout.addWidget(qt.QLabel("Folder"))
        self.create_folder = qt.QLineEdit(str(Path.cwd()))
        folder_layout.addWidget(self.create_folder, 1)
        browse_folder = qt.QPushButton("Browse...")
        browse_folder.clicked.connect(self._browse_create_folder)
        folder_layout.addWidget(browse_folder)
        create_layout.addLayout(folder_layout)

        name_layout = qt.QHBoxLayout()
        name_layout.addWidget(qt.QLabel("Notebook name"))
        self.create_name = qt.QLineEdit(DEFAULT_TUTORIAL_NAME)
        name_layout.addWidget(self.create_name, 1)
        create_layout.addLayout(name_layout)

        create_layout.addStretch(1)
        create_button = qt.QPushButton("Create and open")
        create_button.clicked.connect(self._accept_create)
        create_layout.addWidget(create_button)

        open_tab = qt.QWidget()
        open_layout = qt.QVBoxLayout(open_tab)
        tabs.addTab(open_tab, "Existing notebook")

        file_layout = qt.QHBoxLayout()
        file_layout.addWidget(qt.QLabel("Notebook"))
        self.open_file = qt.QLineEdit()
        file_layout.addWidget(self.open_file, 1)
        browse_file = qt.QPushButton("Browse...")
        browse_file.clicked.connect(self._browse_existing_notebook)
        file_layout.addWidget(browse_file)
        open_layout.addLayout(file_layout)

        open_layout.addStretch(1)
        open_button = qt.QPushButton("Open selected notebook")
        open_button.clicked.connect(self._accept_existing)
        open_layout.addWidget(open_button)

        buttons = qt.QDialogButtonBox(qt.QDialogButtonBox.Cancel)
        buttons.rejected.connect(self.dialog.reject)
        root.addWidget(buttons)

    def run(self) -> Path | None:
        result = self.dialog.exec()
        if result != self.qt.QDialog.Accepted:
            return None
        return self.selected_notebook

    def _browse_create_folder(self) -> None:
        directory = self.qt.QFileDialog.getExistingDirectory(
            self.dialog,
            "Select notebook folder",
            self.create_folder.text().strip() or str(Path.cwd()),
        )
        if directory:
            self.create_folder.setText(directory)

    def _browse_existing_notebook(self) -> None:
        filename, _selected_filter = self.qt.QFileDialog.getOpenFileName(
            self.dialog,
            "Open notebook",
            self.open_file.text().strip() or str(Path.cwd()),
            "Jupyter notebooks (*.ipynb);;All files (*)",
        )
        if filename:
            self.open_file.setText(filename)

    def _accept_create(self) -> None:
        folder = Path(self.create_folder.text().strip() or ".").expanduser()
        name = self.create_name.text().strip()
        if not name:
            self._show_error("Notebook name is required.")
            return

        notebook_path = Path(name)
        if notebook_path.name != name:
            self._show_error("Notebook name must be a filename, not a path.")
            return
        if notebook_path.suffix.lower() != ".ipynb":
            notebook_path = Path(f"{name}.ipynb")

        destination = folder / notebook_path
        if destination.exists() and not self._confirm_replace(destination):
            return

        try:
            self.selected_notebook = prepare_tutorial_copy(destination, overwrite=True)
        except OSError as e:
            self._show_error(f"Cannot create notebook:\n{e}")
            return
        self.dialog.accept()

    def _accept_existing(self) -> None:
        raw_path = self.open_file.text().strip()
        if not raw_path:
            self._show_error("Select an existing notebook.")
            return
        notebook_path = Path(raw_path).expanduser()
        notebook_path = notebook_path.resolve()
        if not notebook_path.exists():
            self._show_error(f"Notebook not found:\n{notebook_path}")
            return
        if not notebook_path.is_file() or notebook_path.suffix.lower() != ".ipynb":
            self._show_error("Select a .ipynb notebook file.")
            return

        self.selected_notebook = notebook_path
        self.dialog.accept()

    def _confirm_replace(self, destination: Path) -> bool:
        answer = self.qt.QMessageBox.question(
            self.dialog,
            "Replace notebook",
            f"The notebook already exists:\n{destination}\n\nReplace it?",
            self.qt.QMessageBox.Yes | self.qt.QMessageBox.No,
            self.qt.QMessageBox.No,
        )
        return answer == self.qt.QMessageBox.Yes

    def _show_error(self, message: str) -> None:
        self.qt.QMessageBox.critical(self.dialog, "SCARLET notebook", message)


def _load_qt() -> Any:
    from PySide6 import QtWidgets

    return QtWidgets


def select_notebook_with_gui() -> Path | None:
    qt = _load_qt()
    app = qt.QApplication.instance() or qt.QApplication([])
    dialog = NotebookLauncherDialog()
    notebook_path = dialog.run()
    if notebook_path is None:
        return None
    app.processEvents()
    return notebook_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scarlet-notebook",
        description="Open the packaged SCARLET tutorial notebook.",
    )
    parser.add_argument(
        "destination",
        nargs="?",
        default=None,
        help=(
            "Notebook file or directory to open. Directory paths and paths without "
            "an extension receive tutorial.ipynb. If omitted, ./tutorial.ipynb is "
            "selected from a graphical launcher."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the destination notebook with the packaged tutorial.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Start Jupyter without opening a browser window.",
    )
    parser.add_argument(
        "--copy-only",
        action="store_true",
        help="Copy or locate the tutorial notebook without starting Jupyter.",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    if args.destination is None and not args.copy_only:
        try:
            notebook_path = select_notebook_with_gui()
        except ModuleNotFoundError as e:
            missing = getattr(e, "name", None) or str(e)
            print(f"Missing dependency: {missing}. Install PySide6.", file=sys.stderr)
            return 2
        except OSError as e:
            print(f"Cannot start notebook launcher: {e}", file=sys.stderr)
            return 2
        if notebook_path is None:
            print("Notebook selection cancelled.")
            return 0

        print(f"Tutorial notebook: {notebook_path}")
        try:
            return open_tutorial(notebook_path, no_browser=args.no_browser)
        except OSError as e:
            print(f"Cannot start JupyterLab: {e}", file=sys.stderr)
            return 2

    destination = DEFAULT_TUTORIAL_NAME if args.destination is None else args.destination
    try:
        notebook_path = prepare_tutorial_copy(
            Path(destination),
            overwrite=args.overwrite,
        )
    except OSError as e:
        print(f"Cannot prepare tutorial notebook: {e}", file=sys.stderr)
        return 2

    print(f"Tutorial notebook: {notebook_path}")
    if args.copy_only:
        return 0

    try:
        return open_tutorial(notebook_path, no_browser=args.no_browser)
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", None) or str(e)
        print(f"Missing dependency: {missing}. Install JupyterLab.", file=sys.stderr)
        return 2
    except OSError as e:
        print(f"Cannot start JupyterLab: {e}", file=sys.stderr)
        return 2


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
