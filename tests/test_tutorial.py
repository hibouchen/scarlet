from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scarlet.tutorial import main, prepare_tutorial_copy


class TestTutorial(unittest.TestCase):
    def test_prepare_tutorial_copy_copies_packaged_notebook(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "tutorial.ipynb"

            result = prepare_tutorial_copy(destination)

            self.assertEqual(result, destination.resolve())
            self.assertTrue(destination.exists())
            with destination.open("r", encoding="utf-8") as handle:
                notebook = json.load(handle)
            self.assertIn("cells", notebook)

    def test_prepare_tutorial_copy_reuses_existing_destination(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "tutorial.ipynb"
            destination.write_text("existing notebook", encoding="utf-8")

            result = prepare_tutorial_copy(destination)

            self.assertEqual(result, destination.resolve())
            self.assertEqual(destination.read_text(encoding="utf-8"), "existing notebook")

    def test_prepare_tutorial_copy_treats_suffixless_destination_as_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "session"

            result = prepare_tutorial_copy(destination)

            self.assertEqual(result, destination.resolve() / "tutorial.ipynb")
            self.assertTrue(result.exists())

    def test_main_opens_tutorial_with_jupyter_lab(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            expected_notebook = Path(tmp).resolve() / "tutorial.ipynb"
            with mock.patch("scarlet.tutorial.subprocess.call", return_value=0) as call:
                status = main([tmp, "--no-browser"])

            self.assertEqual(status, 0)
            call.assert_called_once()
            command = call.call_args.args[0]
            self.assertEqual(command[1:4], ["-m", "jupyter", "lab"])
            self.assertEqual(command[4:], [str(expected_notebook), "--no-browser"])

    def test_main_uses_gui_launcher_when_no_destination(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            notebook = Path(tmp) / "selected.ipynb"
            notebook.write_text("{}", encoding="utf-8")
            with mock.patch("scarlet.tutorial.select_notebook_with_gui", return_value=notebook) as select:
                with mock.patch("scarlet.tutorial.subprocess.call", return_value=0) as call:
                    status = main([])

            self.assertEqual(status, 0)
            select.assert_called_once_with()
            command = call.call_args.args[0]
            self.assertEqual(command[1:4], ["-m", "jupyter", "lab"])
            self.assertEqual(command[4:], [str(notebook)])

    def test_main_returns_ok_when_gui_selection_is_cancelled(self) -> None:
        with mock.patch("scarlet.tutorial.select_notebook_with_gui", return_value=None):
            with mock.patch("scarlet.tutorial.subprocess.call") as call:
                status = main([])

        self.assertEqual(status, 0)
        call.assert_not_called()


if __name__ == "__main__":
    unittest.main()
