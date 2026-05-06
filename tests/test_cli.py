from __future__ import annotations

import unittest

from scarlet.cli import main


class TestCli(unittest.TestCase):
    def test_schema_list_command(self) -> None:
        self.assertEqual(main(["schema", "list"]), 0)

    def test_convert_list_command(self) -> None:
        self.assertEqual(main(["convert", "list"]), 0)

    def test_unknown_converter_returns_usage_error(self) -> None:
        status = main(["convert", "unknown", "input.nxs", "output.nxs"])
        self.assertEqual(status, 2)


if __name__ == "__main__":
    unittest.main()
