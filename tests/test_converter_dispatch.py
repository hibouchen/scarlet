from __future__ import annotations

import unittest

from scarlet.io.converters import get_converter, list_apparatus


class TestConverterDispatch(unittest.TestCase):
    def test_list_apparatus(self) -> None:
        self.assertIn("sam", list_apparatus())
        self.assertIn("sansllb", list_apparatus())

    def test_get_converter_accepts_aliases(self) -> None:
        self.assertTrue(callable(get_converter("sam")))
        self.assertTrue(callable(get_converter("SANSLLB")))
        self.assertTrue(callable(get_converter("sans-llb")))
        self.assertTrue(callable(get_converter("sans_llb")))

    def test_get_converter_rejects_unknown(self) -> None:
        with self.assertRaises(ValueError):
            get_converter("nope")

