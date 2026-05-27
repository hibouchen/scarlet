from __future__ import annotations

from pathlib import Path
from typing import Union
from zipfile import ZipFile
import xml.etree.ElementTree as ET


_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _column_index_from_ref(ref: str) -> int:
    letters = "".join(ch for ch in ref if ch.isalpha()).upper()
    index = 0
    for letter in letters:
        index = index * 26 + (ord(letter) - 64)
    return max(0, index - 1)


def read_xlsx_rows(path: Union[str, Path], *, sheet_path: str = "xl/worksheets/sheet1.xml") -> list[dict[str, str]]:
    """
    Read the first row as headers and return the remaining worksheet rows as dicts.

    The parser supports inline strings, shared strings, and sparse cell references.
    It intentionally stays within the small subset needed by SCARLET workflows and
    notebooks, without requiring third-party spreadsheet dependencies.
    """
    path = Path(path)
    with ZipFile(path, "r") as zf:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            shared_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            shared_strings = ["".join(node.itertext()) for node in shared_root.findall("a:si", _NS)]
        sheet_root = ET.fromstring(zf.read(sheet_path))

    raw_rows: list[list[str]] = []
    max_columns = 0
    for row in sheet_root.findall(".//a:sheetData/a:row", _NS):
        values_by_index: dict[int, str] = {}
        next_index = 0
        for cell in row.findall("a:c", _NS):
            ref = cell.get("r")
            index = _column_index_from_ref(ref) if ref else next_index
            next_index = index + 1

            cell_type = cell.get("t")
            if cell_type == "inlineStr":
                inline = cell.find("a:is", _NS)
                value = "" if inline is None else "".join(inline.itertext())
            else:
                scalar = cell.find("a:v", _NS)
                value = "" if scalar is None or scalar.text is None else scalar.text
                if cell_type == "s" and value:
                    value = shared_strings[int(value)]

            values_by_index[index] = value

        if not values_by_index:
            continue

        row_values = [values_by_index.get(i, "") for i in range(max(values_by_index) + 1)]
        raw_rows.append(row_values)
        max_columns = max(max_columns, len(row_values))

    if not raw_rows:
        return []

    normalized_rows = [row + [""] * (max_columns - len(row)) for row in raw_rows]
    header = normalized_rows[0]
    return [
        {header[i]: row[i] for i in range(len(header))}
        for row in normalized_rows[1:]
        if any(value != "" for value in row)
    ]


__all__ = ["read_xlsx_rows"]
