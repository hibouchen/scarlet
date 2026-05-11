from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile

import h5py

from scarlet.cli import main
from scarlet.validation.schema_loader import load_schema
from scarlet.validation.schema_validator import validate_nexus_file


def _write_simple_excel(path: Path, rows: list[tuple[str, ...]]) -> None:
    """Write the small subset of .xlsx needed by SCARLET's Excel reader."""

    def column_name(index: int) -> str:
        out: list[str] = []
        while index:
            index, rem = divmod(index - 1, 26)
            out.append(chr(65 + rem))
        return "".join(reversed(out))

    def cell(ref: str, value: str) -> str:
        return f'<c r="{ref}" t="inlineStr"><is><t xml:space="preserve">{escape(value)}</t></is></c>'

    row_chunks: list[str] = []
    for row_index, row in enumerate(rows, start=1):
        cell_chunks = [
            cell(f"{column_name(column_index)}{row_index}", value)
            for column_index, value in enumerate(row, start=1)
        ]
        row_chunks.append(f'<row r="{row_index}">{"".join(cell_chunks)}</row>')

    last_cell = f"{column_name(len(rows[0]))}{len(rows)}"
    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<dimension ref="A1:{last_cell}"/>'
        '<sheetViews><sheetView workbookViewId="0"/></sheetViews>'
        '<sheetFormatPr defaultRowHeight="15"/>'
        f'<sheetData>{"".join(row_chunks)}</sheetData>'
        '</worksheet>'
    )

    with ZipFile(path, "w", compression=ZIP_DEFLATED) as zf:
        zf.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/worksheets/sheet1.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            '<Override PartName="/xl/styles.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
            '</Types>',
        )
        zf.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="xl/workbook.xml"/>'
            '</Relationships>',
        )
        zf.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="runs" sheetId="1" r:id="rId1"/></sheets>'
            '</workbook>',
        )
        zf.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            'Target="worksheets/sheet1.xml"/>'
            '<Relationship Id="rId2" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
            'Target="styles.xml"/>'
            '</Relationships>',
        )
        zf.writestr(
            "xl/styles.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
            '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
            '<borders count="1"><border/></borders>'
            '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
            '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>'
            '</styleSheet>',
        )
        zf.writestr("xl/worksheets/sheet1.xml", sheet_xml)


class TestEndToEndReferences(unittest.TestCase):
    def test_convert_generate_references_and_validate(self) -> None:
        raw_file = Path(__file__).resolve().parent / "data" / "sansllb" / "raw_data" / "sans-llb2025n002339.hdf"
        if not raw_file.exists():
            self.skipTest(f"Missing test input file: {raw_file}")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "converted"
            refs_dir = root / "refs"
            data_dir.mkdir()
            refs_dir.mkdir()

            logical_runs = {
                "dark.nxs": ("Cd", "scattering"),
                "empty_beam_transmission.nxs": ("empty_beam", "transmission"),
                "empty_beam_scattering.nxs": ("empty_beam", "scattering"),
                "empty_cell_transmission.nxs": ("empty_cell", "transmission"),
                "empty_cell_scattering.nxs": ("empty_cell", "scattering"),
                "water_scattering.nxs": ("water", "scattering"),
                "water_transmission.nxs": ("water", "transmission"),
            }

            converted_files: list[Path] = []
            for file_name in logical_runs:
                output = data_dir / file_name
                status = main(["convert", "sansllb", str(raw_file), str(output), "--validate"])
                self.assertEqual(status, 0)
                converted_files.append(output)

            excel_path = root / "run_configuration.xlsx"
            rows = [
                ("data_file", "config_id", "configuration", "sample_name", "measurement_type", "measurement_confidence"),
            ]
            rows.extend(
                (file_name, "config_1", "", sample_name, measurement_type, "1")
                for file_name, (sample_name, measurement_type) in logical_runs.items()
            )
            _write_simple_excel(excel_path, rows)

            refs_sub_status = main([
                "refs-sub",
                "from-excel",
                str(excel_path),
                str(data_dir),
                str(refs_dir),
                "--validate",
            ])
            self.assertEqual(refs_sub_status, 0)

            refs_norm_status = main([
                "refs-norm",
                "from-excel",
                str(excel_path),
                str(data_dir),
                str(refs_dir),
                "--validate",
            ])
            self.assertEqual(refs_norm_status, 0)

            raw_schema = load_schema("scarlet_nxsas_raw_v1.3_mono.yaml")
            for output in converted_files:
                report = validate_nexus_file(output, raw_schema)
                self.assertTrue(report.ok, "\n".join(report.format_lines()))

            refs_sub = refs_dir / "refs_sub_config_1.nxs"
            refs_norm = refs_dir / "refs_norm_config_1.nxs"
            self.assertTrue(refs_sub.exists())
            self.assertTrue(refs_norm.exists())

            refs_sub_report = validate_nexus_file(refs_sub, load_schema("scarlet_refs_sub_v1.0.yaml"))
            self.assertTrue(refs_sub_report.ok, "\n".join(refs_sub_report.format_lines()))
            refs_norm_report = validate_nexus_file(refs_norm, load_schema("scarlet_refs_norm_v1.0.yaml"))
            self.assertTrue(refs_norm_report.ok, "\n".join(refs_norm_report.format_lines()))

            with h5py.File(refs_sub, "r") as f:
                refs = f["/entry/references"]
                self.assertIn("empty_beam_transmission", refs)
                self.assertIn("empty_cell_scattering", refs)
                self.assertIn("dark", refs)

            with h5py.File(refs_norm, "r") as f:
                refs = f["/entry/references"]
                self.assertIn("water_scattering", refs)
                self.assertIn("water_transmission", refs)
                self.assertNotIn("source_config_id", refs["water_scattering"])
                self.assertNotIn("source_config_id", refs["water_transmission"])


if __name__ == "__main__":
    unittest.main()
