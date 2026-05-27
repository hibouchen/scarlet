from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scarlet.io.edf import read_edf_image, read_edf_mask


def _write_simple_edf(path: Path, data: np.ndarray) -> None:
    data = np.asarray(data, dtype=np.uint8)
    header_size = 512
    header = (
        "{\n"
        "EDF_DataBlockID = 0.Image.Psd ;\n"
        f"EDF_BinarySize = {data.size} ;\n"
        f"EDF_HeaderSize = {header_size:5d} ;\n"
        "ByteOrder = LowByteFirst ;\n"
        "DataType = UnsignedByte ;\n"
        f"Dim_1 = {data.shape[1]} ;\n"
        f"Dim_2 = {data.shape[0]} ;\n"
        "Image = 0 ;\n"
        "masked_value = nonzero ;\n"
    )
    padded_header = (header + (" " * header_size))[: header_size - 2] + "}\n"
    path.write_bytes(padded_header.encode("ascii") + data.tobytes())


class TestEdfReader(unittest.TestCase):
    def test_reads_temp_edf_image_and_mask(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "mask.edf"
            source = np.zeros((4, 5), dtype=np.uint8)
            source[1, 2] = 7
            _write_simple_edf(path, source)

            image = read_edf_image(path)
            mask = read_edf_mask(path)

            np.testing.assert_array_equal(image, source)
            np.testing.assert_array_equal(mask, (source != 0).astype(np.uint8))

    def test_reads_repository_examples(self) -> None:
        root = Path(__file__).resolve().parent.parent
        for name, nonzero_count in (("mask_GQ.edf", 978), ("mask_PQ.edf", 210)):
            path = root / "data" / "SANSLLB" / "raw" / name
            image = read_edf_image(path)
            mask = read_edf_mask(path)
            self.assertEqual(image.shape, (128, 128))
            self.assertEqual(mask.shape, (128, 128))
            self.assertEqual(int(np.count_nonzero(mask)), nonzero_count)


if __name__ == "__main__":
    unittest.main()
