import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS / "xbox"))

with mock.patch.dict(os.environ, {"XBOX_RDNA2": "recomp"}):
    import run_full


class RunFullWaveCacheTest(unittest.TestCase):
    def test_manifest_wave_is_part_of_the_cache_key(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "1.bin").write_bytes(struct.pack("<I", 0xBF810000))
            (root / "1.bind").write_text("synthetic bindings", encoding="utf-8")
            driver = root / "driver"
            driver.write_bytes(b"synthetic driver identity")
            validator = root / "validator"
            validator.write_bytes(b"synthetic validator identity")
            row = {"idx": "1", "stage": "CS"}
            with mock.patch.multiple(
                run_full,
                BINS=str(root) + os.sep,
                BINDS=str(root) + os.sep,
                VIN=str(root) + os.sep,
                DRIVER_CS=str(driver),
                VAL=str(validator),
            ):
                self.assertNotEqual(
                    run_full.cache_key(dict(row, wave="32"), False),
                    run_full.cache_key(dict(row, wave="64"), False),
                )
                self.assertEqual(
                    run_full.cache_key(row, False),
                    run_full.cache_key(dict(row, wave=""), False),
                )


if __name__ == "__main__":
    unittest.main()
