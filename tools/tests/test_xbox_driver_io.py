import os
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / "build-xbox" / ("driver.exe" if os.name == "nt" else "driver")


@unittest.skipUnless(DRIVER.is_file(), "build the Xbox driver in build-xbox first")
class DriverIoTest(unittest.TestCase):
    def test_reports_failure_when_spirv_output_cannot_be_written(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shader = root / "shader.bin"
            shader.write_bytes(struct.pack("<I", 0xBF810000))
            output = root / "missing" / "shader.spv"
            result = subprocess.run(
                [str(DRIVER), str(shader), "32", str(output)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertEqual(result.stdout.strip(), "IOERROR cannot write SPIR-V output")


if __name__ == "__main__":
    unittest.main()
