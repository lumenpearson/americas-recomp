import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

RUN_FULL = Path(__file__).resolve().parents[1] / "xbox" / "run_full.py"


class RunFullTest(unittest.TestCase):
    def test_carve_error_rows_do_not_abort_the_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "bins").mkdir()
            (root / "binds").mkdir()
            (root / "bins" / "1.bin").write_bytes(struct.pack("<I", 0xBF810000))
            (root / "manifest.csv").write_text(
                "idx,name,kind,stage,bytes,prologue,error\n"
                ",Bad,1,,,,short-fssm\n"
                "1,Good,3,HS,4,0,\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [sys.executable, "-I", str(RUN_FULL), "results.json", "--no-canary", "--no-cache"],
                cwd=root,
                env={**os.environ, "XBOX_CORPUS": str(root), "XBOX_BUILD": str(root), "XBOX_RDNA2": "recomp"},
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = json.loads((root / "results.json").read_text(encoding="utf-8"))
            self.assertEqual([(row["idx"], row["status"]) for row in rows], [("1", "SKIP")])


if __name__ == "__main__":
    unittest.main()
