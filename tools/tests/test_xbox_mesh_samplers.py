import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "xbox"))
os.environ.setdefault("XBOX_RDNA2", "recomp")
import rdna_asm


def mesh_shader(sampler_count, integer_image=False):
    program = rdna_asm.Program()
    bindings = ["node 0 user 8", "desc 0 0 img1", "desc 0 32 buf"]
    program.smem("s_load_dwordx8", 16, 8, 0)
    program.smem("s_load_dwordx4", 32, 8, 32)
    images = [16]
    if integer_image:
        bindings.append("desc 0 64 img1a")
        program.smem("s_load_dwordx8", 40, 8, 64)
        images.append(40)
    program.vop1("v_mov_b32", 1, ("lit", 0x3F000000))
    program.vop1("v_mov_b32", 2, "v1")
    for index in range(sampler_count):
        offset = 96 + index * 16
        bindings.append(f"desc 0 {offset} samp")
        program.smem("s_load_dwordx4", 24, 8, offset)
        program.sopp("s_waitcnt", 0)
        for image_index, image in enumerate(images):
            program.mimg("image_sample_lz", 4, 1, image, dmask=1, ssamp=24)
            program.sopp("s_waitcnt", 0)
            program.mubuf("buffer_store_dword", 4, 0, 32, offset=8 * index + 4 * image_index, offen=False)
    program.sopp("s_endpgm")
    return program.bytes(), "\n".join(bindings) + "\n"


class MeshSamplerTest(unittest.TestCase):
    def setUp(self):
        name = "driver.exe" if os.name == "nt" else "driver"
        self.driver = Path(os.environ.get("XBOX_DRIVER", ROOT / "build-xbox" / name)).resolve()
        if not self.driver.is_file():
            self.skipTest("Xbox driver is not built")
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def compile_mesh(self, sampler_count, integer_image=False):
        code, bindings = mesh_shader(sampler_count, integer_image)
        shader = self.root / "mesh.bin"
        bind = self.root / "mesh.bind"
        output = self.root / "mesh.spv"
        shader.write_bytes(code)
        bind.write_text(bindings, encoding="utf-8")
        result = subprocess.run(
            [str(self.driver), str(shader), "64", str(output), str(bind), "1", "ms"],
            env={**os.environ, "XBOX_DUMP_ABI": "1"},
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result, output

    def test_46_samplers_fit_after_specialization(self):
        result, output = self.compile_mesh(46)
        self.assertEqual(result.returncode, 0, result.stdout.split(" RecompileRequest:")[0])
        self.assertRegex(result.stdout, r"(?m)^  binding set=0 b=57 .* count=46 guest=184 ")
        self.assertTrue(output.is_file())
        val = os.environ.get("XBOX_SPIRV_VAL") or shutil.which("spirv-val")
        if val:
            subprocess.run([val, "--target-env", "vulkan1.3", str(output)], check=True, timeout=30)

    def test_more_than_64_active_sampler_entries_are_rejected(self):
        result, output = self.compile_mesh(33, integer_image=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("prepared sampler heap capacity exceeded", result.stdout.split(" RecompileRequest:")[0])
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
