import csv
import io
import struct
import sys
import tempfile
import unittest
import zlib
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS / "xbox"))

import carve_shaders

END_PGM = 0xBF810000


def build_fssm(code, offset=0x1D0, size=None, total=None):
    if size is None:
        size = len(code)
    length = total if total is not None else max(offset + len(code), 0x1D0)
    blob = bytearray(length)
    blob[:4] = b"FSSM"
    struct.pack_into("<QQ", blob, 0xC0, offset, size)
    blob[offset : offset + len(code)] = code
    return bytes(blob)


def build_xbch(blob):
    compressed = zlib.compress(blob)
    return (
        b"XBCH"
        + struct.pack("<IIIII", 3, len(compressed), 0, len(blob), 0)
        + compressed
    )


def build_dxil(stage, xbch):
    return b"\x00\x00" + struct.pack("<H", stage) + b"\x00\x00" + xbch


def build_dxbc(chunks):
    table = 32 + 4 * len(chunks)
    offsets = []
    body = b""
    for tag, payload in chunks:
        offsets.append(table + len(body))
        body += tag + struct.pack("<I", len(payload)) + payload
    total = table + len(body)
    header = b"DXBC" + bytes(20) + struct.pack("<II", total, len(chunks))
    return header + struct.pack("<%dI" % len(chunks), *offsets) + body


def raw_deflate(data):
    compressor = zlib.compressobj(9, zlib.DEFLATED, -15)
    return compressor.compress(data) + compressor.flush()


def prologue():
    words = [0] * 10
    words[4] = (0x35 << 26) | (0x180 << 16)
    words[7] = (0x35 << 26) | (0x180 << 16)
    return words


def build_entry(name, dxbc):
    return name.encode() + b"_Wrapped\x00" + bytes(17) + raw_deflate(dxbc)


def build_archive(name, code):
    blob = build_fssm(code)
    dxil = build_dxil(1, build_xbch(blob))
    return build_entry(name, build_dxbc([(b"DXIL", dxil)]))


class CarveFssmTest(unittest.TestCase):
    def carve(self, archive):
        name, offset = carve_shaders.find_entries(archive)[0]
        return carve_shaders.carve_one(name, offset, archive)

    def test_strips_prologue_and_trims_after_endpgm(self):
        code = struct.pack("<10I", *prologue()) + struct.pack(
            "<3I", 0x11111111, END_PGM, 0x22222222
        )
        result = self.carve(build_archive("VS_Test_0001", code))
        self.assertTrue(result["prologue"])
        self.assertEqual(result["kind"], 1)
        self.assertEqual(result["code"], struct.pack("<2I", 0x11111111, END_PGM))

    def test_without_prologue_keeps_the_code(self):
        code = struct.pack("<2I", 0x33333333, END_PGM)
        result = self.carve(build_archive("VS_Test_0002", code))
        self.assertFalse(result["prologue"])
        self.assertEqual(result["code"], code)

    def test_without_endpgm_keeps_the_tail(self):
        code = struct.pack("<10I", *prologue()) + struct.pack(
            "<2I", 0x44444444, 0x55555555
        )
        result = self.carve(build_archive("VS_Test_0003", code))
        self.assertEqual(result["code"], struct.pack("<2I", 0x44444444, 0x55555555))

    def test_short_fssm_is_an_entry_error(self):
        blob = bytearray(0xCF)
        blob[:4] = b"FSSM"
        dxil = build_dxil(1, build_xbch(bytes(blob)))
        archive = build_entry("VS_Test_0004", build_dxbc([(b"DXIL", dxil)]))
        self.assertEqual(self.carve(archive)["error"], "short-fssm")

    def test_header_between_the_code_range_and_the_decoder_region_is_an_entry_error(self):
        # issue #128: 0xD0..0x1CF holds the code range but not the wave word (0xD4) or the input tables (to 0x1D0)
        for size in (0xD0, 0xD4, 0x1CF):
            blob = bytearray(size)
            blob[:4] = b"FSSM"
            struct.pack_into("<QQ", blob, 0xC0, 0x1D0, 0)
            dxil = build_dxil(1, build_xbch(bytes(blob)))
            archive = build_entry("VS_Test_0004", build_dxbc([(b"DXIL", dxil)]))
            self.assertEqual(self.carve(archive)["error"], "short-fssm", size)

    def test_code_range_past_end_is_an_entry_error(self):
        code = struct.pack("<2I", 0x66666666, END_PGM)
        blob = build_fssm(code, size=len(code) + 10)
        dxil = build_dxil(1, build_xbch(blob))
        archive = build_entry("VS_Test_0005", build_dxbc([(b"DXIL", dxil)]))
        self.assertEqual(self.carve(archive)["error"], "short-code")

    def test_main_writes_the_blob_and_the_manifest(self):
        code = struct.pack("<10I", *prologue()) + struct.pack(
            "<3I", 0x77777777, END_PGM, 0x88888888
        )
        archive = build_archive("VS_Test_0006", code)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "shaders.rpf"
            source.write_bytes(archive)
            out = root / "out"
            argv = ["carve_shaders.py", str(source), "--out", str(out)]
            with mock.patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                carve_shaders.main()
            self.assertEqual(
                (out / "bins" / "1.bin").read_bytes(),
                struct.pack("<2I", 0x77777777, END_PGM),
            )
            with (out / "manifest.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["idx"], "1")
            self.assertEqual(row["name"], "VS_Test_0006")
            self.assertEqual(row["kind"], "1")
            self.assertEqual(row["stage"], "VS")
            self.assertEqual(row["prologue"], "1")
            self.assertEqual(row["error"], "")


class TrimAtEndpgmTest(unittest.TestCase):
    def test_trims_after_the_last_endpgm(self):
        code = struct.pack("<4I", 0x11111111, END_PGM, 0x22222222, 0x33333333)
        self.assertEqual(carve_shaders.trim_at_endpgm(code), code[:8])

    def test_trailing_partial_word_does_not_hide_an_aligned_endpgm(self):
        for tail in (b"\x00", b"\x00\x00", b"\x00\x00\x00"):
            with self.subTest(tail=len(tail)):
                code = struct.pack("<3I", 0x11111111, END_PGM, 0x22222222) + tail
                self.assertEqual(carve_shaders.trim_at_endpgm(code), code[:8])

    def test_endpgm_bytes_at_an_unaligned_offset_do_not_match(self):
        code = b"\x00\x00" + struct.pack("<I", END_PGM) + b"\x00\x00\x00"
        self.assertEqual(carve_shaders.trim_at_endpgm(code), code)

    def test_code_shorter_than_a_word_is_kept(self):
        for code in (b"", b"\x00\x00\x81"):
            with self.subTest(size=len(code)):
                self.assertEqual(carve_shaders.trim_at_endpgm(code), code)


if __name__ == "__main__":
    unittest.main()
