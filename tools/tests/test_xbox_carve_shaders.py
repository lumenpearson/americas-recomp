import io
import struct
import sys
import tempfile
import unittest
import zlib
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS / "xbox"))

import carve_shaders


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


def build_xbch(stream, uncompressed, zero=0, compressed=None):
    if compressed is None:
        compressed = len(stream)
    return (
        b"XBCH" + struct.pack("<IIIII", 3, compressed, zero, uncompressed, 0) + stream
    )


class FindEntriesTest(unittest.TestCase):
    def test_finds_name_and_payload_offset(self):
        archive = b"\x00Mesh_PS_Wrapped\x00" + bytes(32)
        suffix = archive.index(carve_shaders.SUFFIX)
        self.assertEqual(
            carve_shaders.find_entries(archive),
            [("Mesh_PS", suffix + carve_shaders.PAYLOAD_OFFSET)],
        )

    def test_skips_name_longer_than_255_bytes(self):
        archive = b"\x00" + b"a" * 256 + b"_Wrapped\x00" + bytes(32)
        self.assertEqual(carve_shaders.find_entries(archive), [])

    def test_invalid_character_stops_the_name(self):
        archive = b"\x00bad$name_Wrapped\x00" + bytes(32)
        suffix = archive.index(carve_shaders.SUFFIX)
        self.assertEqual(
            carve_shaders.find_entries(archive),
            [("name", suffix + carve_shaders.PAYLOAD_OFFSET)],
        )

    def test_invalid_character_only_name_is_skipped(self):
        archive = b"\x00$_Wrapped\x00" + bytes(32)
        self.assertEqual(carve_shaders.find_entries(archive), [])

    def test_no_suffix_gives_no_entries(self):
        self.assertEqual(carve_shaders.find_entries(b"DXBC" + bytes(64)), [])


class ParseDxbcTest(unittest.TestCase):
    def test_minimal_container_lists_chunks(self):
        container = build_dxbc([(b"DXIL", b"\x01\x00\x03\x00")])
        self.assertEqual(carve_shaders.parse_dxbc(container), {b"DXIL": (44, 4)})

    def test_total_must_match_length(self):
        container = bytearray(build_dxbc([(b"DXIL", b"\x00\x00\x00\x00")]))
        struct.pack_into("<I", container, 24, len(container) + 1)
        self.assertIsNone(carve_shaders.parse_dxbc(bytes(container)))

    def test_count_above_limit_is_rejected(self):
        container = bytearray(200)
        container[:4] = b"DXBC"
        struct.pack_into("<II", container, 24, len(container), 33)
        self.assertIsNone(carve_shaders.parse_dxbc(bytes(container)))

    def test_chunk_offset_past_end_is_rejected(self):
        container = bytearray(build_dxbc([(b"DXIL", b"\x00\x00\x00\x00")]))
        struct.pack_into("<I", container, 32, len(container) - 4)
        self.assertIsNone(carve_shaders.parse_dxbc(bytes(container)))

    def test_chunk_size_past_end_is_rejected(self):
        container = bytearray(build_dxbc([(b"DXIL", b"\x00\x00\x00\x00")]))
        offset = struct.unpack_from("<I", container, 32)[0]
        struct.pack_into("<I", container, offset + 4, len(container) - offset - 7)
        self.assertIsNone(carve_shaders.parse_dxbc(bytes(container)))


class FssmFromTest(unittest.TestCase):
    def test_without_marker_returns_none(self):
        self.assertIsNone(carve_shaders.fssm_from(b"JUNK" * 8))

    def test_nonzero_reserved_word_returns_none(self):
        nested = build_xbch(zlib.compress(b"FSSM"), 4, zero=1)
        self.assertIsNone(carve_shaders.fssm_from(nested))

    def test_zero_uncompressed_size_returns_none(self):
        nested = build_xbch(zlib.compress(b"FSSM"), 0)
        self.assertIsNone(carve_shaders.fssm_from(nested))

    def test_zero_compressed_size_returns_none(self):
        nested = build_xbch(b"", 4, compressed=0)
        self.assertIsNone(carve_shaders.fssm_from(nested))

    def test_compressed_size_past_end_returns_none(self):
        stream = zlib.compress(b"FSSM")
        nested = build_xbch(stream, 4, compressed=len(stream) + 1)
        self.assertIsNone(carve_shaders.fssm_from(nested))

    def test_broken_zlib_stream_returns_none(self):
        stream = b"\x78\x9c" + b"\xff" * 8
        nested = build_xbch(stream, 4)
        self.assertIsNone(carve_shaders.fssm_from(nested))

    def test_valid_stream_is_inflated(self):
        payload = b"FSSM" + struct.pack("<QQ", 0xC0, 16) + bytes(8)
        nested = build_xbch(zlib.compress(payload), len(payload))
        self.assertEqual(carve_shaders.fssm_from(nested), payload)


class MainTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.out = self.root / "out"
        self.bins = self.out / "bins"
        self.bins.mkdir(parents=True)
        self.archive = self.root / "shaders.rpf"

    def tearDown(self):
        self.directory.cleanup()

    def run_main(self, *arguments):
        argv = ["carve_shaders.py", str(self.archive), "--out", str(self.out)]
        argv.extend(arguments)
        output = io.StringIO()
        with (
            mock.patch.object(sys, "argv", argv),
            redirect_stdout(output),
            redirect_stderr(io.StringIO()),
        ):
            carve_shaders.main()
        return output.getvalue()

    def test_existing_blobs_without_force_abort(self):
        self.archive.write_bytes(b"")
        kept = self.bins / "1.bin"
        kept.write_bytes(b"\x00")
        with self.assertRaises(SystemExit) as raised:
            self.run_main()
        self.assertEqual(raised.exception.code, 2)
        self.assertTrue(kept.exists())
        self.assertFalse((self.out / "manifest.csv").exists())

    def test_existing_vertex_inputs_or_manifest_without_force_abort(self):
        self.archive.write_bytes(b"\x00")
        vins = self.out / "vin"
        vins.mkdir()
        for kept in (vins / "1.vin", self.out / "manifest.csv"):
            with self.subTest(path=kept.name):
                kept.write_bytes(b"previous run")
                with self.assertRaises(SystemExit) as raised:
                    self.run_main()
                self.assertEqual(raised.exception.code, 2)
                self.assertEqual(kept.read_bytes(), b"previous run")
                kept.unlink()

    def test_force_preserves_bins_when_vertex_output_directory_is_unusable(self):
        self.archive.write_bytes(b"\x00")
        (self.out / "vin").write_bytes(b"not a directory")
        kept = self.bins / "1.bin"
        kept.write_bytes(b"previous run")
        with self.assertRaises(FileExistsError):
            self.run_main("--force")
        self.assertEqual(kept.read_bytes(), b"previous run")

    def test_force_preserves_outputs_when_a_destination_is_a_directory(self):
        self.archive.write_bytes(b"\x00")
        vins = self.out / "vin"
        vins.mkdir()
        outputs = (self.bins / "1.bin", vins / "1.vin")
        for blocked in (self.bins / "2.bin", vins / "2.vin", self.out / "manifest.csv"):
            with self.subTest(blocked=blocked.name):
                for path in outputs:
                    path.write_bytes(b"previous run")
                blocked.mkdir()
                try:
                    with self.assertRaises(SystemExit) as raised:
                        self.run_main("--force")
                    self.assertEqual(raised.exception.code, 2)
                    for path in outputs:
                        self.assertEqual(path.read_bytes(), b"previous run")
                finally:
                    blocked.rmdir()

    def test_force_preserves_outputs_when_archive_cannot_be_mapped(self):
        vins = self.out / "vin"
        vins.mkdir()
        outputs = (self.bins / "1.bin", vins / "1.vin", self.out / "manifest.csv")
        for archive_exists, error in ((False, FileNotFoundError), (True, ValueError)):
            with self.subTest(archive_exists=archive_exists):
                for path in outputs:
                    path.write_bytes(b"previous run")
                if archive_exists:
                    self.archive.touch()
                with self.assertRaises(error):
                    self.run_main("--force")
                for path in outputs:
                    with self.subTest(path=path.name):
                        self.assertTrue(path.is_file())
                        self.assertEqual(path.read_bytes(), b"previous run")

    def test_force_refuses_to_replace_the_input_archive(self):
        for archive in (self.bins / "1.bin", self.out / "manifest.csv"):
            with self.subTest(archive=archive.name):
                self.archive = archive
                archive.write_bytes(b"\x00")
                with self.assertRaises(SystemExit) as raised:
                    self.run_main("--force")
                self.assertEqual(raised.exception.code, 2)
                self.assertEqual(archive.read_bytes(), b"\x00")
                archive.unlink()

    def test_force_preserves_previous_run_if_generating_new_outputs_fails(self):
        self.archive.write_bytes(b"\x00")
        vins = self.out / "vin"
        vins.mkdir()
        outputs = (self.bins / "1.bin", vins / "1.vin", self.out / "manifest.csv")
        for path in outputs:
            path.write_bytes(b"previous run")
        generated = {"error": None, "name": "first", "kind": 1, "code": b"new", "vin": "vin\n", "prologue": False, "wave": 64}
        with (
            mock.patch.object(carve_shaders, "find_entries", return_value=[("first", 0), ("second", 0)]),
            mock.patch.object(carve_shaders, "carve_one", side_effect=[generated, OSError("read failed")]),
            self.assertRaisesRegex(OSError, "read failed"),
        ):
            self.run_main("--force")
        for path in outputs:
            self.assertEqual(path.read_bytes(), b"previous run")
        self.assertEqual(set(self.out.rglob("*")), {self.bins, vins, *outputs})

    def test_force_publishes_generated_blobs_vertex_inputs_and_manifest(self):
        code = struct.pack("<I", carve_shaders.END_PGM)
        blob = bytearray(0x1D0)
        blob[:4] = b"FSSM"
        struct.pack_into("<QQ", blob, 0xC0, len(blob), len(code))
        blob[0x138:0x13D] = bytes((6, 2, 3, 4, 1))
        blob.extend(code)
        payload = build_dxbc([(b"DXIL", b"\x00\x00\x01\x00" + build_xbch(zlib.compress(blob), len(blob)))])
        compressor = zlib.compressobj(wbits=-15)
        compressed = compressor.compress(payload) + compressor.flush()
        self.archive.write_bytes(b"\x00VS_Test" + carve_shaders.SUFFIX + bytes(17) + compressed)
        (self.bins / "1.bin").write_bytes(b"old")
        (self.bins / "2.bin").write_bytes(b"stale")
        vins = self.out / "vin"
        vins.mkdir()
        (vins / "1.vin").write_text("old")

        self.run_main("--force")

        self.assertEqual((self.bins / "1.bin").read_bytes(), code)
        self.assertFalse((self.bins / "2.bin").exists())
        self.assertEqual((vins / "1.vin").read_text(), "vin 4 2 2\n")
        manifest = (self.out / "manifest.csv").read_text().splitlines()
        self.assertEqual(manifest, [",".join(carve_shaders.FIELDS), "1,VS_Test,1,VS,4,0,64,"])

    def test_force_removes_only_numbered_blobs(self):
        self.archive.write_bytes(b"\x00")
        managed = ("1.bin", "10.bin")
        untouched = ("keep.txt", "1.bin.bak", "x.bin")
        for name in managed + untouched:
            (self.bins / name).write_bytes(b"old")
        vins = self.out / "vin"
        vins.mkdir()
        (vins / "1.vin").write_bytes(b"old")
        (vins / "keep.vin").write_bytes(b"keep")
        manifest_path = self.out / "manifest.csv"
        manifest_path.write_bytes(b"old manifest")

        output = self.run_main("--force")

        self.assertFalse((vins / "1.vin").exists())
        self.assertEqual((vins / "keep.vin").read_bytes(), b"keep")
        for name in managed:
            self.assertFalse((self.bins / name).exists(), name)
        for name in untouched:
            self.assertTrue((self.bins / name).exists(), name)
        manifest = manifest_path.read_text().splitlines()
        self.assertEqual(manifest, [",".join(carve_shaders.FIELDS)])
        self.assertIn("names found: 0", output)


if __name__ == "__main__":
    unittest.main()
