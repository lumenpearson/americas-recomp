"""Carve the shader blobs out of an RPF9 ``shaders.rpf`` without its table of contents.

Every shader entry in the archive is inline and self-describing::

    <name> "_Wrapped" 0x00 <17-byte header> <raw deflate stream -> DXBC>

The name has no stored length in front of it, so it is read backwards over the characters an
asset name can contain and then validated by the payload. The raw deflate stream starts 26
bytes after the first byte of the ``_Wrapped`` suffix (8 bytes of suffix, the NUL and the
17-byte header) and inflates to a DXBC container (``wbits=-15``). Inside it the ``DXIL`` chunk is a nested
container whose ``XBCH`` section holds a zlib stream::

    "XBCH" <u32 3> <u32 compressedSize> <u32 0> <u32 uncompressedSize> <u32 0> <zlib stream>

That stream inflates to an ``FSSM`` blob whose code range is ``u64 offset`` at ``0xC0`` and
``u64 size`` at ``0xC8``. Most shaders carry a 10-dword prologue in front of the code, and
constant pools follow the last ``s_endpgm``; both are removed here so the result is what
``driver`` and ``run_batch.py`` expect. The program kind is the ``u16`` at offset 2 of the
``DXIL`` chunk.

Writes ``<out>/bins/<idx>.bin`` and ``<out>/manifest.csv`` and needs nothing but the
archive: no table of contents, no decoder, no game executable. For each vertex shader with an
input table in its FSSM header it also writes ``<out>/vin/<idx>.vin`` (``fssm_header.vin_text``),
the VGPRs the runtime-built fetch subroutine fills; ``driver`` takes it as the ``vs`` input layout.
"""

import argparse
import collections
import csv
import mmap
import os
import re
import struct
import sys
import tempfile
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fssm_header import vin_text

SUFFIX = b"_Wrapped\x00"
PAYLOAD_OFFSET = 26
PROLOGUE_DWORDS = 10
PROLOGUE_OPCODE = 0x180
PROLOGUE_ENCODING = 0x35
END_PGM = 0xBF810000
STAGES = {
    0: "PS",
    1: "VS",
    2: "GS",
    3: "HS",
    4: "DS",
    5: "CS",
    6: "MS",
    7: "MS",
    13: "MS",
}
CHUNK_LIMIT = 1 << 24
READ_LIMIT = 1 << 22
NAME = re.compile(rb"[A-Za-z0-9_.:]+$")
MANAGED = re.compile(r"[0-9]+\.bin")
MANAGED_VIN = re.compile(r"[0-9]+\.vin")
FIELDS = ["idx", "name", "kind", "stage", "bytes", "prologue", "wave", "error"]
WAVE_WORD = 0xD4
WAVE32_BIT = 5


def find_entries(archive):
    entries = []
    position = 0
    while True:
        suffix = archive.find(SUFFIX, position)
        if suffix < 0:
            return entries
        start = suffix
        while start > 0:
            previous = archive[start - 1 : start]
            if not (previous.isalnum() or previous in b"_.:"):
                break
            start -= 1
        position = suffix + 1
        name = bytes(archive[start:suffix])
        if not name or len(name) > 255 or NAME.fullmatch(name) is None:
            continue
        entries.append((name.decode("utf-8"), suffix + PAYLOAD_OFFSET))


def parse_dxbc(container):
    if len(container) < 32 or container[:4] != b"DXBC":
        return None
    total, count = struct.unpack_from("<II", container, 24)
    if total != len(container) or count > 32 or 32 + 4 * count > len(container):
        return None
    chunks = {}
    for offset in struct.unpack_from("<%dI" % count, container, 32):
        if offset + 8 > len(container):
            return None
        size = struct.unpack_from("<I", container, offset + 4)[0]
        if offset + 8 + size > len(container):
            return None
        chunks[container[offset : offset + 4]] = (offset + 8, size)
    return chunks


def fssm_from(nested):
    marker = nested.find(b"XBCH")
    if marker < 0 or marker + 24 > len(nested):
        return None
    compressed, zero, uncompressed = struct.unpack_from("<III", nested, marker + 8)
    if zero != 0 or not 0 < uncompressed <= CHUNK_LIMIT or compressed == 0:
        return None
    start = marker + 24
    if start + compressed > len(nested) or nested[start] & 0x0F != 0x08:
        return None
    try:
        inflater = zlib.decompressobj()
        data = inflater.decompress(nested[start : start + compressed], uncompressed + 1)
    except zlib.error:
        return None
    if not inflater.eof or len(data) != uncompressed:
        return None
    return data


def strip_prologue(code):
    if len(code) < 4 * PROLOGUE_DWORDS:
        return code, False
    words = struct.unpack_from("<%dI" % PROLOGUE_DWORDS, code, 0)
    prologue = all(
        words[index] >> 26 == PROLOGUE_ENCODING
        and (words[index] >> 16) & 0x3FF == PROLOGUE_OPCODE
        for index in (4, 7)
    )
    return (code[4 * PROLOGUE_DWORDS :], True) if prologue else (code, False)


def trim_at_endpgm(code):
    for offset in range(len(code) // 4 * 4 - 4, -1, -4):
        if struct.unpack_from("<I", code, offset)[0] == END_PGM:
            return code[: offset + 4]
    return code


def carve_one(name, offset, archive):
    inflater = zlib.decompressobj(-15)
    try:
        container = inflater.decompress(
            archive[offset : offset + READ_LIMIT], CHUNK_LIMIT
        )
    except zlib.error:
        return {"name": name, "error": "inflate"}
    if not inflater.eof or not container.startswith(b"DXBC"):
        return {"name": name, "error": "not-dxbc"}
    chunks = parse_dxbc(container)
    if chunks is None or b"DXIL" not in chunks:
        return {"name": name, "error": "no-dxil"}
    start, size = chunks[b"DXIL"]
    nested = container[start : start + size]
    if len(nested) < 4:
        return {"name": name, "error": "short-dxil"}
    kind = struct.unpack_from("<H", nested, 2)[0]
    blob = fssm_from(nested)
    if blob is None:
        return {"name": name, "kind": kind, "error": "no-xbch"}
    if blob[:4] != b"FSSM" or len(blob) < 0x1D0:
        return {"name": name, "kind": kind, "error": "short-fssm"}
    codeOffset, codeSize = struct.unpack_from("<QQ", blob, 0xC0)
    code = blob[codeOffset : codeOffset + codeSize]
    if len(code) != codeSize:
        return {"name": name, "kind": kind, "error": "short-code"}
    code, prologue = strip_prologue(code)
    wave32 = struct.unpack_from("<I", blob, WAVE_WORD)[0] >> WAVE32_BIT & 1
    return {
        "name": name,
        "kind": kind,
        "code": trim_at_endpgm(code),
        "prologue": prologue,
        "vin": vin_text(blob) if STAGES.get(kind) == "VS" else "",
        "wave": 32 if wave32 else 64,
        "error": None,
    }


def write_carve(archive, entries, bins, vins, manifest):
    print("names found:", len(entries), flush=True)
    rows = []
    errors = collections.Counter()
    index = 0
    for name, offset in entries:
        result = carve_one(name, offset, archive)
        if result["error"]:
            errors[result["error"]] += 1
            rows.append(
                {
                    "idx": "",
                    "name": name,
                    "kind": result.get("kind", ""),
                    "stage": "",
                    "bytes": "",
                    "prologue": "",
                    "wave": "",
                    "error": result["error"],
                }
            )
            continue
        index += 1
        with open(os.path.join(bins, "%d.bin" % index), "wb") as blob:
            blob.write(result["code"])
        if result["vin"]:
            with open(os.path.join(vins, "%d.vin" % index), "w", newline="\n") as table:
                table.write(result["vin"])
        rows.append(
            {
                "idx": index,
                "name": name,
                "kind": result["kind"],
                "stage": STAGES.get(result["kind"], "?"),
                "bytes": len(result["code"]),
                "prologue": int(result["prologue"]),
                "wave": result["wave"],
                "error": "",
            }
        )

    with open(manifest, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("archive", help="path to shaders.rpf")
    parser.add_argument(
        "--out", required=True, help="directory for bins/ and manifest.csv"
    )
    parser.add_argument(
        "--force", action="store_true", help="replace blobs this tool wrote earlier"
    )
    arguments = parser.parse_args()

    bins = os.path.join(arguments.out, "bins")
    vins = os.path.join(arguments.out, "vin")
    manifest = os.path.join(arguments.out, "manifest.csv")
    managed = [
        os.path.join(directory, name)
        for directory, pattern in ((bins, MANAGED), (vins, MANAGED_VIN))
        if os.path.isdir(directory)
        for name in sorted(os.listdir(directory))
        if pattern.fullmatch(name)
    ]
    if not arguments.force and (managed or os.path.lexists(manifest)):
        parser.error(
            "%s already holds carved output; pass --force to replace it"
            % arguments.out
        )
    for path in managed:
        if os.path.isdir(path) and not os.path.islink(path):
            parser.error("carved output path is a directory: %s" % path)
    if os.path.isdir(manifest):
        parser.error("manifest path is a directory: %s" % manifest)

    with open(arguments.archive, "rb") as handle, mmap.mmap(
        handle.fileno(), 0, access=mmap.ACCESS_READ
    ) as archive:
        archive_stat = os.fstat(handle.fileno())
        for path in (*managed, manifest):
            if os.path.exists(path) and os.path.samestat(archive_stat, os.stat(path)):
                parser.error("archive is also a carved output: %s" % path)
        entries = find_entries(archive)
        os.makedirs(bins, exist_ok=True)
        os.makedirs(vins, exist_ok=True)
        with (
            tempfile.TemporaryDirectory(dir=bins) as staged_bins,
            tempfile.TemporaryDirectory(dir=vins) as staged_vins,
            tempfile.TemporaryDirectory(dir=arguments.out) as staged_out,
        ):
            staged_manifest = os.path.join(staged_out, "manifest.csv")
            rows, errors = write_carve(archive, entries, staged_bins, staged_vins, staged_manifest)
            replaced = set()
            for staged, destination in ((staged_bins, bins), (staged_vins, vins)):
                for name in os.listdir(staged):
                    path = os.path.join(destination, name)
                    os.replace(os.path.join(staged, name), path)
                    replaced.add(path)
            for path in managed:
                if path not in replaced:
                    os.remove(path)
            os.replace(staged_manifest, manifest)

    carved = [row for row in rows if not row["error"]]
    sizes = sorted(row["bytes"] for row in carved)
    print("carved:", len(carved), "errors:", dict(errors))
    print("stages:", dict(collections.Counter(row["stage"] for row in carved)))
    print("prologue:", dict(collections.Counter(row["prologue"] for row in carved)))
    print("wave:", dict(collections.Counter(row["wave"] for row in carved)))
    if sizes:
        print(
            "bytes: min %d median %d max %d"
            % (sizes[0], sizes[len(sizes) // 2], sizes[-1])
        )


if __name__ == "__main__":
    main()
