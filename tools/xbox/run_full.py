"""Run every shader of the full corpus (manifest_full.csv, bins_full/, binds_full/) through the recompiler and spirv-val.

usage (from the build dir):  python -I run_full.py <out.json> [--skip results_a.json ...] [--workers N] [--limit SEC] [--only STAGE] [--idx-file F] [--no-canary] [--no-cache]

Before running it checks that every .bind was written by the current bindings.py (harness_guard.py) and that the canary shaders
still give their recorded outcomes (canary.py); either failing stops the run.
Results are cached by a hash of every input (run_full_cache.json in the build dir), and jobs run slowest-first using the run times
recorded in run_full_costs.json, so long compiles overlap with the short ones.

Same drivers and defaults as run_corpus_bind.py (compute, driver_heap.exe, fake-memory variants default/0/2/3 until one validates) and
run_graphics_final.py (PS/VS/MS, driver_gfx2.exe (XBOX_DRIVER_GFX overrides), PS both strict and with XBOX_ALLOW_UNMEASURED_F16_INTERP, barycentric extension
retry; PS interpolants read only at P0 are passed as flat, as the FSSM input table declares them; a VS with vin_full/<idx>.vin gets its fetch-subroutine VGPR layout, see carve_shaders.py). Shaders listed in the --skip files (earlier results, by idx) are not run again. Results are written every 500 shaders, so a
killed run can be resumed by passing its own output as --skip. Stage '?' in the manifest is mesh (type 6/7), GS has no driver mode and
is recorded as SKIP. The wave size passed to the driver is the manifest's wave column (carve_shaders.py: FSSM header word 0xd4, bit 5
set = wave32) when there is one, else 64 for a program that changes exec only with 64-bit SALU ops, else 32;
XBOX_WAVE=32 or 64 forces one size for every shader."""
import subprocess, csv, sys, collections, concurrent.futures as cf, json, os, time
sys.path.insert(0, os.environ.get("XBOX_RDNA2_DIR", r"E:/Programming/rpf9shaders/out_disasm"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # harness_guard, canary (python -I does not add it)
try:
    if os.environ.get("XBOX_RDNA2") == "recomp":
        raise ImportError
    import rdna2
except ImportError:
    import rdna2_recomp as rdna2

M = os.path.join(os.environ.get("XBOX_CORPUS", r"E:/Programming/rpf9shaders/out_disasm/anyps5"), "")
CARVED = not os.path.exists(M + "manifest_full.csv") and os.path.exists(M + "manifest.csv")
BINS = M + ("bins/" if CARVED else "bins_full/")
MANIFEST = M + ("manifest.csv" if CARVED else "manifest_full.csv")
VAL = os.environ.get("XBOX_SPIRV_VAL", r"C:/VulkanSDK/1.4.363.1/Bin/spirv-val.exe")
BUILD = os.path.join(os.environ.get("XBOX_BUILD", r"E:/Programming/anyps5_src/build"), "")
DRIVER_CS = os.environ.get("XBOX_DRIVER_CS", BUILD + "driver_heap.exe")
DRIVER_GFX = os.environ.get("XBOX_DRIVER_GFX", BUILD + "driver_gfx2.exe")    # PS/VS/MS driver; override to test a new build
PATH_VARS = ("XBOX_CORPUS", "XBOX_BUILD", "XBOX_SPIRV_VAL", "XBOX_DRIVER_CS", "XBOX_DRIVER_GFX", "XBOX_RDNA2_DIR")
LIMIT = 120                      # seconds per driver run
BINDS = M + ("binds/" if CARVED else "binds_full/")
VIN = M + ("vin/" if CARVED else "vin_full/")
GFX = {"PS": "ps", "VS": "vs", "?": "ms", "MS": "ms", "GS": "gs"}


def interpolants(idx):
    """(interpolant count, fp16 mask, flat mask) of a pixel shader, from its code. Flat = read only with v_interp_mov_f32 P0 (VINTRP
    OP 2, VSRC 2): those are exactly the interpolants whose FSSM input table entry has flags bit 0 (FLAT_SHADE), see fssm_header.py."""
    d = open(BINS + f"{idx}.bin", "rb").read()
    ins, _ = rdna2.sweep(d, 0, len(d))
    a, f, p0, other = [], [], set(), set()
    for i in ins:
        if i.fmt.startswith("VINTRP"):
            w = i.words[0]
            x = (w >> 10) & 0x3F
            a.append(x)
            (p0 if ((w >> 16) & 3) == 2 and (w & 0xFF) == 2 else other).add(x)
        elif i.name in ("v_interp_p1ll_f16", "v_interp_p1lv_f16", "v_interp_p2_f16"):
            x = i.words[1] & 0x3F
            a.append(x)
            f.append(x)
            other.add(x)
    return (max(a) + 1 if a else 0, sum(1 << x for x in set(f)), sum(1 << x for x in p0 - other))


def attrs(idx):
    return interpolants(idx)[:2]


def esgs_item(idx):
    """ESGS item size in dwords of a VS that is the export half of a legacy GS pipeline, else 0. Those end with s_barrier and an
    s_setpc_b64 into the GS half, and write their item to LDS at lane * item bytes (v_mad_u32_u24 v, lane, <bytes>, s)."""
    d = open(BINS + f"{idx}.bin", "rb").read()
    ins, _ = rdna2.sweep(d, 0, len(d))
    names = [i.name for i in ins if i.name != "s_endpgm"]
    if names[-2:] != ["s_barrier", "s_setpc_b64"]:
        return 0
    sizes = {((i.words[1] >> 9) & 0x1FF) - 128 for i in ins if i.name == "v_mad_u32_u24" and 129 <= ((i.words[1] >> 9) & 0x1FF) <= 192}
    return sizes.pop() // 4 if len(sizes) == 1 and min(sizes) % 4 == 0 else 0


_MANIFEST_WAVE = None


def manifest_wave(idx):
    global _MANIFEST_WAVE
    if _MANIFEST_WAVE is None:
        _MANIFEST_WAVE = {r["idx"]: r["wave"] for r in csv.DictReader(open(MANIFEST)) if r.get("wave")}
    return _MANIFEST_WAVE.get(idx)


def wave(idx):
    forced = os.environ.get("XBOX_WAVE", "auto")
    if forced != "auto":
        return forced
    header = manifest_wave(idx)
    if header:
        return header
    d = open(BINS + f"{idx}.bin", "rb").read()
    ins, _ = rdna2.sweep(d, 0, len(d))
    narrow = wide = False
    for i in ins:
        if i.fmt not in ("SOP1", "SOP2"):
            continue
        w = i.words[0]
        operands = [(w >> 16) & 0x7F, w & 0xFF] + ([(w >> 8) & 0xFF] if i.fmt == "SOP2" else [])
        if "saveexec" not in i.name and 126 not in operands:
            continue
        wide = wide or i.name.endswith("_b64")
        narrow = narrow or i.name.endswith("_b32")
    return "64" if wide and not narrow else "32"


def classify(cmd, spv, env):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=LIMIT, env=env)
        o = (p.stdout.strip().splitlines() or [""])[-1]
        rc = p.returncode
    except subprocess.TimeoutExpired:
        return ("TIMEOUT", "")
    if o.startswith("OK"):
        v = subprocess.run([VAL, "--target-env", "vulkan1.3", spv], capture_output=True, text=True)
        return ("VALID", "") if v.returncode == 0 else ("INVALID", (v.stderr.strip().splitlines() or [""])[0][:150])
    if o.startswith("THROW"):
        return ("THROW", o[6:].split(" RecompileRequest:")[0][:250])
    return ("CRASH", o[:150] or f"rc={rc}")


def compute(idx):
    os.makedirs("spv_full", exist_ok=True)
    for v in (None, 0, 2, 3):          # None = the driver's default (zeros), 0 = unbacked, 2 = tiled T# records, 3 = tiled V# records
        spv = f"spv_full/{idx}.{v}.spv"
        cmd = [DRIVER_CS, BINS + f"{idx}.bin", wave(idx), spv, BINDS + f"{idx}.bind"] + ([str(v)] if v is not None else [])
        s, d = classify(cmd, spv, dict(os.environ))
        if os.path.exists(spv):
            os.remove(spv)
        if s == "VALID":
            return s, d, v
    return s, d, v


def graphics(idx, st, relaxed):
    env = dict(os.environ)
    if relaxed:
        env["XBOX_ALLOW_UNMEASURED_F16_INTERP"] = "1"

    def go(bary):
        os.makedirs("spv_full", exist_ok=True)
        spv = f"spv_full/{idx}.{int(relaxed)}{int(bary)}.spv"
        e = dict(env)
        if bary:
            e["XBOX_FRAGMENT_BARY"] = "1"
        cmd = [DRIVER_GFX, BINS + f"{idx}.bin", wave(idx), spv, BINDS + f"{idx}.bind", "1", st]
        item = esgs_item(idx) if st == "vs" else 0
        if st == "vs" and os.path.exists(VIN + f"{idx}.vin"):
            cmd.append(VIN + f"{idx}.vin")
        elif item:
            cmd.append("-")
        if item:
            cmd.append(str(item))
        if st == "ps":
            n, mask, flat = interpolants(idx)
            cmd += ["2", str(n), "%x" % mask, "%x" % flat]
        r = classify(cmd, spv, e)
        if os.path.exists(spv):
            os.remove(spv)
        return r
    s, d = go(False)
    if s == "THROW" and "emulated barycentric" in d:
        s2, d2 = go(True)
        if s2 == "VALID":
            return "VALID_BARY_EXT", d2
    return s, d


_HASH = {}


def _file_hash(path):
    """sha256 of a file, memoised (driver executables, spirv-val, this script)"""
    if path not in _HASH:
        import hashlib
        _HASH[path] = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
    return _HASH[path]


_LOGIC = None


def _logic_hash():
    """hash of the functions that decide a result (editing main(), messages or the cache code does not invalidate the cache)"""
    global _LOGIC
    if _LOGIC is None:
        import hashlib, inspect
        _LOGIC = hashlib.sha256("".join(inspect.getsource(f) for f in (interpolants, attrs, esgs_item, manifest_wave, wave, classify, compute, graphics, _one)).encode()).hexdigest()[:16]
    return _LOGIC


def cache_key(r, relaxed):
    """Everything that can change a result: the result logic in this script, the driver and spirv-val builds, the shader code, its
    bindings, stage, manifest wave size, f16 opt-in and XBOX_* switches in the environment. Same key = same inputs.
    The time limit is not in the key: a cached TIMEOUT is re-run when the limit is now higher (cache_usable)."""
    import hashlib
    idx, stage = r["idx"], r["stage"]
    h = hashlib.sha256()
    drv = DRIVER_CS if stage == "CS" else DRIVER_GFX
    for part in (_logic_hash(), _file_hash(drv), _file_hash(VAL), stage, r.get("wave") or "", str(relaxed),
                 repr(sorted((k, v) for k, v in os.environ.items() if k.startswith("XBOX_") and k not in PATH_VARS))):
        h.update(part.encode() + b"\0")
    h.update(open(BINS + f"{idx}.bin", "rb").read())
    h.update(b"\0" + open(BINDS + f"{idx}.bind", "rb").read())
    if os.path.exists(VIN + f"{idx}.vin"):
        h.update(b"\0vin\0" + open(VIN + f"{idx}.vin", "rb").read())
    return h.hexdigest()


def one(job):
    t = time.time()
    res = _one(job)
    res["secs"] = round(time.time() - t, 2)
    return res


def _one(job):
    r, relaxed = job
    idx, stage = r["idx"], r["stage"]
    res = dict(idx=idx, stage=stage, run=r.get("run", ""), relaxed=relaxed)
    try:
        if stage == "CS":
            res["status"], res["detail"], res["variant"] = compute(idx)
        elif stage in GFX:
            res["status"], res["detail"] = graphics(idx, GFX[stage], relaxed)
        else:
            res["status"], res["detail"] = "SKIP", "no driver mode for " + stage
    except Exception as e:
        res["status"], res["detail"] = "HARNESS", repr(e)[:150]
    return res


def main():
    global LIMIT
    args = sys.argv[1:]
    OUT = args[0]
    skip_files = [args[i + 1] for i, a in enumerate(args) if a == "--skip"]
    WORKERS = int(args[args.index("--workers") + 1]) if "--workers" in args else 8
    LIMIT = int(args[args.index("--limit") + 1]) if "--limit" in args else 120
    ONLY = args[args.index("--only") + 1] if "--only" in args else None
    import harness_guard
    harness_guard.require_fresh_binds(BINDS)                 # refuse bindings made by another bindings.py
    if "--no-canary" not in args:
        if CARVED:
            sys.exit("canary.json is numbered like manifest_full.csv, not like carve_shaders.py output: pass --no-canary")
        import canary
        if not canary.recompile_ok(WORKERS):                 # known-good shaders must still pass before a corpus run
            sys.exit("canary failed: not starting the corpus run (see above; --no-canary overrides)")
    skip = set()
    for f in skip_files:
        for e in json.load(open(f)):
            skip.add((e["idx"], bool(e.get("relaxed", False))))
    IDX = set(open(args[args.index("--idx-file") + 1]).read().split()) if "--idx-file" in args else None
    rows = [r for r in csv.DictReader(open(MANIFEST)) if (ONLY is None or r["stage"] == ONLY) and (IDX is None or r["idx"] in IDX)]
    jobs = [(r, x) for r in rows for x in ((False, True) if r["stage"] == "PS" else (False,)) if (r["idx"], x) not in skip]
    # result cache (build dir): a job whose inputs hash the same as in an earlier run reuses that result
    use_cache = "--no-cache" not in args
    cache_path, cost_path = BUILD + "run_full_cache.json", BUILD + "run_full_costs.json"
    cache = json.load(open(cache_path)) if use_cache and os.path.exists(cache_path) else {}
    costs = json.load(open(cost_path)) if os.path.exists(cost_path) else {}
    out, todo, keys = [], [], {}
    for job in jobs:
        k = cache_key(*job) if use_cache else None
        if k and k in cache and (cache[k]["status"] != "TIMEOUT" or cache[k].get("limit", 0) >= LIMIT):   # re-run timeouts at a higher limit
            out.append(dict(cache[k], idx=job[0]["idx"], stage=job[0]["stage"], run=job[0].get("run", ""), relaxed=job[1], cached=True))
        else:
            keys[id(job)] = k
            todo.append(job)
    # slowest first (known run times, else code size). This does not shorten a big run, which is bound by total CPU time (in a full mesh
    # run 30 timeouts were 44% of it); it keeps a few long jobs from trailing at the end of a small one.
    size = lambda r: os.path.getsize(BINS + f"{r['idx']}.bin")
    todo.sort(key=lambda j: (-costs.get(f"{j[0]['idx']}|{int(j[1])}", -1), -size(j[0])))
    print(len(jobs), "jobs,", len(jobs) - len(todo), "from the cache,", len(todo), "to run", flush=True)
    t0 = time.time()
    with cf.ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(one, j): j for j in todo}
        for n, f in enumerate(cf.as_completed(futs)):
            j = futs[f]
            res = f.result()
            out.append(res)
            costs[f"{j[0]['idx']}|{int(j[1])}"] = res["secs"]
            k = keys.get(id(j))
            if k and res["status"] not in ("HARNESS", "CRASH"):
                cache[k] = dict({x: res[x] for x in ("status", "detail", "variant", "secs") if x in res}, limit=LIMIT)
            if (n + 1) % 500 == 0:
                json.dump(out, open(OUT, "w"))
                print(n + 1, "of", len(todo), "run,", int(time.time() - t0), "s", flush=True)
    json.dump(costs, open(cost_path, "w"))
    if use_cache:
        json.dump(cache, open(cache_path, "w"))
    print("run time", int(time.time() - t0), "s", flush=True)
    json.dump(out, open(OUT, "w"))
    for st in sorted({e["stage"] for e in out}):
        for x in (False, True):
            c = collections.Counter(e["status"] for e in out if e["stage"] == st and e["relaxed"] == x)
            if c:
                print(st, "relaxed" if x else "strict", dict(c))


if __name__ == "__main__":
    main()
