# tools/xbox

Experimental tooling for running Xbox Series shader code through the shader recompiler. See `docs/xbox/XBOX_SHADERS.md`.

1. `build_msvc.py` builds the recompiler and the tools below with MSVC (run it from PowerShell/cmd). On Linux,
   `cmake -S tools/xbox -B build-xbox -DCMAKE_BUILD_TYPE=Release && cmake --build build-xbox` builds `driver`, `stages` and `exec`
   (needs the `SPIRV-Headers` and `Vulkan-Headers` submodules; `exec` opens the Vulkan loader at run time, as the AGC driver does).
   When a device address does not fit in the 48 bits of a guest pointer (RADV hands out such addresses), `exec` gives the regions remapped
   guest addresses as `lowguest` does. Without `rdna2.py`, `rdna_asm.py` takes its opcode
   tables from LLVM (`rdna2_llvm.py`, needs `llvm-mc` with the AMDGPU target) or, without such an `llvm-mc` (the official Windows LLVM
   builds have no AMDGPU target), from the recompiler's own decoder (`rdna2_recomp.py`, which needs nothing outside the repository;
   `XBOX_RDNA2=llvm|recomp` picks one), so `exec_tests.py` and
   `exec_bvh_tests.py` run with no game files: `XBOX_EXEC=build-xbox/exec python3 tools/xbox/exec_tests.py`. `VK_DRIVER_FILES=<icd.json>`
   picks the GPU. `rdna2_recomp.py` has rdna2.py's whole interface (`decode`, `sweep` with instruction words and formats), so `cpu_ref.py`
   falls back to it as well; its instruction lengths and opcodes follow the recompiler's decoder.
   The same commands build on macOS against MoltenVK (Homebrew `molten-vk`, `vulkan-loader`); `exec` enables portability enumeration
   and `VK_KHR_portability_subset` when they are offered.

   On macOS, `exec` searches for the Vulkan loader using configured runpaths and the standard Homebrew
   library directories (`/opt/homebrew/lib` on Apple Silicon and `/usr/local/lib` on Intel Macs).
   For custom Vulkan SDK installations, set `DYLD_LIBRARY_PATH` to the SDK's absolute library directory
   (where dyld permits it).

   Verify the installation from the repository root:

   ```sh
   python3 tools/xbox/exec_smoke.py --exec build-xbox/exec
   ```

   A device that cannot pin the compute subgroup size (MoltenVK) gets SPIR-V 1.5
   without a required size, so the subgroup size is the device's fixed `subgroupSize`; wave64 then runs as two 32-wide halves as on
   other 32-wide devices. MoltenVK reports `fragmentShaderBarycentric` but MSL has no `PerVertexKHR`: run pixel tests with
   `XBOX_NO_FRAGMENT_BARY=1`.
2. `extract_shaders.py` writes prologue-stripped, trimmed code blobs `<idx>.bin` (it depends on the RPF9
   extraction pipeline and `rdna2.py` decoder kept next to the dumped shaders; adjust its paths).
   `carve_shaders.py` does the same from the archive alone: `python3 tools/xbox/carve_shaders.py <game>/shaders.rpf
   --out <dir>` writes `<dir>/bins/<idx>.bin` and `<dir>/manifest.csv`, with no table of contents and no decoder.
   Whole-corpus run on that output: `XBOX_RDNA2=recomp python -I tools/xbox/bindings.py <dir>/bins <dir>/binds`, then from a run
   directory `python -I tools/xbox/run_full.py out.json --no-canary` with `XBOX_CORPUS=<dir>`, `XBOX_SPIRV_VAL=<spirv-val>` and
   `XBOX_DRIVER_CS` / `XBOX_DRIVER_GFX` set to the `driver.exe` that `build_msvc.py` builds (`XBOX_BUILD` holds the result cache).
   The carver numbers shaders in archive order, which is not the numbering of `docs/xbox/results.json` and `canary.json`, so only
   per-stage totals compare, and the canaries are refused on a carved corpus. Each shader runs at the wave size in the manifest's
   `wave` column (the carver reads it from the FSSM header: word `0xd4`, bit 5 set = wave32), or, for a manifest without one, the
   size its exec-mask instructions imply (64 when they are all 64-bit); `XBOX_WAVE=32` restores the earlier fixed size.
3. `driver.exe <code.bin> <wave 32|64> <out.spv> [user_data_dwords]` recompiles one compute shader with stubbed
   bindings (user-data pointer pairs backed by fake buffer-descriptor memory).
   For `vs`, argument 7 is an optional vertex input table (`<idx>.vin`, written by `carve_shaders.py`, lines
   `vin <first vgpr> <vgpr count> <input register>`): the VGPRs the runtime-built fetch subroutine fills. Each record
   occupies one line. A malformed record stops compilation with its line number. With the table the
   shader's opening `s_swappc_b64` is replaced by host vertex attribute reads (see "Vertex fetch subroutine" in
   `docs/xbox/XBOX_SHADERS.md`). The embedded-fetch analyzer is pointed at s96, outside the 16 user words, because Xbox
   shaders load no attribute tables themselves. Argument 8 (with `-` for argument 7 when there is no table) is the ESGS
   item size in dwords of a VS that is the export half of a legacy GS pipeline: it is compiled as a merged program (user
   data from s8 to s39) whose LDS item writes become the outputs the geometry stage reads (see "Vertex export halves" in
   `docs/xbox/XBOX_SHADERS.md`).
4. `run_batch.py` runs the driver over a directory of blobs and validates with `spirv-val`.
5. `stages.exe <code.bin>` times decode / CFG build / structurize separately (for the slow ray-tracing shaders).
6. `exec.exe <scenario.txt>` is the execution harness: it recompiles a compute shader against REAL Vulkan device memory, dispatches it on the
   host GPU and writes the memory regions (and images) back. Build it on Windows with `build_msvc.py --with-exec`.
   `rdna_asm.py` assembles small RDNA2 test programs (opcode numbers come from the decoder tables; each program is decoded back to check it),
   `exec_tests.py`, `exec_bvh_tests.py` and `image_tests.py` hold the tests. Set `XBOX_EXEC` to the `exec.exe` path. Supported: guest
   buffers, shader data, flattened SRT, identity BDA page table and fault reporting, single-level linear 1D, 2D, 2D array, 3D and cube (array) images in
   any format the AGC driver maps and the device can sample, T#s encoded with the AGC driver's `EncodeTextureResource`, samplers built
   from the guest S# with its `DecodeSamplerResource`. Pixel shaders (`stage ps`, see the pixel smoke, interpolation and per-vertex
   tests): one triangle over up to eight RGBA32F targets; per-vertex inputs need `VK_KHR_fragment_shader_barycentric`. Scenario lines for images: `image <id> <w> <h> [format <IMG_FORMAT>] [dim 1d|2d|3d|cube] [depth <d>] [layers <n>] [mips <n>]
   [clear r g b a] [init <file>]` (default format 77, 32_32_32_32_FLOAT; init files hold raw texels in the image format, row-major, slice
   after slice, level after level; clear values are integers for UINT/SINT formats; read back as `out/image<id>.bin` in the image format) and
   `timg <region> <offset> <image>`, which writes the image's linear T#. Image tables (a T# loaded with a run-time index) bind like the other images. `exec` compiles five AGC driver sources and a no-op run-report stand-in (`stub_run_report.cpp`) for this
   (`CMakeLists.txt`, `build_msvc.py --with-exec`). A cube image has six layers per cube (`layers` sets a multiple of six for a cube array). Mip levels are sampled and loaded; storage access to a mipmapped image stops the run. Block-compressed (BC) images need an init file of raw blocks. Not yet: storage mip levels, texel buffers,
   other graphics stages.
   Compute shaders get the device's subgroup size for their wave size; a wave64 shader on a device whose largest compute subgroup is 32
   (NVIDIA, Intel) is recompiled for 32-wide subgroups, two guest lanes per invocation. `XBOX_HOST_SUBGROUP=<n>` forces the host size.

## Pixel smoke test

With the Vulkan SDK's `glslangValidator` on PATH and the execution harness built,
run this from the repository root:

```powershell
python tools/xbox/pixel_smoke.py --exec "build-xbox/exec.exe"
```

The vertex shader draws a full-screen triangle. The standalone RDNA2 pixel shader
writes the pixel-center X and Y coordinates, zero blue, and unit alpha to a
13 by 7 RGBA32F attachment. The test checks every channel at every coordinate
and reports the first mismatch. It uses no external shader corpus. The scenario's `export 0 9 0xe4` selects
the 32-bit ABGR export mode with the identity component mapping.

## Vertex color interpolation test

With a built execution harness and the Vulkan SDK's `glslangValidator` on PATH:

```powershell
python tools/xbox/interpolation_test.py --exec "build-xbox/exec.exe"
```

A GLSL vertex shader assigns red, green and blue to a triangle. Its vertices
have clip-space W values of 1, 2 and 4. The RDNA2 fragment program interpolates
each channel using VINTRP P1/P2 and exports RGBA32F. At every pixel center, the
CPU derives screen-space weights from the projected triangle, divides each by
the corresponding W, then normalizes their sum to one. Covered pixels must match
within 0.00001 per channel. Outside pixels must exactly retain the clear value.
The triangle edges avoid pixel centers, so the reference needs no edge tie-break.

## Per-vertex input test

With a built execution harness and the Vulkan SDK's `glslangValidator` on PATH:

```powershell
python tools/xbox/per_vertex_test.py --exec "build-xbox/exec.exe"
```

The fragment program reads the P0, P10 and P20 interpolation parameters of two
attributes with `VINTRP MOV` and writes them to six RGBA32F targets. P10 and P20
are the plane coefficients P1 - P0 and P2 - P0. Every pixel must read the
following values. This test requires native per-vertex inputs through
`VK_KHR_fragment_shader_barycentric`.

| Attribute | P0 | P10 | P20 |
| --- | --- | --- | --- |
| 0 | (1, 0, 0, 1) | (-1, 1, 0, 0) | (-1, 0, 1, 0) |
| 1 | (0.25, 0.5, 0.75, 1) | (0.25, 0.25, 0.25, 0.25) | (0.75, 1, 1.25, 1.5) |

With `XBOX_NO_FRAGMENT_BARY=1`, the EXEC-7 guard rejects fragment shaders whose
compiled artifacts require a barycentric emulation companion stage that the execution
harness does not provide. Previously these shaders could reach the pipeline with unmatched
fragment inputs and produce undefined results. For `cpu_compare_ps.py` runs that reach the
guard with a rebuilt pixel harness, the script records `GPU_FAIL` with the
missing-companion-stage diagnostic. These cases produce no readback to compare.
CPU-side `SKIP` handling is unchanged.

The guard tests check both rejection and successful fragment compilation:

```sh
python3 -I tools/xbox/pipeline_capability_test.py --exec build-xbox/exec
```

The native compilation check is skipped if the selected device lacks
`VK_KHR_fragment_shader_barycentric` support. On a device with
`geometryShader=false`, add `--expect-no-geometry` to check early rejection of
`stage gs`.

## Quad swizzle test

With a built execution harness and the Vulkan SDK's `glslangValidator` on PATH:

```powershell
python tools/xbox/quad_test.py --exec "build-xbox/exec.exe"
```

The first fragment program exchanges each pixel with its horizontal and vertical
neighbour inside the 2x2 quad with `ds_swizzle_b32`, subtracts, and exports the
values and differences. The second narrows EXEC to one lane per quad, expands it
with `s_wqm_b32`, and exports a constant; a second run activates two quads only,
so the other quads must stay clear. Both run with `XBOX_NO_FRAGMENT_BARY=1`, the
host perspective path; the quad operations do not need the barycentric
extension.
The two-quad case assumes the 32-lane pixel wave is one host subgroup. `exec`
builds pixel shaders for the device's `subgroupSize` without requesting a size,
and the driver may run the fragment stage narrower: Intel ANV runs it 16 wide
(FRAGMENT is not in its `requiredSubgroupSizeStages`, so it cannot be pinned), so
EXEC applies per 16 lanes and twice the quads light. The test first reads each
lane's `v_mbcnt_lo_u32_b32` index and skips that case when the observed lane span
(highest index plus one) is below 32. The probe is a separate shader, so this is
a heuristic; a probe image that is not a contiguous run of lane indices counts as
a failure and both cases still run.

## Windows quick start

From the repository root in PowerShell, with Python 3, the x64 MSVC C++ toolchain,
the Vulkan SDK, and a Vulkan-capable GPU driver installed. Build the recompiler and
the execution harness:

```powershell
git submodule update --init 3rdparty/SPIRV-Headers
python tools/xbox/build_msvc.py --with-exec
```

The build reads `VULKAN_SDK`; `--vulkan-sdk "path to SDK"` selects another
installation, and `--build-dir` another output directory. It writes
`build-xbox/driver.exe`, `stages.exe` and `exec.exe`. Smoke-test it on the GPU; this
dispatches one 32-lane workgroup and checks all 1024 output bytes, with no shader
corpus or external decoder:

```powershell
python tools/xbox/exec_smoke.py --exec build-xbox/exec.exe
```

`exec_tests.py`, `image_tests.py`, `exec_bvh_tests.py` and `desc_chain_tests.py`
assemble their RDNA2 programs with `rdna_asm.py` and read the harness path from
`XBOX_EXEC`. Two points matter on Windows:

- The official Windows LLVM has no AMDGPU `llvm-mc`, so set `XBOX_RDNA2=recomp` to
  assemble from the in-repo decoder (`rdna2_recomp.py`), which needs nothing outside
  the repository.
- `XBOX_EXEC` is used from the directory you start the test in: the scenario files
  are written to a temporary directory, but `exec.exe` itself is not run there. An
  absolute path avoids depending on that directory.

```powershell
$env:XBOX_EXEC = "$PWD\build-xbox\exec.exe"
$env:XBOX_RDNA2 = "recomp"
python tools/xbox/exec_tests.py
python tools/xbox/image_tests.py
python tools/xbox/exec_bvh_tests.py
python tools/xbox/desc_chain_tests.py
```

The standalone tests (`mad_legacy_clamp_test.py`, `mad_legacy_fused_test.py`, `permlane_test.py`,
`quadmask_test.py`, `quad_test.py`) take `--exec "build-xbox/exec.exe"` and resolve
it themselves. The pixel tests (`texture_test.py`, `interpolation_test.py`,
`per_vertex_test.py`, `pixel_smoke.py`) also need the SDK's `glslangValidator` on
`PATH`:

```powershell
$env:PATH = "$env:VULKAN_SDK\Bin;$env:PATH"
python tools/xbox/texture_test.py --exec build-xbox/exec.exe
```

With more than one GPU, `exec.exe` runs on the last discrete device it enumerates;
pin one with `VK_DRIVER_FILES=<icd.json>` (the loader prints the installed ICD
manifest paths under `VK_LOADER_DEBUG=driver`). A device without
`VK_KHR_fragment_shader_barycentric` runs the pixel tests only with
`XBOX_NO_FRAGMENT_BARY=1`.

### Test v_mad_legacy_f32 clamp

With the execution harness built, run:

```powershell
python tools/xbox/mad_legacy_clamp_test.py --exec "build-xbox/exec.exe"
```

One 32-lane workgroup computes `2.0 * 4.0 + 4.0`, `-2.0 * 4.0 + 4.0` and
`0.5 * 0.5 + 0.0` with `v_mad_legacy_f32` and the VOP3 clamp bit. The clamped
results must be 1.0, 0.0 and 0.25 in every lane. Clearing the clamp bit produces
12.0, -4.0 and 0.25, so the test fails when clamp is ignored.

### Test fused v_mad_legacy_f32 and v_mac_legacy_f32

With the execution harness built, run:

```powershell
python tools/xbox/mad_legacy_fused_test.py --exec "build-xbox/exec.exe"
```

One 32-lane workgroup computes, with `v_mad_legacy_f32`, the two discriminator rows of
boykopovar/AnyPS5#1281, `(1 + 2^-12)^2 - 1` and `max * 2 - max`, its two
denormal-flush rows `+tiny * 1 + -0` and `-tiny * 1 + -0`, and the zero rule
`+0 * +inf + 1`; then the first row again with `v_mac_legacy_f32`. The recompiler runs
twice. By default the discriminator rows must equal `v_fma_f32` (`3A000400`,
`7F7FFFFF`); with `XBOX_NO_FUSED_LEGACY_MAD=1` the product is rounded first
(`3A000000`, `7F800000`). The flush rows give +0 and the zero-rule row 1.0 either way.
The operands are loaded from the output buffer, which the test initialises, not moved in as
literals: an `Fma` whose three operands are constants is folded by Intel's Windows driver
(101.6127) as a rounded multiply then an add, and the test would measure that folder
instead of the GPU.

### Test v_permlane16 and v_permlanex16

With the execution harness built, run:

```powershell
python tools/xbox/permlane_test.py --exec "build-xbox/exec.exe"
```

One 32-lane workgroup supplies a 64-bit lane-select value that reverses each
16-lane group. `v_permlane16_b32` selects within the group and
`v_permlanex16_b32` selects from the opposite group. Each instruction also runs
with `op_sel[3]` set, which Xbox shaders use; the result must match the run
without it. Every lane exports four dwords and all 128 values are checked.

### Test s_quadmask

With the execution harness built, run:

```powershell
python tools/xbox/quadmask_test.py --exec "build-xbox/exec.exe"
```

One 32-lane workgroup reduces four 32-bit masks and two 64-bit masks to one bit
per group of four input bits. The cases cover all ones, one bit per quad, zero,
a high bit, and a 64-bit mask whose set bits are only in the high word. Every
lane must store the expected eight words.

### Test geometry shader execution

With the execution harness built and the SDK's `glslangValidator` on PATH, run:

```powershell
python tools/xbox/gs_test.py --exec "build-xbox/exec.exe"
```

A GLSL vertex shader draws one triangle. The RDNA2 geometry shader writes
vertices to the GSVS LDS slots, emits them with `s_sendmsg`, and exports a
position and one colour parameter. A GLSL fragment shader writes that colour to
an 8x8 RGBA32F target. The first program emits a full-screen triangle, so every
pixel must be red. The second emits a left-half quad, so only the left four
columns must be red; the rest must keep the clear value. The scenarios use
`stage gs`, `vs`, `fs`, and `geom <1|2|3|4|5> 3 <vertices> 8 8 1 1`: point, line,
triangle, line-adjacency and triangle-adjacency host input. The draw supplies
the matching 1, 2, 3, 4 or 6 vertices, and the guest program ignores its inputs,
so the emitted geometry is the same.

### Test vertex shader execution

With the execution harness built and the SDK's `glslangValidator` on PATH, run:

```powershell
python tools/xbox/vs_test.py --exec "build-xbox/exec.exe"
```

The RDNA2 vertex shader reads the vertex index from `v5`, builds the triangle
(-1,-1), (4,-1), (-1,4), and exports that position and one parameter holding
(x, y, vertex index, 1). A GLSL fragment shader writes the interpolated parameter
to a 13x7 RGBA32F target. W is 1 at every vertex, so at each pixel centre the
parameter must equal the pixel's clip-space x and y, and its third channel
((x + 1) + 2 (y + 1)) / 5, within 0.00001. The scenario uses `stage vs` and `fs`;
the draw supplies three vertices and no vertex buffers, so a shader that calls a
fetch subroutine or reads the instance index is not covered.

### Test texture sampling

With the Vulkan execution harness built and the SDK's `glslangValidator` on PATH,
run `python tools/xbox/texture_test.py --exec "build-xbox/exec.exe"`.

A GLSL vertex shader supplies UVs and unequal clip-space W. An RDNA2 fragment
program interpolates U and V, samples a generated 4 by 3 RGBA32F texture at mip
level zero, and exports the result. The CPU calculates perspective-correct UVs
at each pixel center and independently selects the nearest texel with clamp-to-edge
addressing. All 12 texels must appear, all 192 covered pixels must match exactly,
and all 832 outside pixels must retain the clear value.

Use `python tools/xbox/texture_test.py --bilinear` to select linear minification
and magnification in the guest sampler. This uses the same triangle, mip level,
and texture as the nearest test. The CPU shifts UVs to texel centers, clamps the
four neighbors separately, and weights their RGBA values. The absolute channel
tolerance is 0.002 to allow for finite filter-weight precision. The background
must still match the clear value exactly.

To build the guest-sampler runner on Windows, use an x64 Visual Studio developer
shell with LLVM's clang-cl, CMake, Ninja, and the Vulkan SDK tools on PATH:

```powershell
git submodule update --init 3rdparty/SPIRV-Headers 3rdparty/Vulkan-Headers
cmake -S tools/xbox -B build-sampler -G Ninja -DCMAKE_CXX_COMPILER=clang-cl -DCMAKE_BUILD_TYPE=Release
cmake --build build-sampler --target exec
python tools/xbox/texture_test.py --bilinear --exec build-sampler/exec.exe
```
