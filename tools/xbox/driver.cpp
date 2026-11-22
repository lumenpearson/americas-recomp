#include "Recompiler.hpp"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <cstring>
#include <map>
#include <string>
#include <vector>
using namespace ShaderRecompiler;
// usage: driver <bin> <wave> <out.spv> [bindings.bind]
int main(int argc, char** argv) {
    if (argc < 4) return 2;
    std::ifstream f(argv[1], std::ios::binary);
    std::vector<char> raw((std::istreambuf_iterator<char>(f)), {});
    std::vector<std::uint32_t> code(raw.size() / 4);
    std::memcpy(code.data(), raw.data(), code.size() * 4);
    const std::uint32_t wave = std::stoul(argv[2]);
    // argv[5]: fake buffer contents. 0 = unbacked, 1 = zeros, 2 = tiled 2D image T# records, 3 = tiled buffer V# records (default 1)
    const int variant = argc > 5 ? std::atoi(argv[5]) : 1;
    RecompileRequest request{};
    // User data words = SGPRs s0.. (userDataBaseRegister 0). Mesh (merged NGG) shaders: s0-s7 are system/hidden words and user data runs
    // from s8 up to s39 (32 user SGPRs), so 40 words; compute and the other stages keep 16 (for compute, workgroup ids follow the user data).
    const std::string stageArg = argc > 6 ? argv[6] : "cs";
    const std::uint32_t esgsItem = stageArg == "vs" && argc > 8 ? std::stoul(argv[8]) : 0u;
    const std::uint32_t udBase = esgsItem != 0u ? 8u : 0u;
    const std::size_t nud = (stageArg == "ms" || stageArg == "gs") ? 40 : esgsItem != 0u ? 32 : 16;
    std::vector<std::uint32_t> userData(nud);
    struct Node { std::uint32_t id = 0; std::int64_t parent = -1; std::uint32_t off = 0; std::vector<std::uint32_t> mem = std::vector<std::uint32_t>(0x10000); std::uint32_t base = 0; };
    std::map<std::uint32_t, Node> nodes;
    std::vector<std::pair<std::uint64_t, std::vector<std::uint32_t>>> bufferBacking;  // zeroed memory behind each fake V# base
    std::map<std::pair<std::uint32_t, std::uint32_t>, int> descs;  // (node,offset) -> class, higher wins: img 3 > buf 2 > samp 1
    std::map<std::pair<std::uint32_t, std::uint32_t>, bool> bufImgCollisions;
    std::uint64_t counter = 0;
    if (argc > 4) {
        std::ifstream bf(argv[4]);
        std::string kind;
        while (bf >> kind) {
            if (kind == "node") {
                std::uint32_t id; std::string how; bf >> id >> how;
                Node& n = nodes[id];
                n.id = id; n.base = 0x10000000u + 0x100000u * id;
                if (how == "user") { std::uint32_t k; bf >> k; if (k >= udBase && k - udBase < nud) userData[k - udBase] = n.base; }
                else { std::int64_t p; std::uint32_t off; bf >> p >> off; n.parent = p; n.off = off; }
            } else if (kind == "desc") {
                std::uint32_t id, off; std::string cls; bf >> id >> off >> cls;
                const int rank = cls.rfind("img",0)==0 ? 3 : cls == "buf" ? 2 : 1;
                auto& slot = descs[{id, off}];
                const int prevRank = slot & 0xff;
                if ((prevRank == 3 && rank == 2) || (prevRank == 2 && rank == 3)) bufImgCollisions[{id, off}] = true;
                const int atomicBit = (rank == 3 && cls.back() == 'a') ? 0x1000 : (rank == 3 && cls.back() == 'b') ? 0x3000 : 0;  // class "img<dim>a": used by an atomic image op
                if (rank > prevRank) slot = rank | (rank == 3 ? (cls[3]-'0') << 8 : 0) | atomicBit;
                else slot |= atomicBit;
            }
        }
        for (auto& [id, n] : nodes) {
            if (n.parent >= 0 && nodes.count(static_cast<std::uint32_t>(n.parent)) && n.off / 4 + 1 < nodes[static_cast<std::uint32_t>(n.parent)].mem.size()) {
                nodes[static_cast<std::uint32_t>(n.parent)].mem[n.off / 4] = n.base;
                nodes[static_cast<std::uint32_t>(n.parent)].mem[n.off / 4 + 1] = 0;
            }
        }
        for (auto& [key, rank] : descs) {
            
            auto it = nodes.find(key.first);
            if (it == nodes.end()) continue;
            auto& m = it->second.mem;
            const std::size_t w = key.second / 4;
            const bool atomicImage = (rank & 0x1000) != 0; const bool atomic64 = (rank & 0x2000) != 0; const int dim = (rank >> 8) & 0xf; rank &= 0xff;
            if (rank == 3 && w + 8 <= m.size()) {
                const std::uint64_t base = 0x40000000ull + 0x10000ull * counter++;
                const std::uint32_t d[8] = {static_cast<std::uint32_t>(base >> 8), static_cast<std::uint32_t>((base >> 40) & 0xffu) | ((atomic64 ? 62u : atomicImage ? 20u : 56u) << 20) | (3u << 30), 15u | (15u << 14), 0xfacu | ((8u + dim) << 28), 0, 0, 0, 0};
                std::copy(d, d + 8, m.begin() + w);
            } else if (rank == 2 && w + 4 <= m.size()) {
                const std::uint64_t base = 0x50000000ull + 0x10000ull * counter++;
                const std::uint32_t d[4] = {static_cast<std::uint32_t>(base), static_cast<std::uint32_t>((base >> 32) & 0xffffu) | (16u << 16), 0x10000u, 0x30027facu};
                std::copy(d, d + 4, m.begin() + w);
                if (variant != 0) {
                    std::vector<std::uint32_t> mem(0x4000);
                    if (variant == 2) {  // tile with 32-byte 2D image T# records (format 56, as above)
                        const std::uint64_t ib = 0x40000000ull;
                        const std::uint32_t t[8] = {static_cast<std::uint32_t>(ib >> 8), static_cast<std::uint32_t>((ib >> 40) & 0xffu) | (56u << 20) | (3u << 30), 15u | (15u << 14), 0xfacu | (9u << 28), 0, 0, 0, 0};
                        for (std::size_t k = 0; k + 8 <= mem.size(); k += 8) std::copy(t, t + 8, mem.begin() + k);
                    } else if (variant == 3) {  // tile with 16-byte buffer V# records pointing at the first fake buffer
                        const std::uint32_t v[4] = {0x50000000u, 16u << 16, 0x10000u, 0x30027facu};
                        for (std::size_t k = 0; k + 4 <= mem.size(); k += 4) std::copy(v, v + 4, mem.begin() + k);
                    }
                    bufferBacking.push_back({base, std::move(mem)});
                }
            }
        }
    } else {
        for (std::size_t k = 0; k + 1 < nud; k += 2) {
            userData[k] = 0x10000000u + 0x100000u * static_cast<std::uint32_t>(k / 2);
            Node& n = nodes[static_cast<std::uint32_t>(k / 2)];
            n.id = static_cast<std::uint32_t>(k / 2);
            n.base = userData[k];
        }
    }
    std::vector<MemoryRegion> regions;
    for (auto& [id, n] : nodes) regions.push_back({n.base, std::as_bytes(std::span<const std::uint32_t>(n.mem))});
    for (auto& [base, mem] : bufferBacking) regions.push_back({base, std::as_bytes(std::span<const std::uint32_t>(mem))});
    request.context.memory = MemoryRegionView(std::span<const MemoryRegion>(regions));
    const std::vector<std::uint32_t> caps{1u,55u,56u,32u,33u,5283u,9u,10u,11u,22u,39u,61u,62u,63u,64u,65u,66u,67u,68u,4427u,4433u,4434u,4435u,4436u,4437u,4438u,4445u,4447u,4448u,4449u,4450u,5345u,5346u,5347u,5301u,5302u,5303u,5304u,5305u,6u,38u,41u,42u,5u,17u,18u,19u,3u,2u,4u,7u,8u,45u,5249u};
    const std::array<std::string_view, 7> ext{"SPV_EXT_mesh_shader", "SPV_KHR_storage_buffer_storage_class", "SPV_KHR_physical_storage_buffer", "SPV_KHR_16bit_storage", "SPV_KHR_8bit_storage", "SPV_KHR_vulkan_memory_model", "SPV_KHR_shader_ballot"};
    // argv[6] stage: cs (default), ps, vs. argv[7] pixel input address mask (hex), argv[8] interpolator count.
    // Everything not derivable from the code (export mapping/mode, target merges, vertex fetch) is a documented default.
    const std::string stage = argc > 6 ? argv[6] : "cs";
    if (stage == "ps") {
        ShaderPixelStageInfo px{};
        px.interpolatorCount = argc > 8 ? std::stoul(argv[8]) : 0;
        px.wave32 = wave == 32;
        px.inputAddr = argc > 7 ? std::stoul(argv[7], nullptr, 16) : 0x2u;
        px.hasPerspectiveCenterVgpr = (px.inputAddr & 0x2u) != 0;
        px.posX = (px.inputAddr & PixelInputBit(PixelInput::PositionX)) != 0;
        px.posY = (px.inputAddr & PixelInputBit(PixelInput::PositionY)) != 0;
        px.posZ = (px.inputAddr & PixelInputBit(PixelInput::PositionZ)) != 0;
        px.posW = (px.inputAddr & PixelInputBit(PixelInput::PositionW)) != 0;
        px.frontFace = (px.inputAddr & PixelInputBit(PixelInput::FrontFace)) != 0;
        for (std::uint32_t i = 0; i < 8; i++) { px.targetExportMapping[i] = static_cast<std::uint8_t>(i); px.targetOutputMode[i] = 4; px.targetExportPacking[i] = ColorExportPacking::None; }
        // argv[9]: hex mask of attributes the code reads with 16-bit interpolation (FP16_INTERP_MODE = bit 19 of the setting).
        const std::uint32_t fp16Mask = argc > 9 ? std::stoul(argv[9], nullptr, 16) : 0u;
        const std::uint32_t flatMask = argc > 10 ? std::stoul(argv[10], nullptr, 16) : 0u;
        for (std::uint32_t i = 0; i < px.interpolatorCount && i < 32; i++) px.interpolatorSettings[i] = i | (((fp16Mask >> i) & 1u) ? 0x80000u : 0u) | (((flatMask >> i) & 1u) ? 0x400u : 0u);
        request.context.pixel = px;
    } else if (stage == "vs" || stage == "ms" || stage == "gs") {
        ShaderVertexStageInfo vx{};
        if (stage == "vs" && argc > 7 && std::string(argv[7]) != "-") {
            std::ifstream vf(argv[7]);
            if (!vf) { std::printf("THROW cannot open vertex input table %s\n", argv[7]); return 1; }
            std::string kind;
            while (vf >> kind) {
                if (kind != "vin") { std::string rest; std::getline(vf, rest); continue; }
                std::int32_t vgpr, count, attr; vf >> vgpr >> count >> attr;
                if (vx.resourcesNum >= ShaderVertexStageInfo::MaxResources) { std::printf("THROW too many vertex inputs\n"); return 1; }
                vx.resourcesDst[vx.resourcesNum++] = {vgpr, count, attr, 0u};
            }
            if (vx.resourcesNum != 0) {
                vx.fetchEmbedded = true;
                vx.fetchAttribReg = 96u;
                vx.fetchBufferReg = 96u;
            }
        }
        request.context.vertex = vx;
        if (esgsItem != 0u) {
            GraphicsCompileContext g{};
            g.firstUserSgpr = 0;
            GeometryConfiguration geom{};
            geom.esgsItemSize = esgsItem;
            geom.inputAttributeCount = (esgsItem + 3u) / 4u;
            geom.gsvsOffsetDwords = std::max<std::uint32_t>(64u, (3u * esgsItem + 63u) & ~63u);
            g.geometry = geom;
            request.graphics = g;
        }
    }
    // For ps, argv[11] optionally supplies FLOAT_MODE in hex (argv[10] is the flat interpolant mask). Other stages keep their own argument layout.
    const auto floatMode = stage == "ps" && argc > 11 ? std::stoul(argv[11], nullptr, 16) : 0xc0u;
    if (floatMode > 0xffu) {
        std::fprintf(stderr, "pixel FLOAT_MODE must fit in one byte\n");
        return 2;
    }
    if (stage != "cs") request.context.floatMode = ShaderFloatMode{static_cast<std::uint8_t>(floatMode), true, false, false};
    if (stage == "ms") {
        // Default NGG/mesh configuration (nothing here is read from the shader): triangle list, one 128-thread group.
        GraphicsCompileContext g{};
        g.firstUserSgpr = 0;
        g.mesh = MeshConfiguration{3u, 42u, 126u, 126u, 42u, 128u, 0u, 0u, 4u, 3u};
        g.draw = GraphicsDrawParameters{0x30000000ull, 3u, 4u, 1u};
        request.graphics = g;
        request.target.mesh = MeshTargetLimits{{128u, 128u, 128u}, 128u, 32768u, 256u, 256u, 128u, 32768u, 1u, 1u};
    }
    if (stage == "gs") {
        GraphicsCompileContext g{};
        g.firstUserSgpr = 0;
        GeometryConfiguration geom{};
        if (argc > 7) geom.inputPrimitive = std::stoul(argv[7]);
        if (argc > 8) geom.outputTopology = std::stoul(argv[8]);
        if (argc > 9) geom.maxOutputVertices = std::stoul(argv[9]);
        if (argc > 10) geom.esgsItemSize = std::stoul(argv[10]);
        if (argc > 11) geom.gsvsVertexSize = std::stoul(argv[11]);
        if (argc > 12) geom.inputAttributeCount = std::stoul(argv[12]);
        if (argc > 13) geom.outputAttributeCount = std::stoul(argv[13]);
        const std::uint32_t inVerts = geom.inputPrimitive == 1u ? 1u : geom.inputPrimitive == 2u ? 2u : geom.inputPrimitive == 3u ? 3u : geom.inputPrimitive == 4u ? 4u : 6u;
        geom.gsvsOffsetDwords = std::max<std::uint32_t>(64u, ((inVerts * geom.esgsItemSize + 63u) & ~63u));
        g.geometry = geom;
        request.graphics = g;
    }
    request.shader = {stage == "ms" ? ShaderStage::Mesh : stage == "gs" ? ShaderStage::Geometry : stage == "ps" ? ShaderStage::Fragment : stage == "vs" ? ShaderStage::Vertex : ShaderStage::Compute, 0x20000u, code, 0, {}};
    request.context.waveSize = wave;
    request.context.userDataBaseRegister = udBase;
    request.context.userData = userData;
    if (stage == "cs") request.context.compute = ShaderComputeStageInfo{{wave, 1u, 1u}, 0u, {true, true, true}, false, 1u};
    request.target.vulkanVersion = 0x00403000u;
    request.target.spirvVersion = 0x00010600u;
    request.target.subgroupSize = wave;
    request.target.bdaAbiVersion = 1;
    // XBOX_FRAGMENT_BARY=1: model a host with VK_KHR_fragment_shader_barycentric (no geometry-shader barycentric emulation).
    std::vector<std::uint32_t> capsLocal = caps;
    std::vector<std::string_view> extLocal(ext.begin(), ext.end());
    if (std::getenv("XBOX_FRAGMENT_BARY")) {
        request.target.fragmentShaderBarycentricEnabled = true;
        request.target.fragmentBarycentricBuiltins = true;
        capsLocal.push_back(5284u);
        extLocal.push_back("SPV_KHR_fragment_shader_barycentric");
    }
    request.target.supportedCapabilities = capsLocal;
    request.target.supportedExtensions = extLocal;
    request.target.maxWorkgroupSize = {1024u, 1024u, 64u};
    request.target.maxWorkgroupInvocations = 1024;
    request.target.maxWorkgroupSharedMemoryBytes = 65536;
    request.layout = {0, 0, 0, 128};
    request.useCache = false;
    const auto writeOutput = [&](const RecompileResult& result) -> int {
        if (result.spirv.empty()) { std::printf("EMPTY\n"); return 1; }
        std::ofstream o(argv[3], std::ios::binary);
        const auto words = result.spirv.Words();
        o.write(reinterpret_cast<const char*>(words.data()), words.size() * 4);
        o.close();
        if (!o) {
            std::printf("IOERROR cannot write SPIR-V output\n");
            return 1;
        }
        if (std::getenv("XBOX_DUMP_ABI")) {
            std::printf("abi: bdaVer=%u memOffDword=%u shaderDataDwords=%u imgMetaDword=%u runtimeImages=%u push=%zu specConsts=%zu userWords=%zu wg=%u\n", result.bdaAbiVersion, result.memoryOffsetDword, result.shaderDataDwords, result.imageMetadataDword, result.runtimeImageCount, result.pushConstants.size(), result.specialization.size(), result.shaderDataUserWords.size(), result.workgroupMemoryDwords);
            for (const auto& b : result.bindings) std::printf("  binding set=%u b=%u kind=%d role=%d count=%u guest=%zu ro=%d\n", b.descriptorSet, b.binding, static_cast<int>(b.kind), static_cast<int>(b.role), b.count, b.guestDescriptor.size(), b.readOnly ? 1 : 0);
        }
        std::printf("OK %zu\n", words.size());
        return 0;
    };
    const auto reportError = [](const std::exception& e) -> int {
        std::string m = e.what();
        for (auto& c : m) if (c == '\n' || c == '\r') c = ' ';
        std::printf("THROW %s\n", m.c_str());
        return 1;
    };
    try {
        return writeOutput(Recompile(request));
    } catch (const std::exception& e) {
        if (variant == 1 && !bufImgCollisions.empty() && std::string_view(e.what()).find("buffer descriptor uses an unsupported type") != std::string_view::npos) {
            const std::size_t existingBackings = bufferBacking.size();
            for (std::size_t i = 0; i < existingBackings; ++i) {
                if (!bufferBacking[i].second.empty()) bufferBacking[i].second[0] = 1u;
            }
            for (const auto& [key, collided] : bufImgCollisions) {
                if (!collided) continue;
                auto it = nodes.find(key.first);
                if (it == nodes.end()) continue;
                auto& m = it->second.mem;
                const std::size_t w = key.second / 4;
                if (w + 12 <= m.size() && m[w + 8] == 0u && m[w + 11] == 0u) {
                    const std::uint64_t bbase = 0x50000000ull + 0x10000ull * counter++;
                    const std::uint32_t v[4] = {static_cast<std::uint32_t>(bbase), static_cast<std::uint32_t>((bbase >> 32) & 0xffffu) | (16u << 16), 0x10000u, 0x30027facu};
                    std::copy(v, v + 4, m.begin() + w + 8);
                    bufferBacking.push_back({bbase, std::vector<std::uint32_t>(0x4000)});
                }
            }
            regions.clear();
            for (auto& [id, n] : nodes) regions.push_back({n.base, std::as_bytes(std::span<const std::uint32_t>(n.mem))});
            for (auto& [base, mem] : bufferBacking) regions.push_back({base, std::as_bytes(std::span<const std::uint32_t>(mem))});
            request.context.memory = MemoryRegionView(std::span<const MemoryRegion>(regions));
            try {
                return writeOutput(Recompile(request));
            } catch (const std::exception& retryError) {
                return reportError(retryError);
            }
        }
        return reportError(e);
    }
}
