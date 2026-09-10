#include "ge_gxm_shader.hpp"

#ifdef _IRR_COMPILE_WITH_GXM_

#include "ge_gxm_limits.hpp"
#include "ge_gxm_memory.hpp"

#include "../source/Irrlicht/os.h"

#include <shacccg_ext.h>
#include <vitashark.h>

#include <sys/stat.h>

#include <cstdio>
#include <cstring>
#include <memory>
#include <sstream>

namespace GE
{
namespace
{
std::string g_cache_dir;

bool g_shark_ready = false;

/** Whether the SceShaccCg extension patches are in effect. */
bool g_extensions_enabled = false;

SceGxmShaderPatcher* g_patcher = NULL;

/** Backing store the patcher carves its host and buffer allocations out of. */
void* g_patcher_buffer = NULL;
void* g_patcher_vertex_usse = NULL;
void* g_patcher_fragment_usse = NULL;

std::unordered_map<std::string, std::unique_ptr<GEGXMProgram> > g_programs;

unsigned g_compiled_count = 0;
unsigned g_total_count = 0;
unsigned g_generation = 1;

/** Result of probeCapabilities(); 0 until init() has run. */
unsigned g_max_joints = 0;

/** Bumped whenever the shader sources or the compiler settings change in a way
 *  that invalidates every cached binary. Part of the cache key, so an old
 *  cache is simply never hit rather than having to be deleted. */
const unsigned CACHE_VERSION = 2;

// ----------------------------------------------------------------------------
uint64_t hashString(const std::string& s)
{
    // FNV-1a. Only has to separate shader variants, not resist anything.
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : s)
    {
        h ^= c;
        h *= 1099511628211ull;
    }
    return h;
}   // hashString

// ----------------------------------------------------------------------------
void* patcherHostAlloc(void* user_data, SceSize size)
{
    return malloc(size);
}   // patcherHostAlloc

// ----------------------------------------------------------------------------
void patcherHostFree(void* user_data, void* mem)
{
    free(mem);
}   // patcherHostFree

/** Source currently being compiled, so a diagnostic can quote the line it is
 *  complaining about. The line numbers the compiler reports are into the
 *  composed source - the limits block, the shared helpers and the shader body
 *  concatenated - which corresponds to nothing a reader can look up by hand, so
 *  quoting is the only way to make a diagnostic actionable. */
const std::string* g_compiling_source = NULL;
std::string g_compiling_name;

// ----------------------------------------------------------------------------
/** Copies the 1-based line \p line out of \p source, without its newline. */
std::string extractLine(const std::string& source, int line)
{
    if (line < 1)
        return "";
    size_t start = 0;
    for (int i = 1; i < line; i++)
    {
        start = source.find('\n', start);
        if (start == std::string::npos)
            return "";
        start++;
    }
    size_t end = source.find('\n', start);
    if (end == std::string::npos)
        end = source.size();
    return source.substr(start, end - start);
}   // extractLine

// ----------------------------------------------------------------------------
void sharkLog(const char* msg, shark_log_level level, int line)
{
    irr::ELOG_LEVEL ll = irr::ELL_INFORMATION;
    if (level == SHARK_LOG_WARNING)
        ll = irr::ELL_WARNING;
    else if (level == SHARK_LOG_ERROR)
        ll = irr::ELL_ERROR;

    char buf[640];
    snprintf(buf, sizeof(buf), "GXM shader %s (line %d): %s",
        g_compiling_name.empty() ? "compiler" : g_compiling_name.c_str(),
        line, msg);
    irr::os::Printer::log(buf, ll);

    if (g_compiling_source != NULL && line >= 1)
    {
        // Three lines of context is enough to identify the construct without
        // dumping a shader that is a few hundred lines long after composition.
        for (int i = line - 1; i <= line + 1; i++)
        {
            const std::string text = extractLine(*g_compiling_source, i);
            if (text.empty() && i != line)
                continue;
            snprintf(buf, sizeof(buf), "  %s%5d | %.500s",
                i == line ? ">" : " ", i, text.c_str());
            irr::os::Printer::log(buf, ll);
        }
    }
}   // sharkLog

// ----------------------------------------------------------------------------
/** Marker written next to the shader cache when the SceShaccCg extension
 *  patches turn out to break the compiler here.
 *
 *  Worth persisting because finding out costs a probe compile that faults its
 *  way through the corrupted module - harmless, but under Vita3K it produces
 *  hundreds of megabytes of emulator log. With the marker that happens once on
 *  a given install rather than on every launch. */
std::string getNoExtensionsMarkerPath()
{
    if (g_cache_dir.empty())
        return "";
    // Versioned: a verdict recorded by an older build may have blamed the
    // extensions for something else entirely, so it must not be inherited.
    return g_cache_dir + "no_shacccg_ext_v" + std::to_string(CACHE_VERSION);
}   // getNoExtensionsMarkerPath

// ----------------------------------------------------------------------------
bool extensionsKnownBroken()
{
    const std::string path = getNoExtensionsMarkerPath();
    if (path.empty())
        return false;
    FILE* f = fopen(path.c_str(), "rb");
    if (f == NULL)
        return false;
    fclose(f);
    return true;
}   // extensionsKnownBroken

// ----------------------------------------------------------------------------
void markExtensionsBroken()
{
    const std::string path = getNoExtensionsMarkerPath();
    if (path.empty())
        return;
    FILE* f = fopen(path.c_str(), "wb");
    if (f == NULL)
        return;
    // Content is for whoever finds the file, not for this code.
    fputs("The SceShaccCg extension patches crash the shader compiler on this "
          "system, so they are skipped. Delete this file to try them again.\n",
          f);
    fclose(f);
}   // markExtensionsBroken

// ----------------------------------------------------------------------------
std::string getCachePath(const std::string& name, uint64_t hash)
{
    if (g_cache_dir.empty())
        return "";
    char buf[64];
    snprintf(buf, sizeof(buf), "_%016llx.gxp", (unsigned long long)hash);
    return g_cache_dir + name + buf;
}   // getCachePath

// ----------------------------------------------------------------------------
bool readCache(const std::string& path, std::vector<uint8_t>* out)
{
    if (path.empty())
        return false;
    FILE* f = fopen(path.c_str(), "rb");
    if (f == NULL)
        return false;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (size <= 0)
    {
        fclose(f);
        return false;
    }
    out->resize((size_t)size);
    bool ok = fread(out->data(), 1, (size_t)size, f) == (size_t)size;
    fclose(f);
    if (!ok)
    {
        out->clear();
        return false;
    }
    // Never hand sceGxm a truncated or corrupt binary; it will fault inside
    // the shader patcher rather than return an error.
    if (sceGxmProgramCheck((const SceGxmProgram*)out->data()) < 0)
    {
        out->clear();
        return false;
    }
    return true;
}   // readCache

// ----------------------------------------------------------------------------
void writeCache(const std::string& path, const std::vector<uint8_t>& data)
{
    if (path.empty() || data.empty())
        return;
    // Write to a temporary then rename, so a crash or a power loss halfway
    // through cannot leave a half written binary that would be loaded (and
    // rejected, or worse accepted) next boot.
    std::string tmp = path + ".tmp";
    FILE* f = fopen(tmp.c_str(), "wb");
    if (f == NULL)
        return;
    bool ok = fwrite(data.data(), 1, data.size(), f) == data.size();
    fclose(f);
    if (!ok)
    {
        remove(tmp.c_str());
        return;
    }
    remove(path.c_str());
    rename(tmp.c_str(), path.c_str());
}   // writeCache

// ----------------------------------------------------------------------------
GEGXMProgram* getProgram(const std::string& name, const std::string& defines,
                         bool fragment)
{
    std::string key = std::string(fragment ? "f:" : "v:") + name + "|" +
        defines;
    auto it = g_programs.find(key);
    if (it != g_programs.end())
        return it->second.get();

    const std::string& body = getGXMShaderSource(name);
    if (body.empty())
    {
        irr::os::Printer::log("GXM: unknown shader", name.c_str(),
            irr::ELL_ERROR);
        return NULL;
    }

    std::string source = getGXMLimitsDefines();
    if (!defines.empty())
        source += defines + "\n";
    source += body;

    std::ostringstream cache_key;
    cache_key << CACHE_VERSION << "|" << (fragment ? 'f' : 'v') << "|"
        << source;
    uint64_t hash = hashString(cache_key.str());
    std::string cache_path = getCachePath(name, hash);

    std::vector<uint8_t> binary;
    if (!readCache(cache_path, &binary))
    {
        if (!g_shark_ready)
        {
            irr::os::Printer::log("GXM: shader not cached and the runtime "
                "compiler is unavailable", name.c_str(), irr::ELL_ERROR);
            return NULL;
        }
        // In/out: vitaShaRK reads this as the length of the source
        // (shark_input.size = *size) and only afterwards overwrites it with the
        // size of the compiled program. Passing 0 hands the compiler an empty
        // source, which it reports as "fatal internal error" with no line
        // number rather than as an empty input.
        uint32_t size = (uint32_t)source.size();
        // O2 with fast maths: the SGX has no denormals to preserve and STK's
        // shaders do not depend on strict IEEE behaviour. Fast integer maths
        // is left off because the skinning and material id paths do rely on
        // exact 32 bit integer arithmetic.
        g_compiling_source = &source;
        g_compiling_name = name;
        SceGxmProgram* compiled = shark_compile_shader_extended(
            source.c_str(), &size,
            fragment ? SHARK_FRAGMENT_SHADER : SHARK_VERTEX_SHADER,
            SHARK_OPT_DEFAULT, SHARK_ENABLE, SHARK_ENABLE, SHARK_DISABLE);
        g_compiling_source = NULL;
        g_compiling_name.clear();
        if (compiled == NULL || size == 0)
        {
            irr::os::Printer::log("GXM: failed to compile shader",
                name.c_str(), irr::ELL_ERROR);
            shark_clear_output();
            return NULL;
        }
        binary.assign((uint8_t*)compiled, (uint8_t*)compiled + size);
        shark_clear_output();
        g_compiled_count++;
        writeCache(cache_path, binary);
    }
    g_total_count++;

    std::unique_ptr<GEGXMProgram> program(
        new GEGXMProgram(name, std::move(binary), fragment));
    if (program->getProgram() == NULL)
        return NULL;
    GEGXMProgram* ret = program.get();
    g_programs[key] = std::move(program);
    return ret;
}   // getProgram
}   // namespace

// ============================================================================
GEGXMProgram::GEGXMProgram(const std::string& name,
                           std::vector<uint8_t> binary, bool fragment)
{
    m_id = NULL;
    m_program = NULL;
    m_name = name;
    m_fragment = fragment;
    m_binary = std::move(binary);

    const SceGxmProgram* p = (const SceGxmProgram*)m_binary.data();
    if (sceGxmProgramCheck(p) < 0)
    {
        irr::os::Printer::log("GXM: sceGxmProgramCheck failed", name.c_str(),
            irr::ELL_ERROR);
        return;
    }
    SceGxmProgramType type = sceGxmProgramGetType(p);
    if ((type == SCE_GXM_FRAGMENT_PROGRAM) != fragment)
    {
        irr::os::Printer::log("GXM: shader compiled for the wrong pipeline "
            "stage", name.c_str(), irr::ELL_ERROR);
        return;
    }
    if (sceGxmShaderPatcherRegisterProgram(g_patcher, p, &m_id) < 0)
    {
        irr::os::Printer::log("GXM: sceGxmShaderPatcherRegisterProgram failed",
            name.c_str(), irr::ELL_ERROR);
        m_id = NULL;
        return;
    }
    m_program = p;
}   // GEGXMProgram

// ----------------------------------------------------------------------------
GEGXMProgram::~GEGXMProgram()
{
    if (m_id != NULL && g_patcher != NULL)
        sceGxmShaderPatcherUnregisterProgram(g_patcher, m_id);
}   // ~GEGXMProgram

// ----------------------------------------------------------------------------
const SceGxmProgramParameter* GEGXMProgram::getParameter(
                                                const std::string& name) const
{
    auto it = m_parameters.find(name);
    if (it != m_parameters.end())
        return it->second;
    const SceGxmProgramParameter* p = m_program == NULL ? NULL :
        sceGxmProgramFindParameterByName(m_program, name.c_str());
    m_parameters[name] = p;
    return p;
}   // getParameter

// ============================================================================
bool GEGXMVertexLayout::operator==(const GEGXMVertexLayout& o) const
{
    if (m_attributes.size() != o.m_attributes.size() ||
        m_streams.size() != o.m_streams.size())
        return false;
    return memcmp(m_attributes.data(), o.m_attributes.data(),
        m_attributes.size() * sizeof(SceGxmVertexAttribute)) == 0 &&
        memcmp(m_streams.data(), o.m_streams.data(),
        m_streams.size() * sizeof(SceGxmVertexStream)) == 0;
}   // operator==

// ----------------------------------------------------------------------------
std::string GEGXMVertexLayout::getKey() const
{
    std::string key;
    key.reserve(m_attributes.size() * sizeof(SceGxmVertexAttribute) +
        m_streams.size() * sizeof(SceGxmVertexStream));
    key.append((const char*)m_attributes.data(),
        m_attributes.size() * sizeof(SceGxmVertexAttribute));
    key.push_back('|');
    key.append((const char*)m_streams.data(),
        m_streams.size() * sizeof(SceGxmVertexStream));
    return key;
}   // getKey

// ----------------------------------------------------------------------------
bool buildVertexLayout(GEGXMProgram* program,
                       const GEGXMAttributeDesc* attributes, unsigned count,
                       const SceGxmVertexStream* streams,
                       unsigned stream_count, GEGXMVertexLayout* out)
{
    if (program == NULL || program->getProgram() == NULL || out == NULL)
        return false;

    out->m_attributes.clear();
    out->m_streams.assign(streams, streams + stream_count);
    for (unsigned i = 0; i < count; i++)
    {
        const GEGXMAttributeDesc& d = attributes[i];
        const SceGxmProgramParameter* p = program->getParameter(d.m_name);
        if (p == NULL)
            continue;
        if (sceGxmProgramParameterGetCategory(p) !=
            SCE_GXM_PARAMETER_CATEGORY_ATTRIBUTE)
            continue;
        SceGxmVertexAttribute attr;
        attr.streamIndex = d.m_stream;
        attr.offset = d.m_offset;
        attr.format = d.m_format;
        attr.componentCount = d.m_component_count;
        attr.regIndex = (uint16_t)sceGxmProgramParameterGetResourceIndex(p);
        out->m_attributes.push_back(attr);
    }
    return true;
}   // buildVertexLayout

// ============================================================================
GEGXMBlendState::GEGXMBlendState()
{
    m_enabled = false;
    m_color_mask = SCE_GXM_COLOR_MASK_ALL;
    m_color_func = SCE_GXM_BLEND_FUNC_NONE;
    m_alpha_func = SCE_GXM_BLEND_FUNC_NONE;
    m_color_src = SCE_GXM_BLEND_FACTOR_ONE;
    m_color_dst = SCE_GXM_BLEND_FACTOR_ZERO;
    m_alpha_src = SCE_GXM_BLEND_FACTOR_ONE;
    m_alpha_dst = SCE_GXM_BLEND_FACTOR_ZERO;
}   // GEGXMBlendState

// ----------------------------------------------------------------------------
GEGXMBlendState GEGXMBlendState::opaque()
{
    return GEGXMBlendState();
}   // opaque

// ----------------------------------------------------------------------------
GEGXMBlendState GEGXMBlendState::alphaBlend()
{
    GEGXMBlendState b;
    b.m_enabled = true;
    b.m_color_func = SCE_GXM_BLEND_FUNC_ADD;
    b.m_alpha_func = SCE_GXM_BLEND_FUNC_ADD;
    b.m_color_src = SCE_GXM_BLEND_FACTOR_SRC_ALPHA;
    b.m_color_dst = SCE_GXM_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    b.m_alpha_src = SCE_GXM_BLEND_FACTOR_ONE;
    b.m_alpha_dst = SCE_GXM_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    return b;
}   // alphaBlend

// ----------------------------------------------------------------------------
GEGXMBlendState GEGXMBlendState::premultipliedAlphaBlend()
{
    GEGXMBlendState b = alphaBlend();
    b.m_color_src = SCE_GXM_BLEND_FACTOR_ONE;
    return b;
}   // premultipliedAlphaBlend

// ----------------------------------------------------------------------------
GEGXMBlendState GEGXMBlendState::additive()
{
    GEGXMBlendState b;
    b.m_enabled = true;
    b.m_color_func = SCE_GXM_BLEND_FUNC_ADD;
    b.m_alpha_func = SCE_GXM_BLEND_FUNC_ADD;
    b.m_color_src = SCE_GXM_BLEND_FACTOR_ONE;
    b.m_color_dst = SCE_GXM_BLEND_FACTOR_ONE;
    b.m_alpha_src = SCE_GXM_BLEND_FACTOR_ONE;
    b.m_alpha_dst = SCE_GXM_BLEND_FACTOR_ONE;
    return b;
}   // additive

// ----------------------------------------------------------------------------
uint32_t GEGXMBlendState::getKey() const
{
    if (!m_enabled)
        return 0x80000000u | m_color_mask;
    return (uint32_t)m_color_mask |
        ((uint32_t)m_color_func << 8) | ((uint32_t)m_alpha_func << 11) |
        ((uint32_t)m_color_src << 14) | ((uint32_t)m_color_dst << 18) |
        ((uint32_t)m_alpha_src << 22) | ((uint32_t)m_alpha_dst << 26);
}   // getKey

// ============================================================================
GEGXMProgramCache::~GEGXMProgramCache()
{
    destroy();
}   // ~GEGXMProgramCache

// ----------------------------------------------------------------------------
void GEGXMProgramCache::destroy()
{
    SceGxmShaderPatcher* patcher = GEGXMShaderManager::getPatcher();
    if (patcher != NULL)
    {
        for (auto& p : m_vertex_programs)
            sceGxmShaderPatcherReleaseVertexProgram(patcher, p.second);
        for (auto& p : m_fragment_programs)
            sceGxmShaderPatcherReleaseFragmentProgram(patcher, p.second);
    }
    const bool had_programs = !m_vertex_programs.empty() ||
        !m_fragment_programs.empty();
    m_vertex_programs.clear();
    m_fragment_programs.clear();
    // Bumped by the code that actually released the programs, so a draw call
    // holding a patched program pointer is told even if the caller forgets.
    if (had_programs)
        GEGXMShaderManager::invalidateGeneration();
}   // destroy

// ----------------------------------------------------------------------------
SceGxmVertexProgram* GEGXMProgramCache::getVertexProgram(
                    GEGXMProgram* program, const GEGXMVertexLayout& layout)
{
    if (program == NULL || program->getId() == NULL)
        return NULL;

    std::string key = program->getName() + "\x1f" +
        std::to_string((uintptr_t)program) + "\x1f" + layout.getKey();
    auto it = m_vertex_programs.find(key);
    if (it != m_vertex_programs.end())
        return it->second;

    SceGxmVertexProgram* vp = NULL;
    int err = sceGxmShaderPatcherCreateVertexProgram(
        GEGXMShaderManager::getPatcher(), program->getId(),
        layout.m_attributes.empty() ? NULL : layout.m_attributes.data(),
        (unsigned)layout.m_attributes.size(),
        layout.m_streams.empty() ? NULL : layout.m_streams.data(),
        (unsigned)layout.m_streams.size(), &vp);
    if (err < 0 || vp == NULL)
    {
        char buf[128];
        snprintf(buf, sizeof(buf), "GXM: CreateVertexProgram failed for %s "
            "(0x%08x)", program->getName().c_str(), (unsigned)err);
        irr::os::Printer::log(buf, irr::ELL_ERROR);
        return NULL;
    }
    m_vertex_programs[key] = vp;
    return vp;
}   // getVertexProgram

// ----------------------------------------------------------------------------
SceGxmFragmentProgram* GEGXMProgramCache::getFragmentProgram(
                              GEGXMProgram* program,
                              GEGXMProgram* vertex_program,
                              const GEGXMBlendState& blend,
                              SceGxmOutputRegisterFormat format,
                              SceGxmMultisampleMode msaa)
{
    if (program == NULL || program->getId() == NULL)
        return NULL;

    char suffix[64];
    snprintf(suffix, sizeof(suffix), "\x1f%08x\x1f%u\x1f%u",
        (unsigned)blend.getKey(), (unsigned)format, (unsigned)msaa);
    std::string key = program->getName() + "\x1f" +
        std::to_string((uintptr_t)program) + "\x1f" +
        std::to_string((uintptr_t)vertex_program) + suffix;
    auto it = m_fragment_programs.find(key);
    if (it != m_fragment_programs.end())
        return it->second;

    SceGxmBlendInfo blend_info;
    SceGxmBlendInfo* blend_ptr = NULL;
    if (blend.m_enabled || blend.m_color_mask != SCE_GXM_COLOR_MASK_ALL)
    {
        memset(&blend_info, 0, sizeof(blend_info));
        blend_info.colorMask = blend.m_color_mask;
        blend_info.colorFunc = blend.m_color_func;
        blend_info.alphaFunc = blend.m_alpha_func;
        blend_info.colorSrc = blend.m_color_src;
        blend_info.colorDst = blend.m_color_dst;
        blend_info.alphaSrc = blend.m_alpha_src;
        blend_info.alphaDst = blend.m_alpha_dst;
        blend_ptr = &blend_info;
    }

    SceGxmFragmentProgram* fp = NULL;
    int err = sceGxmShaderPatcherCreateFragmentProgram(
        GEGXMShaderManager::getPatcher(), program->getId(), format, msaa,
        blend_ptr,
        vertex_program == NULL ? NULL : vertex_program->getProgram(), &fp);
    if (err < 0 || fp == NULL)
    {
        char buf[128];
        snprintf(buf, sizeof(buf), "GXM: CreateFragmentProgram failed for %s "
            "(0x%08x)", program->getName().c_str(), (unsigned)err);
        irr::os::Printer::log(buf, irr::ELL_ERROR);
        return NULL;
    }
    m_fragment_programs[key] = fp;
    return fp;
}   // getFragmentProgram

// ============================================================================
namespace GEGXMShaderManager
{
// ----------------------------------------------------------------------------
/** Compiles a shader without caching or registering it, purely to find out
 *  whether the compiler accepts it. Used by probeCapabilities(). */
bool tryCompile(const std::string& name, const std::string& defines,
                bool fragment)
{
    const std::string& body = getGXMShaderSource(name);
    if (body.empty())
        return false;
    std::string source = getGXMLimitsDefines();
    if (!defines.empty())
        source += defines + "\n";
    source += body;

    // In/out, see getProgram(): this has to be the source length going in.
    uint32_t size = (uint32_t)source.size();
    g_compiling_source = &source;
    g_compiling_name = name;
    SceGxmProgram* compiled = shark_compile_shader_extended(source.c_str(),
        &size, fragment ? SHARK_FRAGMENT_SHADER : SHARK_VERTEX_SHADER,
        SHARK_OPT_DEFAULT, SHARK_ENABLE, SHARK_ENABLE, SHARK_DISABLE);
    g_compiling_source = NULL;
    g_compiling_name.clear();
    const bool ok = compiled != NULL && size != 0;
    shark_clear_output();
    return ok;
}   // tryCompile

// ----------------------------------------------------------------------------
/** Works out what this device's shader compiler can actually do.
 *
 *  Two things have to be discovered rather than assumed:
 *
 *  1. Whether SceShaccCgExt's module patches work at all. They are taiHEN
 *     injections into a system module, and under Vita3K the patched code faults
 *     - the compiler then fails every shader with "fatal internal error". So the
 *     compiler is brought up with the patches, probed with the simplest shader
 *     in the set, and if that fails it is torn down and brought back up without
 *     them (shark_init_simple), which is what that entry point is for.
 *  2. How large a joint palette the vertex stage will take. That depends on the
 *     patches, because raising the secondary attribute register limit is one of
 *     the extensions. Rather than reason about how the compiler assigns
 *     registers to a dynamically indexed array, just ask it, halving until it
 *     agrees. Zero means skinned meshes must be drawn in bind pose.
 *
 *  Each probe compile is a few tens of milliseconds and happens once. */
void probeCapabilities()
{
    g_max_joints = 0;
    if (!g_shark_ready)
        return;

    if (!tryCompile("fullscreen.vert", "", false))
    {
        irr::os::Printer::log("GXM: the shader compiler rejects even a trivial "
            "shader with the SceShaccCg extensions enabled; releasing the "
            "extension patches and retrying without them", irr::ELL_WARNING);
        // Release the taiHEN injections rather than tearing the compiler down.
        // shark_end() would call sceShaccCgReleaseCompiler() first, which runs
        // through the very code the patches corrupted; disabling the extensions
        // restores the original instructions, and the allocator and callback
        // list shark_init() set up are still valid afterwards.
        if (g_extensions_enabled)
        {
            sceShaccCgExtDisableExtensions();
            g_extensions_enabled = false;
            // So the next launch does not have to find this out the hard way.
            markExtensionsBroken();
        }
        if (!tryCompile("fullscreen.vert", "", false))
        {
            irr::os::Printer::log("GXM: the runtime shader compiler is "
                "unusable; only shaders already in the on-disk cache can be "
                "used", irr::ELL_ERROR);
            g_shark_ready = false;
            return;
        }
        irr::os::Printer::log("GXM: running the shader compiler without the "
            "SceShaccCg extensions; the joint palette will be smaller",
            irr::ELL_INFORMATION);
    }

    for (unsigned joints = GXM_MAX_JOINTS_CEILING; joints > 0; joints /= 2)
    {
        g_max_joints = joints;
        if (tryCompile("spm.vert", "#define GXM_SKINNING 1", false))
            break;
        g_max_joints = 0;
    }

    char buf[128];
    if (g_max_joints == 0)
    {
        snprintf(buf, sizeof(buf), "GXM: no joint palette size compiled; "
            "skinned meshes will be drawn in bind pose");
        irr::os::Printer::log(buf, irr::ELL_WARNING);
    }
    else
    {
        snprintf(buf, sizeof(buf), "GXM: joint palette holds %u joints",
            g_max_joints);
        irr::os::Printer::log(buf, irr::ELL_INFORMATION);
    }
}   // probeCapabilities

// ----------------------------------------------------------------------------
bool init()
{
    if (g_patcher != NULL)
        return true;

    g_compiled_count = 0;
    g_total_count = 0;

    // The patcher needs three kinds of memory from us: a GPU visible scratch
    // buffer for the patched programs' data, and one USSE code region per
    // pipeline stage. Sizes here are the usual ones from Sony's samples,
    // doubled for the buffer because STK patches a lot of blend variants.
    const size_t BUFFER_SIZE = 1024 * 1024;
    const size_t VERTEX_USSE_SIZE = 256 * 1024;
    const size_t FRAGMENT_USSE_SIZE = 256 * 1024;

    g_patcher_buffer = GEGXMMemory::allocate(GGMP_LPDDR_UNCACHED, BUFFER_SIZE,
        4096);
    unsigned vertex_usse_offset = 0;
    unsigned fragment_usse_offset = 0;
    g_patcher_vertex_usse = GEGXMMemory::allocateVertexUsse(VERTEX_USSE_SIZE,
        &vertex_usse_offset);
    g_patcher_fragment_usse = GEGXMMemory::allocateFragmentUsse(
        FRAGMENT_USSE_SIZE, &fragment_usse_offset);
    if (g_patcher_buffer == NULL || g_patcher_vertex_usse == NULL ||
        g_patcher_fragment_usse == NULL)
    {
        irr::os::Printer::log("GXM: out of memory setting up the shader "
            "patcher", irr::ELL_ERROR);
        destroy();
        return false;
    }

    SceGxmShaderPatcherParams params;
    memset(&params, 0, sizeof(params));
    params.userData = NULL;
    params.hostAllocCallback = &patcherHostAlloc;
    params.hostFreeCallback = &patcherHostFree;
    params.bufferAllocCallback = NULL;
    params.bufferFreeCallback = NULL;
    params.bufferMem = g_patcher_buffer;
    params.bufferMemSize = BUFFER_SIZE;
    params.vertexUsseAllocCallback = NULL;
    params.vertexUsseFreeCallback = NULL;
    params.vertexUsseMem = g_patcher_vertex_usse;
    params.vertexUsseMemSize = VERTEX_USSE_SIZE;
    params.vertexUsseOffset = vertex_usse_offset;
    params.fragmentUsseAllocCallback = NULL;
    params.fragmentUsseFreeCallback = NULL;
    params.fragmentUsseMem = g_patcher_fragment_usse;
    params.fragmentUsseMemSize = FRAGMENT_USSE_SIZE;
    params.fragmentUsseOffset = fragment_usse_offset;

    if (sceGxmShaderPatcherCreate(&params, &g_patcher) < 0)
    {
        irr::os::Printer::log("GXM: sceGxmShaderPatcherCreate failed",
            irr::ELL_ERROR);
        g_patcher = NULL;
        destroy();
        return false;
    }

    // The runtime compiler is only needed on a cold cache, so a failure here
    // is a warning: if every shader is already cached the driver still works.
    if (!g_shark_ready)
    {
        shark_install_log_cb(&sharkLog);
        shark_set_warnings_level(SHARK_WARN_MEDIUM);
        // shark_init() enables the SceShaccCg extension patches;
        // shark_init_simple() is the same thing without them. Skip straight to
        // the latter if a previous run recorded that they do not work here.
        const bool skip_extensions = extensionsKnownBroken();
        if (skip_extensions)
        {
            irr::os::Printer::log("GXM: skipping the SceShaccCg extensions, a "
                "previous run found them broken on this system",
                irr::ELL_INFORMATION);
        }
        const int shark_err = skip_extensions ? shark_init_simple(NULL) :
            shark_init(NULL);
        if (shark_err < 0)
        {
            irr::os::Printer::log("GXM: vitaShaRK is unavailable, only cached "
                "shaders can be used", irr::ELL_WARNING);
        }
        else if (skip_extensions)
        {
            g_shark_ready = true;
            g_extensions_enabled = false;
        }
        else
        {
            g_shark_ready = true;
            g_extensions_enabled = true;
            // The vertex stage defaults to 128 secondary attribute registers,
            // i.e. 128 floats of uniform data; the raised limit is 512, which is
            // what makes a useful joint palette possible. This is a patch into
            // the shader compiler module rather than a documented knob, so
            // probeCapabilities() below checks whether it worked instead of
            // assuming.
            if (sceShaccCgExtSetVertexSecondaryLimit(
                SCE_SHACCCG_EXT_VERTEX_SA_REG_LIMIT_512) < 0)
            {
                irr::os::Printer::log("GXM: could not raise the vertex uniform "
                    "register limit", irr::ELL_WARNING);
            }
        }
    }

    if (!g_cache_dir.empty())
        mkdir(g_cache_dir.c_str(), 0777);

    probeCapabilities();
    return true;
}   // init

// ----------------------------------------------------------------------------
void destroy()
{
    g_generation++;
    g_programs.clear();
    if (g_patcher != NULL)
    {
        sceGxmShaderPatcherDestroy(g_patcher);
        g_patcher = NULL;
    }
    if (g_shark_ready)
    {
        shark_end();
        g_shark_ready = false;
    }
    else if (g_extensions_enabled)
    {
        // shark_end() would have done this, but it is not being called - either
        // the compiler was never usable or probeCapabilities() gave up on it.
        sceShaccCgExtDisableExtensions();
    }
    g_extensions_enabled = false;
    g_max_joints = 0;
    if (g_patcher_vertex_usse != NULL)
    {
        GEGXMMemory::freeVertexUsse(g_patcher_vertex_usse);
        g_patcher_vertex_usse = NULL;
    }
    if (g_patcher_fragment_usse != NULL)
    {
        GEGXMMemory::freeFragmentUsse(g_patcher_fragment_usse);
        g_patcher_fragment_usse = NULL;
    }
    if (g_patcher_buffer != NULL)
    {
        GEGXMMemory::free(g_patcher_buffer);
        g_patcher_buffer = NULL;
    }
}   // destroy

// ----------------------------------------------------------------------------
SceGxmShaderPatcher* getPatcher()
{
    return g_patcher;
}   // getPatcher

// ----------------------------------------------------------------------------
GEGXMProgram* getVertexProgram(const std::string& name,
                               const std::string& defines)
{
    return GE::getProgram(name, defines, false);
}   // getVertexProgram

// ----------------------------------------------------------------------------
GEGXMProgram* getFragmentProgram(const std::string& name,
                                 const std::string& defines)
{
    return GE::getProgram(name, defines, true);
}   // getFragmentProgram

// ----------------------------------------------------------------------------
unsigned getCompiledCount()
{
    return g_compiled_count;
}   // getCompiledCount

// ----------------------------------------------------------------------------
unsigned getTotalCount()
{
    return g_total_count;
}   // getTotalCount

// ----------------------------------------------------------------------------
unsigned getGeneration()
{
    return g_generation;
}   // getGeneration

// ----------------------------------------------------------------------------
void invalidateGeneration()
{
    g_generation++;
}   // invalidateGeneration
}   // namespace GEGXMShaderManager

// ----------------------------------------------------------------------------
unsigned getGXMMaxJoints()
{
    return g_max_joints;
}   // getGXMMaxJoints

// ----------------------------------------------------------------------------
void setGXMShaderCacheDir(const std::string& dir)
{
    g_cache_dir = dir;
    if (!g_cache_dir.empty() && g_cache_dir.back() != '/')
        g_cache_dir += '/';
}   // setGXMShaderCacheDir

// ----------------------------------------------------------------------------
const std::string& getGXMShaderCacheDir()
{
    return g_cache_dir;
}   // getGXMShaderCacheDir

}   // namespace GE

#endif   // _IRR_COMPILE_WITH_GXM_
