#ifndef HEADER_GE_GXM_SHADER_HPP
#define HEADER_GE_GXM_SHADER_HPP

#include "IrrCompileConfig.h"

#ifdef _IRR_COMPILE_WITH_GXM_

#include <psp2/gxm.h>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace GE
{

/** A compiled and registered GXM program, plus the reflection the draw code
 *  needs to poke uniforms into it.
 *
 *  A ".gxp" is the compiled form of a Cg shader. It is produced offline by
 *  Sony's psp2cgc on a licensed SDK, which is not something a homebrew build
 *  has, so the driver compiles the Cg at runtime through vitaShaRK (which
 *  drives the SceShaccCg module present on every retail Vita) and caches the
 *  result on disk. First launch pays a few seconds; every launch after that
 *  reads the cache. */
class GEGXMProgram
{
private:
    SceGxmShaderPatcherId m_id;

    const SceGxmProgram* m_program;

    /** Owned copy of the compiled binary; sceGxm keeps pointing at it for as
     *  long as the program is registered, so it must outlive the patcher
     *  registration. */
    std::vector<uint8_t> m_binary;

    std::string m_name;

    bool m_fragment;

    mutable std::unordered_map<std::string, const SceGxmProgramParameter*>
        m_parameters;

public:
    // ------------------------------------------------------------------------
    GEGXMProgram(const std::string& name, std::vector<uint8_t> binary,
                 bool fragment);
    // ------------------------------------------------------------------------
    ~GEGXMProgram();
    // ------------------------------------------------------------------------
    SceGxmShaderPatcherId getId() const                        { return m_id; }
    // ------------------------------------------------------------------------
    const SceGxmProgram* getProgram() const               { return m_program; }
    // ------------------------------------------------------------------------
    const std::string& getName() const                       { return m_name; }
    // ------------------------------------------------------------------------
    bool isFragment() const                              { return m_fragment; }
    // ------------------------------------------------------------------------
    /** Cached sceGxmProgramFindParameterByName(). Returns NULL when the
     *  compiler optimised the uniform away, which is normal and which every
     *  caller has to tolerate. */
    const SceGxmProgramParameter* getParameter(const std::string& name) const;
    // ------------------------------------------------------------------------
    unsigned getDefaultUniformBufferSize() const
              { return sceGxmProgramGetDefaultUniformBufferSize(m_program); }
};   // GEGXMProgram

/** Everything to do with turning Cg text into something sceGxm can draw with:
 *  the shader patcher, the runtime compiler, the on-disk cache, and the
 *  vertex/fragment program variant caches. */
namespace GEGXMShaderManager
{
// ----------------------------------------------------------------------------
/** Brings up vitaShaRK and the shader patcher. Must be called after
 *  GEGXMMemory::init() and sceGxmInitialize().
 *  \return false if the runtime compiler is unavailable, in which case the
 *          driver cannot be created and STK falls back to another one. */
bool init();
// ----------------------------------------------------------------------------
void destroy();
// ----------------------------------------------------------------------------
SceGxmShaderPatcher* getPatcher();
// ----------------------------------------------------------------------------
/** Looks the named shader up in the built in source table, compiles it (or
 *  loads the cached binary), registers it with the patcher and keeps it.
 *  Repeated calls with the same name return the same object.
 *  \param defines optional preprocessor defines appended to the source; they
 *         are part of the cache key, so "solid" and "solid|PBR" are distinct
 *         cached binaries. */
GEGXMProgram* getVertexProgram(const std::string& name,
                               const std::string& defines = "");
// ----------------------------------------------------------------------------
GEGXMProgram* getFragmentProgram(const std::string& name,
                                 const std::string& defines = "");
// ----------------------------------------------------------------------------
/** Number of shaders compiled rather than read from the cache this run, for
 *  the loading screen and the log. */
unsigned getCompiledCount();
// ----------------------------------------------------------------------------
unsigned getTotalCount();
// ----------------------------------------------------------------------------
/** Bumped whenever the programs registered here, or the patched programs built
 *  from them, are thrown away - a driver teardown, a shader reload, a material
 *  set change.
 *
 *  GEGXMDrawCall caches raw patched vertex and fragment program pointers per
 *  material, and draw calls are recycled through the driver across track loads,
 *  so without this a recycled draw call could hand the GPU a pointer to a
 *  released program - which faults rather than failing cleanly. Draw calls
 *  compare the generation they were built against and drop their cache when it
 *  moves. */
unsigned getGeneration();
// ----------------------------------------------------------------------------
void invalidateGeneration();
};   // GEGXMShaderManager

/** Describes a vertex program variant: which attributes it reads out of which
 *  streams. Two draw calls asking for the same (program, layout) share one
 *  SceGxmVertexProgram. */
struct GEGXMVertexLayout
{
    std::vector<SceGxmVertexAttribute> m_attributes;
    std::vector<SceGxmVertexStream> m_streams;
    // ------------------------------------------------------------------------
    bool operator==(const GEGXMVertexLayout& o) const;
    // ------------------------------------------------------------------------
    std::string getKey() const;
};   // GEGXMVertexLayout

/** One entry of a vertex layout, before the register index is known.
 *
 *  GXM does not have named or numbered attribute locations: a
 *  SceGxmVertexAttribute has to carry the hardware register the compiler
 *  happened to assign to that input, which is only discoverable by looking the
 *  parameter up by name in the compiled program. So layouts are described by
 *  name here and resolved against a specific program by buildVertexLayout(). */
struct GEGXMAttributeDesc
{
    const char* m_name;
    uint16_t m_stream;
    uint16_t m_offset;
    /** One of SceGxmAttributeFormat. */
    uint8_t m_format;
    uint8_t m_component_count;
};   // GEGXMAttributeDesc

// ----------------------------------------------------------------------------
/** Resolves \p attributes against \p program, dropping any the compiler did
 *  not keep (an unused attribute is optimised out, which is normal - the depth
 *  only variants read a fraction of the vertex).
 *  \return false only if the program itself is unusable. */
bool buildVertexLayout(GEGXMProgram* program,
                       const GEGXMAttributeDesc* attributes, unsigned count,
                       const SceGxmVertexStream* streams,
                       unsigned stream_count, GEGXMVertexLayout* out);

/** Describes a fragment program variant: the blend state and output format it
 *  was patched for. On GXM the blend equation is baked into the fragment
 *  program at patch time rather than being dynamic state, so every distinct
 *  blend mode is a separate program object. */
struct GEGXMBlendState
{
    bool m_enabled;
    uint8_t m_color_mask;
    uint8_t m_color_func, m_alpha_func;
    uint8_t m_color_src, m_color_dst;
    uint8_t m_alpha_src, m_alpha_dst;
    // ------------------------------------------------------------------------
    GEGXMBlendState();
    // ------------------------------------------------------------------------
    static GEGXMBlendState opaque();
    // ------------------------------------------------------------------------
    static GEGXMBlendState alphaBlend();
    // ------------------------------------------------------------------------
    /** For shaders that have already multiplied colour by alpha. The 3D
     *  transparent shader does, matching the GLSL renderer, so applying alpha
     *  again in the blender would darken every transparent surface. */
    static GEGXMBlendState premultipliedAlphaBlend();
    // ------------------------------------------------------------------------
    static GEGXMBlendState additive();
    // ------------------------------------------------------------------------
    uint32_t getKey() const;
};   // GEGXMBlendState

/** Creates and owns the patched program objects, keyed so that identical
 *  requests are shared instead of re-patched (each patch costs USSE memory,
 *  which is a scarce 16 KB-granular resource). */
class GEGXMProgramCache
{
private:
    std::unordered_map<std::string, SceGxmVertexProgram*> m_vertex_programs;

    std::unordered_map<std::string, SceGxmFragmentProgram*>
        m_fragment_programs;

public:
    // ------------------------------------------------------------------------
    ~GEGXMProgramCache();
    // ------------------------------------------------------------------------
    SceGxmVertexProgram* getVertexProgram(GEGXMProgram* program,
                                          const GEGXMVertexLayout& layout);
    // ------------------------------------------------------------------------
    SceGxmFragmentProgram* getFragmentProgram(GEGXMProgram* program,
                                          GEGXMProgram* vertex_program,
                                          const GEGXMBlendState& blend,
                                          SceGxmOutputRegisterFormat format =
                                          SCE_GXM_OUTPUT_REGISTER_FORMAT_UCHAR4,
                                          SceGxmMultisampleMode msaa =
                                          SCE_GXM_MULTISAMPLE_NONE);
    // ------------------------------------------------------------------------
    void destroy();
};   // GEGXMProgramCache

// ----------------------------------------------------------------------------
/** The Cg source of a built in shader, or an empty string when unknown. */
const std::string& getGXMShaderSource(const std::string& name);
// ----------------------------------------------------------------------------
/** The #define block prepended to every shader, reflecting the startup probe. */
std::string getGXMLimitsDefines();

}   // namespace GE

#endif   // _IRR_COMPILE_WITH_GXM_

#endif
