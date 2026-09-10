#ifndef HEADER_GE_GXM_MESH_CACHE_HPP
#define HEADER_GE_GXM_MESH_CACHE_HPP

#include "IrrCompileConfig.h"

#ifdef _IRR_COMPILE_WITH_GXM_

#include "../source/Irrlicht/CMeshCache.h"

#include <cstddef>
#include <cstdint>

namespace irr
{
    namespace video { struct S3DVertexSkinnedMesh; }
}

namespace GE
{
class GEGXMDriver;
struct GXMSPMVertex;

// ----------------------------------------------------------------------------
/** Converts one irrlicht skinned vertex into the layout the GXM vertex fetch
 *  can read. Shared with GEGXMDrawCall, which has to do the same conversion for
 *  streamed buffers that never reach the cache. */
void convertSPMVertex(const irr::video::S3DVertexSkinnedMesh& in,
                      GXMSPMVertex* out);

/** Packs every loaded SPM mesh into one pair of GPU buffers.
 *
 *  Two things are happening here. The first is the same trick the Vulkan
 *  backend uses: one big vertex buffer and one big index buffer for the whole
 *  scene, with each mesh buffer recording its offset, so switching mesh does
 *  not mean rebinding a buffer - on GXM it is just a different pointer handed
 *  to sceGxmSetVertexStream().
 *
 *  The second is a format conversion that the Vulkan backend does not need.
 *  S3DVertexSkinnedMesh stores the normal and the tangent as A2B10G10R10
 *  packed into a u32, and SceGxmAttributeFormat has no packed 10-bit format at
 *  all - the vertex fetch hardware cannot unpack it. So both are expanded to
 *  four S16N components on the way in. That costs 8 bytes per vertex over the
 *  irrlicht layout and buys 16-bit normals instead of 10-bit, which is a fair
 *  trade at the resolution this renders at.
 *
 *  Skinning data goes in a second stream so that static meshes, which are the
 *  overwhelming majority, do not carry 16 bytes of unused joint indices per
 *  vertex through the vertex fetch. */
class GEGXMMeshCache : public irr::scene::CMeshCache
{
private:
    GEGXMDriver* m_driver;

    uint64_t m_irrlicht_cache_time, m_ge_cache_time;

    /** One allocation holding, in order: static vertices, indices, skinning
     *  vertices. */
    uint8_t* m_buffer;

    size_t m_buffer_size;

    size_t m_index_offset;

    /** Where the skinning stream starts, already biased so that a mesh
     *  buffer's vertex offset indexes it directly. See updateCache(). */
    size_t m_skinning_offset;

public:
    // ------------------------------------------------------------------------
    GEGXMMeshCache();
    // ------------------------------------------------------------------------
    virtual ~GEGXMMeshCache();
    // ------------------------------------------------------------------------
    virtual void meshCacheChanged();
    // ------------------------------------------------------------------------
    /** Rebuilds the packed buffers if any mesh has been added or removed since
     *  the last call. Cheap when nothing changed, which is every frame during
     *  a race. */
    void updateCache();
    // ------------------------------------------------------------------------
    void destroy();
    // ------------------------------------------------------------------------
    uint8_t* getBuffer() const                             { return m_buffer; }
    // ------------------------------------------------------------------------
    /** Address of the static vertex stream for a mesh buffer at \p vbo_offset
     *  vertices into the packed buffer. */
    void* getVertexStream(size_t vbo_offset) const;
    // ------------------------------------------------------------------------
    void* getSkinningStream(size_t vbo_offset) const;
    // ------------------------------------------------------------------------
    void* getIndexStream(size_t ibo_offset) const;
    // ------------------------------------------------------------------------
    bool isValid() const                          { return m_buffer != NULL; }
};   // GEGXMMeshCache

}   // namespace GE

#endif   // _IRR_COMPILE_WITH_GXM_

#endif
