#include "ge_gxm_mesh_cache.hpp"

#ifdef _IRR_COMPILE_WITH_GXM_

#include "ge_gxm_driver.hpp"
#include "ge_gxm_limits.hpp"
#include "ge_gxm_memory.hpp"
#include "ge_main.hpp"
#include "ge_spm_buffer.hpp"
#include "mini_glm.hpp"

#include "../source/Irrlicht/os.h"

#include "IAnimatedMesh.h"

#include <algorithm>
#include <cstring>
#include <vector>

namespace GE
{
namespace
{
// ----------------------------------------------------------------------------
inline int16_t toSnorm16(float v)
{
    if (v > 1.0f)
        v = 1.0f;
    else if (v < -1.0f)
        v = -1.0f;
    return (int16_t)(v * 32767.0f);
}   // toSnorm16

// ----------------------------------------------------------------------------
/** Unpacks the two-bit W of an A2B10G10R10_SNORM word.
 *
 *  For the tangent this is the bitangent handedness. The mapping is the one the
 *  Vulkan renderer's vertex fetch produces, so a mesh authored with a zero sign
 *  behaves identically on both backends rather than differing here. */
inline float unpackSnorm2(uint32_t packed)
{
    const unsigned part = (packed >> 30) & 3u;
    if (part == 1u)
        return 1.0f;
    if (part == 0u)
        return 0.0f;
    return -1.0f;
}   // unpackSnorm2

}   // namespace

// ----------------------------------------------------------------------------
void convertSPMVertex(const irr::video::S3DVertexSkinnedMesh& in,
                      GXMSPMVertex* out)
{
    out->m_position[0] = in.m_position.X;
    out->m_position[1] = in.m_position.Y;
    out->m_position[2] = in.m_position.Z;

    const irr::core::vector3df normal =
        MiniGLM::decompressVector3(in.m_normal);
    out->m_normal[0] = toSnorm16(normal.X);
    out->m_normal[1] = toSnorm16(normal.Y);
    out->m_normal[2] = toSnorm16(normal.Z);
    out->m_normal[3] = 0;

    // Stored R, G, B, A so the U8N attribute reads it in the natural order and
    // the shader needs no swizzle, unlike the Vulkan path's v_color.zyxw.
    out->m_color[0] = in.m_color.getRed();
    out->m_color[1] = in.m_color.getGreen();
    out->m_color[2] = in.m_color.getBlue();
    out->m_color[3] = in.m_color.getAlpha();

    out->m_uv[0] = in.m_all_uvs[0];
    out->m_uv[1] = in.m_all_uvs[1];
    out->m_uv_two[0] = in.m_all_uvs[2];
    out->m_uv_two[1] = in.m_all_uvs[3];

    const irr::core::vector3df tangent =
        MiniGLM::decompressVector3(in.m_tangent);
    out->m_tangent[0] = toSnorm16(tangent.X);
    out->m_tangent[1] = toSnorm16(tangent.Y);
    out->m_tangent[2] = toSnorm16(tangent.Z);
    out->m_tangent[3] = toSnorm16(unpackSnorm2(in.m_tangent));
}   // convertSPMVertex

// ============================================================================
GEGXMMeshCache::GEGXMMeshCache()
              : irr::scene::CMeshCache()
{
    m_driver = getGXMDriver();
    m_irrlicht_cache_time = getMonoTimeMs();
    m_ge_cache_time = 0;
    m_buffer = NULL;
    m_buffer_size = 0;
    m_index_offset = 0;
    m_skinning_offset = 0;
}   // GEGXMMeshCache

// ----------------------------------------------------------------------------
GEGXMMeshCache::~GEGXMMeshCache()
{
    destroy();
}   // ~GEGXMMeshCache

// ----------------------------------------------------------------------------
void GEGXMMeshCache::meshCacheChanged()
{
    m_irrlicht_cache_time = getMonoTimeMs();
}   // meshCacheChanged

// ----------------------------------------------------------------------------
void GEGXMMeshCache::destroy()
{
    if (m_buffer != NULL)
    {
        // The GPU may still be reading the old buffer from a frame that has
        // been submitted but not displayed.
        if (m_driver != NULL)
            m_driver->waitIdle();
        GEGXMMemory::free(m_buffer);
        m_buffer = NULL;
    }
    m_buffer_size = 0;
    m_index_offset = 0;
    m_skinning_offset = 0;
}   // destroy

// ----------------------------------------------------------------------------
void GEGXMMeshCache::updateCache()
{
    if (m_irrlicht_cache_time <= m_ge_cache_time)
        return;
    m_ge_cache_time = m_irrlicht_cache_time;

    std::vector<GESPMBuffer*> buffers;
    size_t vertex_count = 0;
    size_t index_count = 0;
    size_t skinned_vertex_count = 0;
    for (unsigned i = 0; i < Meshes.size(); i++)
    {
        irr::scene::IAnimatedMesh* mesh = Meshes[i].Mesh;
        if (mesh->getMeshType() != irr::scene::EAMT_SPM)
            continue;
        for (unsigned j = 0; j < mesh->getMeshBufferCount(); j++)
        {
            GESPMBuffer* mb =
                static_cast<GESPMBuffer*>(mesh->getMeshBuffer(j));
            if (mb->getVertexCount() == 0 || mb->getIndexCount() == 0)
                continue;
            vertex_count += mb->getVertexCount();
            index_count += mb->getIndexCount();
            if (mb->hasSkinning())
                skinned_vertex_count += mb->getVertexCount();
            buffers.push_back(mb);
        }
    }

    destroy();
    if (buffers.empty())
        return;

    // Static buffers first. That makes the skinned buffers a contiguous run at
    // the end, which is what lets one biased base pointer serve the skinning
    // stream for all of them.
    std::stable_partition(buffers.begin(), buffers.end(),
        [](const GESPMBuffer* mb) { return !mb->hasSkinning(); });

    const size_t vertex_size = vertex_count * sizeof(GXMSPMVertex);
    const size_t index_size = index_count * sizeof(uint16_t);
    // Vertex streams want a 4 byte aligned base and 16 is free, so pad both
    // section boundaries to 16.
    const size_t index_offset = (vertex_size + 15u) & ~(size_t)15u;
    const size_t skinning_offset =
        (index_offset + index_size + 15u) & ~(size_t)15u;
    const size_t total =
        skinning_offset + skinned_vertex_count * sizeof(GXMSPMSkinning);

    m_buffer = (uint8_t*)GEGXMMemory::allocate(GGMP_CDRAM, total, 16);
    if (m_buffer == NULL)
    {
        irr::os::Printer::log("GXM: out of video memory for the merged mesh "
            "buffer, falling back to LPDDR", irr::ELL_WARNING);
        m_buffer = (uint8_t*)GEGXMMemory::allocate(GGMP_LPDDR_UNCACHED, total,
            16);
    }
    if (m_buffer == NULL)
    {
        irr::os::Printer::log("GXM: could not allocate the merged mesh buffer;"
            " no 3D geometry will be drawn", irr::ELL_ERROR);
        return;
    }
    m_buffer_size = total;
    m_index_offset = index_offset;

    size_t vertex_cursor = 0;
    size_t index_cursor = 0;
    for (GESPMBuffer* mb : buffers)
    {
        GXMSPMVertex* dest =
            (GXMSPMVertex*)(m_buffer + vertex_cursor * sizeof(GXMSPMVertex));
        const irr::video::S3DVertexSkinnedMesh* src =
            (const irr::video::S3DVertexSkinnedMesh*)mb->getVertices();
        const unsigned count = mb->getVertexCount();
        for (unsigned v = 0; v < count; v++)
            convertSPMVertex(src[v], dest + v);
        mb->setVBOOffset(vertex_cursor);
        vertex_cursor += count;

        // Indices stay local to the mesh buffer; the base vertex is applied by
        // offsetting the stream pointer, which is how GXM does base vertex
        // rendering.
        memcpy(m_buffer + index_offset + index_cursor * sizeof(uint16_t),
            mb->getIndices(), mb->getIndexCount() * sizeof(uint16_t));
        mb->setIBOOffset(index_cursor);
        index_cursor += mb->getIndexCount();
    }

    // Bias the skinning base so that a buffer's vertex offset - which counts
    // from the very start of the packed vertex buffer, static meshes included -
    // still indexes the skinning stream correctly. Valid because the static
    // buffers were partitioned to the front, so the first skinned buffer's
    // offset is exactly the number of static vertices.
    const size_t static_vertex_count = vertex_count - skinned_vertex_count;
    m_skinning_offset = skinning_offset -
        static_vertex_count * sizeof(GXMSPMSkinning);

    size_t skinning_cursor = 0;
    for (GESPMBuffer* mb : buffers)
    {
        if (!mb->hasSkinning())
            continue;
        GXMSPMSkinning* dest = (GXMSPMSkinning*)(m_buffer + skinning_offset +
            skinning_cursor * sizeof(GXMSPMSkinning));
        const irr::video::S3DVertexSkinnedMesh* src =
            (const irr::video::S3DVertexSkinnedMesh*)mb->getVertices();
        const unsigned count = mb->getVertexCount();
        for (unsigned v = 0; v < count; v++)
        {
            memcpy(dest[v].m_joint, src[v].m_joint_idx,
                sizeof(dest[v].m_joint));
            memcpy(dest[v].m_weight, src[v].m_weight,
                sizeof(dest[v].m_weight));
        }
        skinning_cursor += count;
    }

    char buf[128];
    snprintf(buf, sizeof(buf), "GXM: packed %u mesh buffers into %.2f MB "
        "(%u vertices, %u indices)", (unsigned)buffers.size(),
        (double)total / (1024.0 * 1024.0), (unsigned)vertex_count,
        (unsigned)index_count);
    irr::os::Printer::log(buf, irr::ELL_INFORMATION);
}   // updateCache

// ----------------------------------------------------------------------------
void* GEGXMMeshCache::getVertexStream(size_t vbo_offset) const
{
    if (m_buffer == NULL)
        return NULL;
    return m_buffer + vbo_offset * sizeof(GXMSPMVertex);
}   // getVertexStream

// ----------------------------------------------------------------------------
void* GEGXMMeshCache::getSkinningStream(size_t vbo_offset) const
{
    if (m_buffer == NULL)
        return NULL;
    return m_buffer + m_skinning_offset +
        vbo_offset * sizeof(GXMSPMSkinning);
}   // getSkinningStream

// ----------------------------------------------------------------------------
void* GEGXMMeshCache::getIndexStream(size_t ibo_offset) const
{
    if (m_buffer == NULL)
        return NULL;
    return m_buffer + m_index_offset + ibo_offset * sizeof(uint16_t);
}   // getIndexStream

}   // namespace GE

#endif   // _IRR_COMPILE_WITH_GXM_
