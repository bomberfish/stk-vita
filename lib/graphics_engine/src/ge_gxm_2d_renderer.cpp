#include "ge_gxm_2d_renderer.hpp"

#ifdef _IRR_COMPILE_WITH_GXM_

#include "ge_gxm_driver.hpp"
#include "ge_gxm_limits.hpp"
#include "ge_gxm_memory.hpp"
#include "ge_gxm_render_target.hpp"
#include "ge_gxm_shader.hpp"
#include "ge_gxm_texture.hpp"

#include "../source/Irrlicht/os.h"

#include <algorithm>
#include <cstring>
#include <vector>

namespace GE
{
namespace GEGXM2dRenderer
{
namespace
{
/** 20 bytes: position, packed colour, texture coordinate. Deliberately smaller
 *  than irrlicht's S3DVertex, which carries a normal and a 3D position the 2D
 *  path has no use for - at several thousand quads a frame the difference is
 *  real bandwidth. */
struct Vertex2D
{
    float m_position[2];
    uint8_t m_color[4];
    float m_uv[2];
};

struct Batch
{
    const irr::video::ITexture* m_texture;
    unsigned m_index_start;
    unsigned m_index_count;
};

GEGXMDriver* g_driver = NULL;

GEGXMProgram* g_vertex_program = NULL;
GEGXMProgram* g_fragment_program = NULL;
SceGxmVertexProgram* g_patched_vertex = NULL;
SceGxmFragmentProgram* g_patched_fragment = NULL;

std::vector<Vertex2D> g_vertices;
std::vector<uint16_t> g_indices;
std::vector<Batch> g_batches;

bool g_ready = false;

// ----------------------------------------------------------------------------
void startBatch(const irr::video::ITexture* texture)
{
    if (!g_batches.empty() && g_batches.back().m_texture == texture)
        return;
    g_batches.push_back({ texture, (unsigned)g_indices.size(), 0 });
}   // startBatch

// ----------------------------------------------------------------------------
const SceGxmTexture* resolveTexture(const irr::video::ITexture* texture)
{
    if (texture == NULL)
        texture = g_driver->getWhiteTexture();
    const GEGXMTexture* gxm = dynamic_cast<const GEGXMTexture*>(texture);
    if (gxm != NULL && gxm->getGXMTexture() != NULL)
        return gxm->getGXMTexture();
    const GEGXMFBOTexture* fbo = dynamic_cast<const GEGXMFBOTexture*>(texture);
    if (fbo != NULL && fbo->getGXMTexture() != NULL)
        return fbo->getGXMTexture();
    // A texture that failed to upload still has to draw something, or the
    // widget silently vanishes.
    const GEGXMTexture* white =
        static_cast<const GEGXMTexture*>(g_driver->getWhiteTexture());
    return white == NULL ? NULL : white->getGXMTexture();
}   // resolveTexture
}   // namespace

// ----------------------------------------------------------------------------
void init(GEGXMDriver* driver)
{
    g_driver = driver;
    g_ready = false;

    g_vertex_program = GEGXMShaderManager::getVertexProgram("2d.vert");
    g_fragment_program = GEGXMShaderManager::getFragmentProgram("2d.frag");
    if (g_vertex_program == NULL || g_fragment_program == NULL)
    {
        irr::os::Printer::log("GXM: could not build the 2D programs",
            irr::ELL_ERROR);
        return;
    }

    SceGxmVertexStream stream;
    stream.stride = sizeof(Vertex2D);
    stream.indexSource = SCE_GXM_INDEX_SOURCE_INDEX_16BIT;
    const GEGXMAttributeDesc attributes[] =
    {
        { "a_position", 0, offsetof(Vertex2D, m_position),
          SCE_GXM_ATTRIBUTE_FORMAT_F32, 2 },
        { "a_color", 0, offsetof(Vertex2D, m_color),
          SCE_GXM_ATTRIBUTE_FORMAT_U8N, 4 },
        { "a_uv", 0, offsetof(Vertex2D, m_uv),
          SCE_GXM_ATTRIBUTE_FORMAT_F32, 2 },
    };
    GEGXMVertexLayout layout;
    if (!buildVertexLayout(g_vertex_program, attributes, 3, &stream, 1,
        &layout))
        return;

    g_patched_vertex = driver->getProgramCache()->getVertexProgram(
        g_vertex_program, layout);
    // The GUI is drawn with straight alpha blending. Note that the blend
    // equation is baked into the patched fragment program on this hardware
    // rather than being dynamic state, which is why there is one program per
    // blend mode instead of a setBlend call.
    g_patched_fragment = driver->getProgramCache()->getFragmentProgram(
        g_fragment_program, g_vertex_program, GEGXMBlendState::alphaBlend());
    if (g_patched_vertex == NULL || g_patched_fragment == NULL)
        return;

    g_ready = true;
}   // init

// ----------------------------------------------------------------------------
void destroy()
{
    clear();
    g_vertices.clear();
    g_vertices.shrink_to_fit();
    g_indices.clear();
    g_indices.shrink_to_fit();
    g_batches.clear();
    g_batches.shrink_to_fit();
    g_patched_vertex = NULL;
    g_patched_fragment = NULL;
    g_vertex_program = NULL;
    g_fragment_program = NULL;
    g_driver = NULL;
    g_ready = false;
}   // destroy

// ----------------------------------------------------------------------------
void clear()
{
    g_vertices.clear();
    g_indices.clear();
    g_batches.clear();
}   // clear

// ----------------------------------------------------------------------------
void handleDeletedTexture(const irr::video::ITexture* texture)
{
    // Anything already batched against this texture would dereference freed
    // memory at render time, so the safe move is to redirect it to the
    // placeholder rather than to drop the geometry and leave a hole.
    for (Batch& b : g_batches)
    {
        if (b.m_texture == texture)
            b.m_texture = NULL;
    }
}   // handleDeletedTexture

// ----------------------------------------------------------------------------
void addQuad(const irr::video::ITexture* texture,
             const irr::core::rect<irr::s32>& dest_rect,
             const irr::core::rect<irr::s32>& source_rect,
             const irr::core::rect<irr::s32>* clip_rect,
             const irr::video::SColor* colors)
{
    if (!g_ready || texture == NULL)
        return;

    irr::core::rect<irr::s32> dest = dest_rect;
    irr::core::rect<irr::s32> source = source_rect;

    if (clip_rect != NULL)
    {
        // Trim the quad instead of using a scissor rectangle: a scissor change
        // is a state change and would split the batch, and STK clips almost
        // every widget it draws.
        if (!clip_rect->isValid())
            return;
        const float inv_w = 1.0f / std::max(1, dest.getWidth());
        const float inv_h = 1.0f / std::max(1, dest.getHeight());
        const float source_w = (float)source.getWidth();
        const float source_h = (float)source.getHeight();

        int left = std::max(dest.UpperLeftCorner.X,
            clip_rect->UpperLeftCorner.X);
        int top = std::max(dest.UpperLeftCorner.Y,
            clip_rect->UpperLeftCorner.Y);
        int right = std::min(dest.LowerRightCorner.X,
            clip_rect->LowerRightCorner.X);
        int bottom = std::min(dest.LowerRightCorner.Y,
            clip_rect->LowerRightCorner.Y);
        if (right <= left || bottom <= top)
            return;

        const float u0 = (left - dest.UpperLeftCorner.X) * inv_w;
        const float u1 = (right - dest.UpperLeftCorner.X) * inv_w;
        const float v0 = (top - dest.UpperLeftCorner.Y) * inv_h;
        const float v1 = (bottom - dest.UpperLeftCorner.Y) * inv_h;
        const int sx = source.UpperLeftCorner.X;
        const int sy = source.UpperLeftCorner.Y;
        source.UpperLeftCorner.X = sx + (int)(u0 * source_w);
        source.LowerRightCorner.X = sx + (int)(u1 * source_w);
        source.UpperLeftCorner.Y = sy + (int)(v0 * source_h);
        source.LowerRightCorner.Y = sy + (int)(v1 * source_h);
        dest = irr::core::rect<irr::s32>(left, top, right, bottom);
    }

    const irr::core::dimension2du& size = texture->getSize();
    if (size.Width == 0 || size.Height == 0)
        return;
    const float tw = 1.0f / (float)size.Width;
    const float th = 1.0f / (float)size.Height;

    startBatch(texture);

    const unsigned base = (unsigned)g_vertices.size();
    // SCE_GXM_INDEX_SOURCE_INDEX_16BIT is documented as requiring index values
    // below 64000, not below 65536. Running out means this frame's GUI is
    // pathological, so stop adding rather than emit an out of range index.
    if (base + 4 > 64000u)
        return;

    const irr::video::SColor white(255, 255, 255, 255);
    // irrlicht's corner order for draw2DImage is upper left, lower left, lower
    // right, upper right.
    const irr::video::SColor c[4] =
    {
        colors == NULL ? white : colors[0],
        colors == NULL ? white : colors[1],
        colors == NULL ? white : colors[2],
        colors == NULL ? white : colors[3]
    };
    const float px[4] =
    {
        (float)dest.UpperLeftCorner.X, (float)dest.UpperLeftCorner.X,
        (float)dest.LowerRightCorner.X, (float)dest.LowerRightCorner.X
    };
    const float py[4] =
    {
        (float)dest.UpperLeftCorner.Y, (float)dest.LowerRightCorner.Y,
        (float)dest.LowerRightCorner.Y, (float)dest.UpperLeftCorner.Y
    };
    const float u[4] =
    {
        source.UpperLeftCorner.X * tw, source.UpperLeftCorner.X * tw,
        source.LowerRightCorner.X * tw, source.LowerRightCorner.X * tw
    };
    const float v[4] =
    {
        source.UpperLeftCorner.Y * th, source.LowerRightCorner.Y * th,
        source.LowerRightCorner.Y * th, source.UpperLeftCorner.Y * th
    };

    for (unsigned i = 0; i < 4; i++)
    {
        Vertex2D vert;
        vert.m_position[0] = px[i];
        vert.m_position[1] = py[i];
        // The attribute is read as four normalised bytes in memory order, so
        // the components have to be written R, G, B, A rather than as
        // irrlicht's 0xAARRGGBB word.
        vert.m_color[0] = c[i].getRed();
        vert.m_color[1] = c[i].getGreen();
        vert.m_color[2] = c[i].getBlue();
        vert.m_color[3] = c[i].getAlpha();
        vert.m_uv[0] = u[i];
        vert.m_uv[1] = v[i];
        g_vertices.push_back(vert);
    }

    const uint16_t quad_indices[6] = { 0, 1, 2, 0, 2, 3 };
    for (unsigned i = 0; i < 6; i++)
        g_indices.push_back((uint16_t)(base + quad_indices[i]));
    g_batches.back().m_index_count += 6;
}   // addQuad

// ----------------------------------------------------------------------------
void addVerticesIndices(irr::video::S3DVertex* vertices,
                        unsigned vertices_count, uint16_t* indices,
                        unsigned indices_count,
                        const irr::video::ITexture* texture)
{
    if (!g_ready || vertices == NULL || indices == NULL ||
        vertices_count == 0 || indices_count == 0)
        return;

    startBatch(texture);
    const unsigned base = (unsigned)g_vertices.size();
    if (base + vertices_count > 64000u)
        return;

    for (unsigned i = 0; i < vertices_count; i++)
    {
        Vertex2D vert;
        vert.m_position[0] = vertices[i].Pos.X;
        vert.m_position[1] = vertices[i].Pos.Y;
        vert.m_color[0] = vertices[i].Color.getRed();
        vert.m_color[1] = vertices[i].Color.getGreen();
        vert.m_color[2] = vertices[i].Color.getBlue();
        vert.m_color[3] = vertices[i].Color.getAlpha();
        vert.m_uv[0] = vertices[i].TCoords.X;
        vert.m_uv[1] = vertices[i].TCoords.Y;
        g_vertices.push_back(vert);
    }
    for (unsigned i = 0; i < indices_count; i++)
        g_indices.push_back((uint16_t)(base + indices[i]));
    g_batches.back().m_index_count += indices_count;
}   // addVerticesIndices

// ----------------------------------------------------------------------------
void render()
{
    if (!g_ready || g_batches.empty() || g_indices.empty())
        return;
    SceGxmContext* ctx = g_driver->getContext();
    if (ctx == NULL || !g_driver->isInScene())
        return;

    // The geometry has to live in memory the GPU can read until this frame has
    // been displayed, which is what the per frame arena is for; a std::vector
    // would be reused before the GPU had finished with it.
    const size_t vertices_size = g_vertices.size() * sizeof(Vertex2D);
    const size_t indices_size = g_indices.size() * sizeof(uint16_t);
    void* gpu_vertices = g_driver->allocateFrameMemory(vertices_size, 16);
    void* gpu_indices = g_driver->allocateFrameMemory(indices_size, 16);
    if (gpu_vertices == NULL || gpu_indices == NULL)
        return;
    memcpy(gpu_vertices, g_vertices.data(), vertices_size);
    memcpy(gpu_indices, g_indices.data(), indices_size);

    const irr::core::dimension2du& target =
        g_driver->getCurrentRenderTargetSize();

    // The GUI owns the whole target and must not be depth tested against the
    // 3D scene it is drawn over.
    sceGxmSetFrontDepthFunc(ctx, SCE_GXM_DEPTH_FUNC_ALWAYS);
    sceGxmSetFrontDepthWriteEnable(ctx, SCE_GXM_DEPTH_WRITE_DISABLED);
    sceGxmSetCullMode(ctx, SCE_GXM_CULL_NONE);
    g_driver->applyViewport(irr::core::rect<irr::s32>(0, 0, target.Width,
        target.Height));

    sceGxmSetVertexProgram(ctx, g_patched_vertex);
    sceGxmSetFragmentProgram(ctx, g_patched_fragment);

    // u_screen turns pixel coordinates into clip space in the vertex shader,
    // so no projection matrix is needed and it only has to be set once.
    void* vertex_uniforms = NULL;
    if (sceGxmReserveVertexDefaultUniformBuffer(ctx, &vertex_uniforms) >= 0 &&
        vertex_uniforms != NULL)
    {
        const SceGxmProgramParameter* p =
            g_vertex_program->getParameter("u_screen");
        if (p != NULL)
        {
            const float screen[4] =
            {
                2.0f / (float)std::max(1u, target.Width),
                2.0f / (float)std::max(1u, target.Height),
                0.0f, 0.0f
            };
            sceGxmSetUniformDataF(vertex_uniforms, p, 0, 4, screen);
        }
    }

    sceGxmSetVertexStream(ctx, 0, gpu_vertices);

    const SceGxmTexture* previous = NULL;
    for (const Batch& b : g_batches)
    {
        if (b.m_index_count == 0)
            continue;
        const SceGxmTexture* tex = resolveTexture(b.m_texture);
        if (tex == NULL)
            continue;
        if (tex != previous)
        {
            sceGxmSetFragmentTexture(ctx, GTU_MATERIAL_0, tex);
            previous = tex;
        }
        sceGxmDraw(ctx, SCE_GXM_PRIMITIVE_TRIANGLES, SCE_GXM_INDEX_FORMAT_U16,
            (const uint16_t*)gpu_indices + b.m_index_start, b.m_index_count);
    }

    sceGxmSetFrontDepthFunc(ctx, SCE_GXM_DEPTH_FUNC_LESS_EQUAL);
    sceGxmSetFrontDepthWriteEnable(ctx, SCE_GXM_DEPTH_WRITE_ENABLED);
}   // render

}   // namespace GEGXM2dRenderer
}   // namespace GE

#endif   // _IRR_COMPILE_WITH_GXM_
