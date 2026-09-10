#include "ge_gxm_render_target.hpp"

#ifdef _IRR_COMPILE_WITH_GXM_

#include "ge_gxm_driver.hpp"
#include "ge_gxm_memory.hpp"
#include "ge_main.hpp"

#include "../source/Irrlicht/os.h"

#include <cstring>

namespace GE
{
namespace
{
// ----------------------------------------------------------------------------
inline unsigned alignUp(unsigned value, unsigned alignment)
{
    return (value + alignment - 1) & ~(alignment - 1);
}   // alignUp
}   // namespace

// ============================================================================
GEGXMSurface::GEGXMSurface()
{
    m_render_target = NULL;
    m_shared_render_target = false;
    m_color_data = NULL;
    m_depth_data = NULL;
    m_shared_depth = false;
    m_stride = 0;
    m_depth_stride = 0;
    m_color_format = SCE_GXM_COLOR_FORMAT_A8B8G8R8;
    m_has_depth = false;
    memset(&m_color_surface, 0, sizeof(m_color_surface));
    memset(&m_depth_surface, 0, sizeof(m_depth_surface));
}   // GEGXMSurface

// ----------------------------------------------------------------------------
GEGXMSurface::~GEGXMSurface()
{
    destroy();
}   // ~GEGXMSurface

// ----------------------------------------------------------------------------
void GEGXMSurface::destroy()
{
    if (m_render_target != NULL && !m_shared_render_target)
        sceGxmDestroyRenderTarget(m_render_target);
    m_render_target = NULL;
    m_shared_render_target = false;
    if (m_color_data != NULL)
    {
        GEGXMMemory::free(m_color_data);
        m_color_data = NULL;
    }
    if (m_depth_data != NULL && !m_shared_depth)
        GEGXMMemory::free(m_depth_data);
    m_depth_data = NULL;
    m_shared_depth = false;
    m_depth_stride = 0;
    m_has_depth = false;
}   // destroy

// ----------------------------------------------------------------------------
bool GEGXMSurface::init(unsigned width, unsigned height,
                        SceGxmColorFormat format, void* color_data,
                        bool with_depth, GEGXMSurface* shared_depth,
                        unsigned stride_in_pixels,
                        SceGxmRenderTarget* shared_render_target)
{
    destroy();
    m_size = irr::core::dimension2d<irr::u32>(width, height);
    m_color_format = format;
    // The pitch has to be whatever the owner of the memory used. Deriving it
    // from the width is only safe for memory this function allocates: the
    // display buffers are 960 wide but 1024 pitched, and a mismatch here writes
    // every scanline at the wrong offset, which shows up as a diagonal smear
    // rather than as an error.
    m_stride = (stride_in_pixels != 0 ? stride_in_pixels :
        alignUp(width, SCE_GXM_TILE_SIZEX)) * 4;

    if (shared_render_target != NULL)
    {
        m_render_target = shared_render_target;
        m_shared_render_target = true;
    }
    else
    {
        SceGxmRenderTargetParams params;
        memset(&params, 0, sizeof(params));
        params.flags = 0;
        params.width = (uint16_t)width;
        params.height = (uint16_t)height;
        // The post processing chain begins two scenes per frame on the same
        // bloom target, and this is the hint sceGxm sizes its internal state
        // from.
        params.scenesPerFrame = 4;
        params.multisampleMode = SCE_GXM_MULTISAMPLE_NONE;
        params.multisampleLocations = 0;
        // Letting sceGxm allocate the tiling structures itself keeps them out of
        // our heaps, where they would be an odd sized permanent allocation.
        params.driverMemBlock = -1;
        if (sceGxmCreateRenderTarget(&params, &m_render_target) < 0)
        {
            irr::os::Printer::log("GXM: sceGxmCreateRenderTarget failed",
                irr::ELL_ERROR);
            return false;
        }
    }

    void* color = color_data;
    if (color == NULL)
    {
        color = GEGXMMemory::allocate(GGMP_CDRAM,
            (size_t)m_stride * alignUp(height, SCE_GXM_TILE_SIZEY),
            // Texture alignment, not colour surface alignment (which is only
            // 4): GEGXMFBOTexture binds this same memory through
            // sceGxmTextureInitLinearStrided.
            SCE_GXM_TEXTURE_ALIGNMENT);
        if (color == NULL)
        {
            irr::os::Printer::log("GXM: out of memory for a colour surface",
                irr::ELL_ERROR);
            destroy();
            return false;
        }
        m_color_data = color;
    }

    if (sceGxmColorSurfaceInit(&m_color_surface, format,
        SCE_GXM_COLOR_SURFACE_LINEAR, SCE_GXM_COLOR_SURFACE_SCALE_NONE,
        SCE_GXM_OUTPUT_REGISTER_SIZE_32BIT, width, height, m_stride / 4,
        color) < 0)
    {
        irr::os::Printer::log("GXM: sceGxmColorSurfaceInit failed",
            irr::ELL_ERROR);
        destroy();
        return false;
    }

    if (!with_depth)
    {
        sceGxmDepthStencilSurfaceInitDisabled(&m_depth_surface);
        return true;
    }

    const unsigned depth_stride = alignUp(width, SCE_GXM_TILE_SIZEX);
    if (shared_depth != NULL && shared_depth->m_depth_data != NULL &&
        shared_depth->getSize().Width >= width &&
        shared_depth->getSize().Height >= height)
    {
        // Reuse an existing, larger depth buffer. It is addressed with the
        // owner's stride, which the surface has to be told about, otherwise
        // rows would be read at the wrong pitch.
        m_depth_data = shared_depth->m_depth_data;
        m_shared_depth = true;
        // The donor's recorded pitch, not one recomputed from its width: the
        // donor may itself be sharing from a third, wider surface, in which case
        // its width says nothing about how the memory is strided.
        m_depth_stride = shared_depth->m_depth_stride;
        if (sceGxmDepthStencilSurfaceInit(&m_depth_surface,
            SCE_GXM_DEPTH_STENCIL_FORMAT_D16,
            SCE_GXM_DEPTH_STENCIL_SURFACE_LINEAR, m_depth_stride,
            m_depth_data, NULL) < 0)
        {
            irr::os::Printer::log("GXM: sceGxmDepthStencilSurfaceInit failed",
                irr::ELL_ERROR);
            destroy();
            return false;
        }
    }
    else
    {
        // D16 rather than DF32: half the bandwidth, and 16 bits is enough for
        // STK's near/far range once the projection is set up. The shadow
        // cascade uses DF32M instead, see initDepthOnly().
        m_depth_data = GEGXMMemory::allocate(GGMP_CDRAM,
            (size_t)depth_stride * alignUp(height, SCE_GXM_TILE_SIZEY) * 2,
            SCE_GXM_DEPTHSTENCIL_SURFACE_ALIGNMENT);
        if (m_depth_data == NULL)
        {
            irr::os::Printer::log("GXM: out of memory for a depth surface",
                irr::ELL_ERROR);
            destroy();
            return false;
        }
        m_depth_stride = depth_stride;
        if (sceGxmDepthStencilSurfaceInit(&m_depth_surface,
            SCE_GXM_DEPTH_STENCIL_FORMAT_D16,
            SCE_GXM_DEPTH_STENCIL_SURFACE_LINEAR, depth_stride, m_depth_data,
            NULL) < 0)
        {
            irr::os::Printer::log("GXM: sceGxmDepthStencilSurfaceInit failed",
                irr::ELL_ERROR);
            destroy();
            return false;
        }
    }
    m_has_depth = true;
    sceGxmDepthStencilSurfaceSetBackgroundDepth(&m_depth_surface, 1.0f);
    // Not loading and not storing depth is the whole point of a tile based GPU:
    // the tile starts at the background value and is thrown away at the end of
    // the scene, so the depth buffer never touches memory.
    sceGxmDepthStencilSurfaceSetForceLoadMode(&m_depth_surface,
        SCE_GXM_DEPTH_STENCIL_FORCE_LOAD_DISABLED);
    sceGxmDepthStencilSurfaceSetForceStoreMode(&m_depth_surface,
        SCE_GXM_DEPTH_STENCIL_FORCE_STORE_DISABLED);
    return true;
}   // init

// ----------------------------------------------------------------------------
bool GEGXMSurface::initDepthOnly(unsigned width, unsigned height,
                                 bool sample_depth)
{
    destroy();
    m_size = irr::core::dimension2d<irr::u32>(width, height);
    m_stride = 0;

    SceGxmRenderTargetParams params;
    memset(&params, 0, sizeof(params));
    params.width = (uint16_t)width;
    params.height = (uint16_t)height;
    params.scenesPerFrame = 1;
    params.multisampleMode = SCE_GXM_MULTISAMPLE_NONE;
    params.driverMemBlock = -1;
    if (sceGxmCreateRenderTarget(&params, &m_render_target) < 0)
    {
        irr::os::Printer::log("GXM: sceGxmCreateRenderTarget failed for the "
            "shadow cascade", irr::ELL_ERROR);
        return false;
    }

    sceGxmColorSurfaceInitDisabled(&m_color_surface);

    const unsigned depth_stride = alignUp(width, SCE_GXM_TILE_SIZEX);
    // DF32M is the format that can be read back as a texture
    // (SCE_GXM_TEXTURE_FORMAT_F32M_R), which is what makes a sampled shadow
    // map possible; one bit is reserved for the tile mask, which costs nothing
    // here.
    m_depth_data = GEGXMMemory::allocate(GGMP_CDRAM,
        (size_t)depth_stride * alignUp(height, SCE_GXM_TILE_SIZEY) * 4,
        SCE_GXM_DEPTHSTENCIL_SURFACE_ALIGNMENT);
    if (m_depth_data == NULL)
    {
        irr::os::Printer::log("GXM: out of memory for the shadow cascade",
            irr::ELL_ERROR);
        destroy();
        return false;
    }
    if (sceGxmDepthStencilSurfaceInit(&m_depth_surface,
        SCE_GXM_DEPTH_STENCIL_FORMAT_DF32M,
        SCE_GXM_DEPTH_STENCIL_SURFACE_LINEAR, depth_stride, m_depth_data,
        NULL) < 0)
    {
        irr::os::Printer::log("GXM: sceGxmDepthStencilSurfaceInit failed for "
            "the shadow cascade", irr::ELL_ERROR);
        destroy();
        return false;
    }
    m_has_depth = true;
    m_depth_stride = depth_stride;
    sceGxmDepthStencilSurfaceSetBackgroundDepth(&m_depth_surface, 1.0f);
    sceGxmDepthStencilSurfaceSetForceLoadMode(&m_depth_surface,
        SCE_GXM_DEPTH_STENCIL_FORCE_LOAD_DISABLED);
    sceGxmDepthStencilSurfaceSetForceStoreMode(&m_depth_surface,
        sample_depth ? SCE_GXM_DEPTH_STENCIL_FORCE_STORE_ENABLED :
        SCE_GXM_DEPTH_STENCIL_FORCE_STORE_DISABLED);
    return true;
}   // initDepthOnly

// ----------------------------------------------------------------------------
void GEGXMSurface::setStoreDepth(bool store)
{
    if (!m_has_depth)
        return;
    sceGxmDepthStencilSurfaceSetForceStoreMode(&m_depth_surface, store ?
        SCE_GXM_DEPTH_STENCIL_FORCE_STORE_ENABLED :
        SCE_GXM_DEPTH_STENCIL_FORCE_STORE_DISABLED);
}   // setStoreDepth

// ----------------------------------------------------------------------------
void GEGXMSurface::setSRGBWrite(bool srgb)
{
    sceGxmColorSurfaceSetGammaMode(&m_color_surface, srgb ?
        SCE_GXM_COLOR_SURFACE_GAMMA_BGR : SCE_GXM_COLOR_SURFACE_GAMMA_NONE);
}   // setSRGBWrite

// ============================================================================
GEGXMFBOTexture::GEGXMFBOTexture(const std::string& name, unsigned width,
                                 unsigned height, GEGXMSurface* shared_depth)
               : irr::video::ITexture(name.c_str())
{
    m_valid = false;
    memset(&m_texture, 0, sizeof(m_texture));
    m_size = irr::core::dimension2d<irr::u32>(width, height);

    if (!m_surface.init(width, height, SCE_GXM_COLOR_FORMAT_A8B8G8R8, NULL,
        true, shared_depth))
    {
        LoadingFailed = true;
        return;
    }
    // The same memory is both the render output and a sampled texture. A
    // colour surface is always linear, so the sampled view has to be linear
    // strided too, which rules out mip maps - acceptable for something drawn
    // as a screen aligned quad.
    if (sceGxmTextureInitLinearStrided(&m_texture, m_surface.getColorData(),
        SCE_GXM_TEXTURE_FORMAT_U8U8U8U8_ABGR, width, height,
        m_surface.getStride()) < 0)
    {
        irr::os::Printer::log("GXM: could not create a sampled view of a "
            "render target", irr::ELL_ERROR);
        LoadingFailed = true;
        return;
    }
    sceGxmTextureSetMinFilter(&m_texture, SCE_GXM_TEXTURE_FILTER_LINEAR);
    sceGxmTextureSetMagFilter(&m_texture, SCE_GXM_TEXTURE_FILTER_LINEAR);
    sceGxmTextureSetMipFilter(&m_texture,
        SCE_GXM_TEXTURE_MIP_FILTER_DISABLED);
    sceGxmTextureSetUAddrMode(&m_texture, SCE_GXM_TEXTURE_ADDR_CLAMP);
    sceGxmTextureSetVAddrMode(&m_texture, SCE_GXM_TEXTURE_ADDR_CLAMP);
    m_valid = true;
}   // GEGXMFBOTexture

// ----------------------------------------------------------------------------
GEGXMFBOTexture::~GEGXMFBOTexture()
{
    // Same reason as GEGXMTexture::clearGXMData(): an in-flight frame may still
    // be sampling this render target, and destroy() both frees its memory and
    // destroys the render target object.
    GEGXMDriver* driver = getGXMDriver();
    if (driver != NULL)
        driver->waitIdle();
    m_surface.destroy();
}   // ~GEGXMFBOTexture

}   // namespace GE

#endif   // _IRR_COMPILE_WITH_GXM_
