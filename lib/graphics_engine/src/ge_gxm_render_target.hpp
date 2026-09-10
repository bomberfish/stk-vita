#ifndef HEADER_GE_GXM_RENDER_TARGET_HPP
#define HEADER_GE_GXM_RENDER_TARGET_HPP

#include "IrrCompileConfig.h"

#ifdef _IRR_COMPILE_WITH_GXM_

#include <psp2/gxm.h>

#include <cstdint>
#include <string>

#include <ITexture.h>
#include <dimension2d.h>

namespace GE
{

/** A colour surface, an optional depth/stencil surface and the SceGxmRenderTarget
 *  that ties them together.
 *
 *  A SceGxmRenderTarget is not the surfaces: it is the tiling state the GPU
 *  needs for a given width and height, and it is expensive enough to create
 *  that it wants to be made once and reused. The surfaces are separate, which
 *  is what lets the three display buffers share one render target.
 *
 *  Note that there is exactly one colour surface. The Vita's tile accelerator
 *  has no multiple render target support at all, which is the single fact that
 *  shapes this whole renderer: a deferred G-buffer pass is not expressible, so
 *  the GXM pipeline is forward. */
class GEGXMSurface
{
private:
    SceGxmRenderTarget* m_render_target;

    SceGxmColorSurface m_color_surface;

    SceGxmDepthStencilSurface m_depth_surface;

    void* m_color_data;

    void* m_depth_data;

    /** Set when the render target belongs to someone else and must not be
     *  destroyed here. */
    bool m_shared_render_target;

    /** Set when the depth surface belongs to someone else and must not be
     *  freed here. The main scene and every render target texture share one
     *  depth buffer, because only one scene is ever in flight. */
    bool m_shared_depth;

    irr::core::dimension2d<irr::u32> m_size;

    /** Colour row pitch in bytes. */
    unsigned m_stride;

    /** Depth row pitch in samples. Recorded rather than re-derived, because a
     *  surface sharing depth has to address it with the pitch of whichever
     *  surface allocated the memory, not with its own width. */
    unsigned m_depth_stride;

    SceGxmColorFormat m_color_format;

    bool m_has_depth;

public:
    // ------------------------------------------------------------------------
    GEGXMSurface();
    // ------------------------------------------------------------------------
    ~GEGXMSurface();
    // ------------------------------------------------------------------------
    /** \param color_data when non NULL the surface renders into memory the
     *         caller owns, which is how the display buffers work.
     *  \param shared_depth when non NULL, reuse that depth surface's memory
     *         instead of allocating one.
     *  \param stride_in_pixels row pitch of \p color_data. Must be passed
     *         whenever the caller allocated the memory, because the pitch it
     *         used is not derivable from the width - the display buffers are
     *         960 wide but 1024 pitched. 0 derives a pitch from the width, which
     *         is only correct for surfaces this class allocates itself.
     *  \param shared_render_target reuse an existing render target instead of
     *         creating one; the tiling state only depends on the dimensions, so
     *         the three display buffers share one. */
    bool init(unsigned width, unsigned height, SceGxmColorFormat format,
              void* color_data = NULL, bool with_depth = true,
              GEGXMSurface* shared_depth = NULL,
              unsigned stride_in_pixels = 0,
              SceGxmRenderTarget* shared_render_target = NULL);
    // ------------------------------------------------------------------------
    /** Depth only variant with the colour surface disabled, for the shadow
     *  caster pass. Storing no colour is a large saving on a tile based GPU:
     *  the tile buffer is never written out to memory at all. */
    bool initDepthOnly(unsigned width, unsigned height, bool sample_depth);
    // ------------------------------------------------------------------------
    void destroy();
    // ------------------------------------------------------------------------
    SceGxmRenderTarget* getRenderTarget() const     { return m_render_target; }
    // ------------------------------------------------------------------------
    SceGxmColorSurface* getColorSurface()          { return &m_color_surface; }
    // ------------------------------------------------------------------------
    SceGxmDepthStencilSurface* getDepthSurface()   { return &m_depth_surface; }
    // ------------------------------------------------------------------------
    void* getColorData() const                         { return m_color_data; }
    // ------------------------------------------------------------------------
    void* getDepthData() const                         { return m_depth_data; }
    // ------------------------------------------------------------------------
    const irr::core::dimension2d<irr::u32>& getSize() const  { return m_size; }
    // ------------------------------------------------------------------------
    unsigned getStride() const                             { return m_stride; }
    // ------------------------------------------------------------------------
    SceGxmColorFormat getColorFormat() const         { return m_color_format; }
    // ------------------------------------------------------------------------
    bool hasDepth() const                               { return m_has_depth; }
    // ------------------------------------------------------------------------
    /** Whether the depth buffer's contents survive to the end of the scene.
     *  Off by default: a forward renderer only needs depth within a scene, and
     *  not storing it saves the whole write out. Turned on for the shadow
     *  cascade, which is sampled afterwards. */
    void setStoreDepth(bool store);
    // ------------------------------------------------------------------------
    /** Turns on the linear to sRGB conversion the render output does on write.
     *  Used for the HDR scene target so that the 8 bit intermediate keeps its
     *  precision where the eye is sensitive. */
    void setSRGBWrite(bool srgb);
};   // GEGXMSurface

/** A render target that STK can also sample, i.e. an irrlicht RTT.
 *
 *  Used for the kart previews in the menus and by anything that goes through
 *  IrrDriver::createRenderTarget(). Kept linear rather than swizzled because a
 *  colour surface has to be linear, and a resolve into a swizzled copy would
 *  cost more than the better filtering is worth for something drawn as a
 *  screen aligned quad. */
class GEGXMFBOTexture : public irr::video::ITexture
{
private:
    GEGXMSurface m_surface;

    SceGxmTexture m_texture;

    irr::core::dimension2d<irr::u32> m_size;

    bool m_valid;

public:
    // ------------------------------------------------------------------------
    GEGXMFBOTexture(const std::string& name, unsigned width, unsigned height,
                    GEGXMSurface* shared_depth);
    // ------------------------------------------------------------------------
    virtual ~GEGXMFBOTexture();
    // ------------------------------------------------------------------------
    virtual void* lock(irr::video::E_TEXTURE_LOCK_MODE mode =
                       irr::video::ETLM_READ_WRITE, irr::u32 mipmap_level = 0)
                                                              { return NULL; }
    // ------------------------------------------------------------------------
    virtual void unlock()                                                    {}
    // ------------------------------------------------------------------------
    virtual const irr::core::dimension2d<irr::u32>& getOriginalSize() const
                                                             { return m_size; }
    // ------------------------------------------------------------------------
    virtual const irr::core::dimension2d<irr::u32>& getSize() const
                                                             { return m_size; }
    // ------------------------------------------------------------------------
    virtual irr::video::E_DRIVER_TYPE getDriverType() const
                                               { return irr::video::EDT_GXM; }
    // ------------------------------------------------------------------------
    virtual irr::video::ECOLOR_FORMAT getColorFormat() const
                                          { return irr::video::ECF_A8R8G8B8; }
    // ------------------------------------------------------------------------
    virtual irr::u32 getPitch() const        { return m_surface.getStride(); }
    // ------------------------------------------------------------------------
    virtual bool hasMipMaps() const                          { return false; }
    // ------------------------------------------------------------------------
    virtual void regenerateMipMapLevels(void* mipmap_data = NULL)            {}
    // ------------------------------------------------------------------------
    virtual bool isRenderTarget() const                       { return true; }
    // ------------------------------------------------------------------------
    virtual irr::u64 getTextureHandler() const
                                          { return (irr::u64)(uintptr_t)this; }
    // ------------------------------------------------------------------------
    GEGXMSurface* getSurface()                          { return &m_surface; }
    // ------------------------------------------------------------------------
    const SceGxmTexture* getGXMTexture() const
                                  { return m_valid ? &m_texture : NULL; }
    // ------------------------------------------------------------------------
    bool isValid() const                                     { return m_valid; }
};   // GEGXMFBOTexture

}   // namespace GE

#endif   // _IRR_COMPILE_WITH_GXM_

#endif
