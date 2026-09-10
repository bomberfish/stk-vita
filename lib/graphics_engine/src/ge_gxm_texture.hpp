#ifndef HEADER_GE_GXM_TEXTURE_HPP
#define HEADER_GE_GXM_TEXTURE_HPP

#include "IrrCompileConfig.h"

#ifdef _IRR_COMPILE_WITH_GXM_

#include <psp2/gxm.h>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <ITexture.h>

namespace GE
{

/** A texture living in GPU visible memory as an SceGxmTexture.
 *
 *  Layout is chosen per texture rather than fixed:
 *
 *  - Power of two textures are stored swizzled (Morton order). This is the
 *    layout the texture unit was designed around: it turns a 2D
 *    neighbourhood into a contiguous run of memory, so a bilinear tap reads
 *    one cache line instead of two. It is also the only layout that supports
 *    mip maps, which matter more here than on a desktop - undersampled
 *    textures on a 960x544 screen alias badly, and the memory bandwidth a mip
 *    chain saves is the Vita's scarcest resource.
 *  - Everything else is stored linear with an explicit stride. That covers
 *    non power of two GUI images and the font atlas, whose glyphs are written
 *    in as sub rectangles - cheap into a linear image, a scatter into a
 *    swizzled one - and which are drawn at native size so lose nothing by
 *    having no mip maps.
 */
class GEGXMTexture : public irr::video::ITexture
{
private:
    irr::core::dimension2d<irr::u32> m_size, m_orig_size;

    std::function<void(irr::video::IImage*)> m_image_mani;

    uint8_t* m_locked_data;

    SceGxmTexture m_texture;

    /** GPU memory holding every mip level, packed contiguously. */
    void* m_data;

    unsigned m_texture_size;

    unsigned m_mipmap_count;

    /** Bytes per row for the linear layout; 0 when swizzled. */
    unsigned m_stride;

    SceGxmTextureFormat m_format;

    const bool m_disable_reload;

    bool m_single_channel;

    bool m_swizzled;

    /** Set for textures that are written into after upload. They must stay
     *  linear, because a sub rectangle write into a swizzled image is a scatter
     *  rather than a row copy. Inferring this from "is single channel" is not
     *  enough: a colour font atlas is power of two and multi channel, and would
     *  otherwise be chosen as swizzled and then reject every glyph. */
    bool m_force_linear;

    bool m_srgb;

    // ------------------------------------------------------------------------
    void upload(uint8_t* data, bool has_alpha_hint);
    // ------------------------------------------------------------------------
    void clearGXMData();
    // ------------------------------------------------------------------------
    bool allocate(unsigned mipmap_count);
    // ------------------------------------------------------------------------
    void applySamplerState();

public:
    // ------------------------------------------------------------------------
    GEGXMTexture(const std::string& path,
        std::function<void(irr::video::IImage*)> image_mani = nullptr);
    // ------------------------------------------------------------------------
    GEGXMTexture(irr::video::IImage* img, const std::string& name);
    // ------------------------------------------------------------------------
    GEGXMTexture(const std::string& name, unsigned size, bool single_channel);
    // ------------------------------------------------------------------------
    virtual ~GEGXMTexture();
    // ------------------------------------------------------------------------
    virtual void* lock(irr::video::E_TEXTURE_LOCK_MODE mode =
                       irr::video::ETLM_READ_WRITE, irr::u32 mipmap_level = 0);
    // ------------------------------------------------------------------------
    virtual void unlock()
    {
        if (m_locked_data)
        {
            delete [] m_locked_data;
            m_locked_data = NULL;
        }
    }
    // ------------------------------------------------------------------------
    virtual const irr::core::dimension2d<irr::u32>& getOriginalSize() const
                                                        { return m_orig_size; }
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
    virtual irr::u32 getPitch() const                       { return m_stride; }
    // ------------------------------------------------------------------------
    virtual bool hasMipMaps() const              { return m_mipmap_count > 1; }
    // ------------------------------------------------------------------------
    virtual void regenerateMipMapLevels(void* mipmap_data = NULL)            {}
    // ------------------------------------------------------------------------
    virtual irr::u64 getTextureHandler() const
                                            { return (irr::u64)(uintptr_t)this; }
    // ------------------------------------------------------------------------
    virtual unsigned getTextureSize() const          { return m_texture_size; }
    // ------------------------------------------------------------------------
    virtual void reload();
    // ------------------------------------------------------------------------
    virtual void updateTexture(void* data, irr::video::ECOLOR_FORMAT format,
                               irr::u32 w, irr::u32 h, irr::u32 x, irr::u32 y);
    // ------------------------------------------------------------------------
    /** The handle to hand to sceGxmSetFragmentTexture(). NULL for a texture
     *  whose upload failed, which callers must treat as "bind the driver's
     *  white placeholder instead". */
    const SceGxmTexture* getGXMTexture() const
                             { return m_data == NULL ? NULL : &m_texture; }
    // ------------------------------------------------------------------------
    /** Turns on the texture unit's sRGB to linear conversion. Set per material
     *  layer from GEMaterial::m_srgb_settings: albedo maps are authored in
     *  sRGB and have to be linearised before the BRDF sees them, while normal
     *  and PBR data maps must not be. */
    void setSRGB(bool srgb);
    // ------------------------------------------------------------------------
    bool isSRGB() const                                      { return m_srgb; }
    // ------------------------------------------------------------------------
    /** Overrides the default repeat addressing. Needed by the environment
     *  panorama, which wraps in u but must clamp in v so that a sample past a
     *  pole reads the pole rather than the opposite one. */
    void setAddressMode(SceGxmTextureAddrMode u, SceGxmTextureAddrMode v);
    // ------------------------------------------------------------------------
    bool isSingleChannel() const                   { return m_single_channel; }
};   // GEGXMTexture

// ----------------------------------------------------------------------------
/** Rearranges a linear image into the GPU's Morton order.
 *
 *  Both dimensions must be powers of two. The low bits of x and y interleave;
 *  whichever dimension is larger contributes its remaining high bits above the
 *  interleaved field, which is how the hardware addresses a non square
 *  swizzled surface. Exposed for the render target code, which swizzles a
 *  resolved colour surface before handing it back as a sampled texture. */
void gxmSwizzleImage(uint8_t* dest, const uint8_t* src, unsigned width,
                     unsigned height, unsigned bytes_per_pixel);

}   // namespace GE

#endif   // _IRR_COMPILE_WITH_GXM_

#endif
