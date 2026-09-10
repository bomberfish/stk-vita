#include "ge_gxm_texture.hpp"

#include <cmath>

#ifdef _IRR_COMPILE_WITH_GXM_

#include "ge_gxm_driver.hpp"
#include "ge_gxm_memory.hpp"
#include "ge_main.hpp"
#include "ge_texture.hpp"

#include "../source/Irrlicht/os.h"

#include <IAttributes.h>
#include <IVideoDriver.h>

#include <cstring>
#include <vector>

namespace GE
{
namespace
{
// ----------------------------------------------------------------------------
/** sRGB <-> linear tables for the mip filter. 16 bit on the linear side so that
 *  averaging four samples does not quantise the shadows, 12 bit on the way back
 *  because that is plenty to land on the right 8 bit value. */
struct SRGBTables
{
    uint16_t m_to_linear[256];
    uint8_t m_to_srgb[4096];
    SRGBTables()
    {
        for (unsigned i = 0; i < 256; i++)
        {
            const double c = i / 255.0;
            const double linear = c <= 0.04045 ? c / 12.92 :
                pow((c + 0.055) / 1.055, 2.4);
            m_to_linear[i] = (uint16_t)(linear * 65535.0 + 0.5);
        }
        for (unsigned i = 0; i < 4096; i++)
        {
            const double linear = i / 4095.0;
            const double c = linear <= 0.0031308 ? linear * 12.92 :
                1.055 * pow(linear, 1.0 / 2.4) - 0.055;
            m_to_srgb[i] = (uint8_t)(c * 255.0 + 0.5);
        }
    }
};

// ----------------------------------------------------------------------------
/** Builds the whole mip chain with a 2x2 box filter in linear light, returning
 *  levels 1..n (level 0 is the caller's own buffer).
 *
 *  graphics_engine's shared generator, imBuildMipmapCascade, uses a wide gamma
 *  correct filter in floating point. That is the right trade on a desktop, but
 *  here it costs about 190 ms for a 512x512 texture and is paid a few hundred
 *  times while a track loads - measurably the single largest cost of getting
 *  into a race. A box filter is a shade softer at 544 lines and costs about a
 *  fiftieth as much, so the GXM backend uses this instead. Alpha is averaged
 *  directly: it is not gamma encoded. */
void buildBoxMipmaps(const uint8_t* source, unsigned width, unsigned height,
                     std::vector<std::vector<uint8_t>>* out)
{
    static const SRGBTables tables;
    out->clear();
    const uint8_t* src = source;
    unsigned sw = width;
    unsigned sh = height;
    while (sw > 1 || sh > 1)
    {
        const unsigned dw = sw < 2 ? 1 : sw >> 1;
        const unsigned dh = sh < 2 ? 1 : sh >> 1;
        out->push_back(std::vector<uint8_t>((size_t)dw * dh * 4));
        uint8_t* dst = out->back().data();
        // When a dimension has already bottomed out the box degenerates to a
        // 1xN or Nx1 average rather than reading past the edge.
        const unsigned step_x = sw > 1 ? 2 : 1;
        const unsigned step_y = sh > 1 ? 2 : 1;
        const unsigned dx = sw > 1 ? 1 : 0;
        const unsigned dy = sh > 1 ? 1 : 0;
        for (unsigned y = 0; y < dh; y++)
        {
            const uint8_t* row0 = src + (size_t)(y * step_y) * sw * 4;
            const uint8_t* row1 = row0 + (size_t)dy * sw * 4;
            uint8_t* d = dst + (size_t)y * dw * 4;
            for (unsigned x = 0; x < dw; x++)
            {
                const unsigned x0 = x * step_x * 4;
                const unsigned x1 = x0 + dx * 4;
                for (unsigned c = 0; c < 3; c++)
                {
                    const unsigned sum =
                        tables.m_to_linear[row0[x0 + c]] +
                        tables.m_to_linear[row0[x1 + c]] +
                        tables.m_to_linear[row1[x0 + c]] +
                        tables.m_to_linear[row1[x1 + c]];
                    d[x * 4 + c] = tables.m_to_srgb[sum >> 6];
                }
                d[x * 4 + 3] = (uint8_t)((row0[x0 + 3] + row0[x1 + 3] +
                    row1[x0 + 3] + row1[x1 + 3] + 2) >> 2);
            }
        }
        src = dst;
        sw = dw;
        sh = dh;
    }
}   // buildBoxMipmaps
}   // namespace

namespace
{
// ----------------------------------------------------------------------------
inline bool isPowerOfTwo(unsigned v)
{
    return v != 0 && (v & (v - 1)) == 0;
}   // isPowerOfTwo

// ----------------------------------------------------------------------------
inline unsigned log2u(unsigned v)
{
    return 31u - (unsigned)__builtin_clz(v);
}   // log2u

// ----------------------------------------------------------------------------
/** Swaps the red and blue channels in place.
 *
 *  Irrlicht's ECF_A8R8G8B8 is a little endian 0xAARRGGBB word, so in memory the
 *  bytes run B, G, R, A. GXM's U8U8U8U8_ABGR - the format that pairs with the
 *  display's A8B8G8R8 - is R, G, B, A. Only the two ends differ. */
void bgraToRgba(uint8_t* data, unsigned pixels)
{
    for (unsigned i = 0; i < pixels; i++)
    {
        uint8_t tmp = data[i * 4];
        data[i * 4] = data[i * 4 + 2];
        data[i * 4 + 2] = tmp;
    }
}   // bgraToRgba
}   // namespace

// ----------------------------------------------------------------------------
void gxmSwizzleImage(uint8_t* dest, const uint8_t* src, unsigned width,
                     unsigned height, unsigned bytes_per_pixel)
{
    const unsigned min_dim = width < height ? width : height;
    const unsigned interleave_bits = log2u(min_dim);
    const bool wide = width > height;

    // The address of a texel is its row bits interleaved with its column bits.
    // Computing that per texel means an inner loop over the bits, which for a
    // 1024x1024 texture is around eleven million iterations before the mip
    // chain - measurable even on a desktop, and STK loads hundreds of textures.
    //
    // Since the row and column contributions occupy disjoint bit positions,
    // each can be spread once per coordinate value and the pair simply OR'd:
    // one table of height entries, one of width, then two lookups per texel.
    //
    // Row bits go into the even positions and column bits into the odd ones -
    // not the other way round, which transposes the image and reads as a 90
    // degree rotation. That order is not derivable from the headers; it is
    // pinned by vitaGL's dxt_compress(), which decodes a Morton index and
    // addresses its source as src + offs_y * 16 + offs_x * w * 16, i.e. the
    // even-bit component is the one multiplied by the row pitch.
    std::vector<unsigned> row_code(height);
    std::vector<unsigned> column_code(width);
    for (unsigned y = 0; y < height; y++)
    {
        unsigned code = 0;
        for (unsigned b = 0; b < interleave_bits; b++)
            code |= ((y >> b) & 1u) << (2 * b);
        // Above the square region the two dimensions share, the larger one
        // simply counts on in the high bits.
        if (!wide)
            code |= (y >> interleave_bits) << (2 * interleave_bits);
        row_code[y] = code;
    }
    for (unsigned x = 0; x < width; x++)
    {
        unsigned code = 0;
        for (unsigned b = 0; b < interleave_bits; b++)
            code |= ((x >> b) & 1u) << (2 * b + 1);
        if (wide)
            code |= (x >> interleave_bits) << (2 * interleave_bits);
        column_code[x] = code;
    }

    if (bytes_per_pixel == 4)
    {
        // The overwhelmingly common case, and worth a word-at-a-time copy
        // rather than a four byte memcpy per texel.
        const uint32_t* source = (const uint32_t*)src;
        uint32_t* destination = (uint32_t*)dest;
        for (unsigned y = 0; y < height; y++)
        {
            const unsigned row = row_code[y];
            const uint32_t* source_row = source + (size_t)y * width;
            for (unsigned x = 0; x < width; x++)
                destination[row | column_code[x]] = source_row[x];
        }
        return;
    }

    for (unsigned y = 0; y < height; y++)
    {
        const unsigned row = row_code[y];
        for (unsigned x = 0; x < width; x++)
        {
            memcpy(dest + (size_t)(row | column_code[x]) * bytes_per_pixel,
                src + ((size_t)y * width + x) * bytes_per_pixel,
                bytes_per_pixel);
        }
    }
}   // gxmSwizzleImage

// ============================================================================
GEGXMTexture::GEGXMTexture(const std::string& path,
                           std::function<void(irr::video::IImage*)> image_mani)
            : irr::video::ITexture(path.c_str()), m_image_mani(image_mani),
              m_locked_data(NULL), m_data(NULL), m_texture_size(0),
              m_mipmap_count(1), m_stride(0),
              m_format(SCE_GXM_TEXTURE_FORMAT_U8U8U8U8_ABGR),
              m_disable_reload(false), m_single_channel(false),
              m_swizzled(false), m_force_linear(false), m_srgb(false)
{
    memset(&m_texture, 0, sizeof(m_texture));
    reload();
}   // GEGXMTexture

// ----------------------------------------------------------------------------
GEGXMTexture::GEGXMTexture(irr::video::IImage* img, const std::string& name)
            : irr::video::ITexture(name.c_str()), m_image_mani(nullptr),
              m_locked_data(NULL), m_data(NULL), m_texture_size(0),
              m_mipmap_count(1), m_stride(0),
              m_format(SCE_GXM_TEXTURE_FORMAT_U8U8U8U8_ABGR),
              m_disable_reload(true), m_single_channel(false),
              m_swizzled(false), m_force_linear(false), m_srgb(false)
{
    memset(&m_texture, 0, sizeof(m_texture));
    if (!img)
    {
        LoadingFailed = true;
        return;
    }
    m_size = m_orig_size = img->getDimension();
    uint8_t* data = (uint8_t*)img->lock();
    upload(data, true);
    img->unlock();
    img->drop();
}   // GEGXMTexture

// ----------------------------------------------------------------------------
GEGXMTexture::GEGXMTexture(const std::string& name, unsigned size,
                           bool single_channel)
            : irr::video::ITexture(name.c_str()), m_image_mani(nullptr),
              m_locked_data(NULL), m_data(NULL), m_texture_size(0),
              m_mipmap_count(1), m_stride(0),
              m_format(SCE_GXM_TEXTURE_FORMAT_U8U8U8U8_ABGR),
              m_disable_reload(true), m_single_channel(single_channel),
              m_swizzled(false), m_force_linear(true), m_srgb(false)
{
    memset(&m_texture, 0, sizeof(m_texture));
    m_orig_size.Width = size;
    m_orig_size.Height = size;
    m_size = m_orig_size;
    // A font atlas is written glyph by glyph through updateTexture(), so it
    // stays linear. Single channel costs a quarter of the memory of the RGBA
    // fallback and the texture unit expands it to (1, 1, 1, coverage) for free.
    m_format = m_single_channel ? SCE_GXM_TEXTURE_FORMAT_U8_R111 :
        SCE_GXM_TEXTURE_FORMAT_U8U8U8U8_ABGR;
    std::vector<uint8_t> data(size * size * (m_single_channel ? 1 : 4), 0);
    upload(data.data(), true);
}   // GEGXMTexture

// ----------------------------------------------------------------------------
GEGXMTexture::~GEGXMTexture()
{
    clearGXMData();
    if (m_locked_data)
    {
        delete [] m_locked_data;
        m_locked_data = NULL;
    }
}   // ~GEGXMTexture

// ----------------------------------------------------------------------------
void GEGXMTexture::clearGXMData()
{
    if (m_data != NULL)
    {
        // A frame that has been submitted but not yet displayed may still be
        // sampling this. Freeing it can return the pages to the kernel, at
        // which point the GPU faults on them, so drain first. Only paid when a
        // texture is actually destroyed or reloaded, not per frame.
        GEGXMDriver* driver = getGXMDriver();
        if (driver != NULL)
            driver->waitIdle();
        GEGXMMemory::free(m_data);
        m_data = NULL;
    }
    m_texture_size = 0;
}   // clearGXMData

// ----------------------------------------------------------------------------
void GEGXMTexture::reload()
{
    if (m_disable_reload)
        return;
    const irr::core::dimension2du& max_size = getDriver()
        ->getDriverAttributes().getAttributeAsDimension2d("MAX_TEXTURE_SIZE");
    irr::video::IImage* texture_image = getResizedImage(NamedPath.getPtr(),
        max_size, &m_orig_size);
    if (texture_image == NULL)
    {
        LoadingFailed = true;
        return;
    }
    m_size = texture_image->getDimension();
    if (m_image_mani)
        m_image_mani(texture_image);
    uint8_t* data = (uint8_t*)texture_image->lock();
    upload(data, true);
    texture_image->unlock();
    texture_image->drop();
}   // reload

// ----------------------------------------------------------------------------
bool GEGXMTexture::allocate(unsigned mipmap_count)
{
    const unsigned bpp = m_single_channel ? 1 : 4;
    size_t total = 0;
    if (m_swizzled)
    {
        unsigned w = m_size.Width;
        unsigned h = m_size.Height;
        for (unsigned i = 0; i < mipmap_count; i++)
        {
            total += (size_t)w * h * bpp;
            w = w < 2 ? 1 : w >> 1;
            h = h < 2 ? 1 : h >> 1;
        }
        m_stride = 0;
    }
    else
    {
        // A linear strided texture's stride has to be a multiple of four
        // bytes; with 32 bpp that is automatic, with 8 bpp it is not.
        m_stride = (m_size.Width * bpp + 3u) & ~3u;
        total = (size_t)m_stride * m_size.Height;
    }

    clearGXMData();
    // Textures are written once and then sampled every frame, which is exactly
    // what CDRAM is for. GEGXMMemory falls back to LPDDR when the 128 MB is
    // gone rather than failing the allocation.
    m_data = GEGXMMemory::allocate(GGMP_CDRAM, total, SCE_GXM_TEXTURE_ALIGNMENT);
    if (m_data == NULL)
    {
        irr::os::Printer::log("GXM: out of GPU memory for texture",
            NamedPath.getPtr(), irr::ELL_WARNING);
        LoadingFailed = true;
        return false;
    }
    m_texture_size = (unsigned)total;
    m_mipmap_count = mipmap_count;
    return true;
}   // allocate

// ----------------------------------------------------------------------------
void GEGXMTexture::upload(uint8_t* data, bool has_alpha_hint)
{
    if (m_size.Width == 0 || m_size.Height == 0)
    {
        LoadingFailed = true;
        return;
    }

    const unsigned bpp = m_single_channel ? 1 : 4;
    // Mip maps need the swizzled layout, and the swizzled layout needs both
    // dimensions to be powers of two. The font atlas opts out because its
    // glyphs are patched in afterwards.
    m_swizzled = !m_force_linear && !m_single_channel &&
        isPowerOfTwo(m_size.Width) && isPowerOfTwo(m_size.Height);

    if (!m_single_channel)
        bgraToRgba(data, m_size.Width * m_size.Height);

    unsigned mipmap_count = 1;
    std::vector<std::vector<uint8_t>> mipmaps;
    if (m_swizzled && m_size.Width > 1 && m_size.Height > 1)
    {
        // Down to 1x1, which is what the texture unit wants for a complete
        // chain.
        buildBoxMipmaps(data, m_size.Width, m_size.Height, &mipmaps);
        mipmap_count = (unsigned)mipmaps.size() + 1;
        // The mip count field in a GXM texture is four bits wide.
        if (mipmap_count > 16)
            mipmap_count = 16;
    }

    if (!allocate(mipmap_count))
        return;

    uint8_t* dest = (uint8_t*)m_data;
    if (m_swizzled)
    {
        unsigned w = m_size.Width;
        unsigned h = m_size.Height;
        for (unsigned level = 0; level < mipmap_count; level++)
        {
            const uint8_t* src = level == 0 ? data :
                mipmaps[level - 1].data();
            gxmSwizzleImage(dest, src, w, h, bpp);
            dest += (size_t)w * h * bpp;
            w = w < 2 ? 1 : w >> 1;
            h = h < 2 ? 1 : h >> 1;
        }
        if (sceGxmTextureInitSwizzled(&m_texture, m_data, m_format,
            m_size.Width, m_size.Height, mipmap_count) < 0)
        {
            irr::os::Printer::log("GXM: sceGxmTextureInitSwizzled failed",
                NamedPath.getPtr(), irr::ELL_WARNING);
            clearGXMData();
            LoadingFailed = true;
            return;
        }
    }
    else
    {
        const unsigned row_bytes = m_size.Width * bpp;
        for (unsigned y = 0; y < m_size.Height; y++)
            memcpy(dest + (size_t)y * m_stride, data + (size_t)y * row_bytes,
                row_bytes);
        if (sceGxmTextureInitLinearStrided(&m_texture, m_data, m_format,
            m_size.Width, m_size.Height, m_stride) < 0)
        {
            irr::os::Printer::log("GXM: sceGxmTextureInitLinearStrided failed",
                NamedPath.getPtr(), irr::ELL_WARNING);
            clearGXMData();
            LoadingFailed = true;
            return;
        }
    }
    applySamplerState();
}   // upload

// ----------------------------------------------------------------------------
void GEGXMTexture::applySamplerState()
{
    // GXM has no separate sampler objects: filtering and addressing live in
    // the texture control words, so they are set once here.
    sceGxmTextureSetMinFilter(&m_texture, SCE_GXM_TEXTURE_FILTER_LINEAR);
    sceGxmTextureSetMagFilter(&m_texture, SCE_GXM_TEXTURE_FILTER_LINEAR);
    sceGxmTextureSetMipFilter(&m_texture, m_mipmap_count > 1 ?
        SCE_GXM_TEXTURE_MIP_FILTER_ENABLED :
        SCE_GXM_TEXTURE_MIP_FILTER_DISABLED);
    sceGxmTextureSetUAddrMode(&m_texture, SCE_GXM_TEXTURE_ADDR_REPEAT);
    sceGxmTextureSetVAddrMode(&m_texture, SCE_GXM_TEXTURE_ADDR_REPEAT);
    sceGxmTextureSetGammaMode(&m_texture, m_srgb ?
        SCE_GXM_TEXTURE_GAMMA_BGR : SCE_GXM_TEXTURE_GAMMA_NONE);
}   // applySamplerState

// ----------------------------------------------------------------------------
void GEGXMTexture::setSRGB(bool srgb)
{
    if (m_srgb == srgb || m_single_channel)
        return;
    m_srgb = srgb;
    if (m_data != NULL)
    {
        sceGxmTextureSetGammaMode(&m_texture, m_srgb ?
            SCE_GXM_TEXTURE_GAMMA_BGR : SCE_GXM_TEXTURE_GAMMA_NONE);
    }
}   // setSRGB

// ----------------------------------------------------------------------------
void GEGXMTexture::setAddressMode(SceGxmTextureAddrMode u,
                                  SceGxmTextureAddrMode v)
{
    if (m_data == NULL)
        return;
    sceGxmTextureSetUAddrMode(&m_texture, u);
    sceGxmTextureSetVAddrMode(&m_texture, v);
}   // setAddressMode

// ----------------------------------------------------------------------------
void* GEGXMTexture::lock(irr::video::E_TEXTURE_LOCK_MODE mode,
                         irr::u32 mipmap_level)
{
    if (mode != irr::video::ETLM_READ_ONLY)
        return NULL;

    // Reading a swizzled texture back would mean un-swizzling it, and CDRAM
    // reads from the CPU are glacial, so re-decode the file instead. This is
    // the same trade the GLES path makes.
    const irr::core::dimension2du& max_size = getDriver()
        ->getDriverAttributes().getAttributeAsDimension2d("MAX_TEXTURE_SIZE");
    irr::video::IImage* img = getResizedImage(NamedPath.getPtr(), max_size,
        NULL, &m_size);
    if (!img)
        return NULL;
    img->setDeleteMemory(false);
    m_locked_data = (uint8_t*)img->lock();
    img->unlock();
    img->drop();
    return m_locked_data;
}   // lock

// ----------------------------------------------------------------------------
void GEGXMTexture::updateTexture(void* data, irr::video::ECOLOR_FORMAT format,
                                 irr::u32 w, irr::u32 h, irr::u32 x,
                                 irr::u32 y)
{
    if (m_data == NULL || w == 0 || h == 0)
        return;
    if (x + w > m_size.Width || y + h > m_size.Height)
        return;
    // Only the linear layout can take a sub rectangle cheaply; every caller
    // that uses this (the font atlas) creates its texture linear.
    if (m_swizzled)
    {
        irr::os::Printer::log("GXM: updateTexture on a swizzled texture is "
            "not supported", irr::ELL_WARNING);
        return;
    }

    uint8_t* dest = (uint8_t*)m_data;
    if (m_single_channel)
    {
        if (format != irr::video::ECF_R8)
            return;
        const uint8_t* src = (const uint8_t*)data;
        for (unsigned row = 0; row < h; row++)
        {
            memcpy(dest + (size_t)(y + row) * m_stride + x,
                src + (size_t)row * w, w);
        }
        return;
    }

    if (format == irr::video::ECF_R8)
    {
        // Coverage only glyph data expanded into the alpha channel of an
        // otherwise white texel, matching the single channel swizzle.
        const uint8_t* src = (const uint8_t*)data;
        for (unsigned row = 0; row < h; row++)
        {
            uint8_t* d = dest + (size_t)(y + row) * m_stride + (size_t)x * 4;
            for (unsigned col = 0; col < w; col++)
            {
                d[col * 4 + 0] = 255;
                d[col * 4 + 1] = 255;
                d[col * 4 + 2] = 255;
                d[col * 4 + 3] = src[(size_t)row * w + col];
            }
        }
    }
    else if (format == irr::video::ECF_A8R8G8B8)
    {
        uint8_t* src = (uint8_t*)data;
        bgraToRgba(src, w * h);
        for (unsigned row = 0; row < h; row++)
        {
            memcpy(dest + (size_t)(y + row) * m_stride + (size_t)x * 4,
                src + (size_t)row * w * 4, (size_t)w * 4);
        }
    }
}   // updateTexture

}   // namespace GE

#endif   // _IRR_COMPILE_WITH_GXM_
