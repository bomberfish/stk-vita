#include "ge_gxm_environment_map.hpp"

#ifdef _IRR_COMPILE_WITH_GXM_

#include "ge_gxm_driver.hpp"
#include "ge_gxm_limits.hpp"
#include "ge_gxm_texture.hpp"
#include "ge_main.hpp"

#include "../source/Irrlicht/os.h"

#include <ISceneNode.h>
#include <ITexture.h>
#include <IVideoDriver.h>

#include <cmath>
#include <cstring>

namespace GE
{
namespace
{
const float PI = 3.14159265358979323846f;

/** Order in which irrlicht's skybox node stores its six faces relative to the
 *  +X, -X, +Y, -Y, +Z, -Z the maths below wants. Taken from the Vulkan
 *  backend's GEVulkanSkyBoxRenderer so both renderers orient the sky the same
 *  way. */
const int FACE_ORDER[6] = { 1, 3, 4, 5, 2, 0 };

struct FaceImage
{
    std::vector<uint32_t> m_data;
    unsigned m_width;
    unsigned m_height;
};

// ----------------------------------------------------------------------------
/** Says why the environment map could not be built, at most a few times.
 *
 *  Worth reporting rather than failing quietly: with no environment map there is
 *  no skybox and no image based lighting, so the sky falls back to the fog
 *  colour - which looks like a deliberate art choice rather than a fault. */
void reportFailure(const char* reason, unsigned face, const char* name)
{
    static unsigned reported = 0;
    if (reported++ > 2)
        return;
    char line[256];
    snprintf(line, sizeof(line), "GXM: no environment map: %s (face %u%s%s)",
        reason, face, name == NULL ? "" : ", ", name == NULL ? "" : name);
    irr::os::Printer::log(line, irr::ELL_WARNING);
}   // reportFailure

// ----------------------------------------------------------------------------
/** Direction to (face, u, v). Standard cube map addressing: the largest
 *  magnitude component picks the face, the other two divided by it give the
 *  position on it. */
void dirToFace(float x, float y, float z, int* face, float* u, float* v)
{
    const float ax = fabsf(x), ay = fabsf(y), az = fabsf(z);
    if (ax >= ay && ax >= az)
    {
        if (x > 0.0f) { *face = 0; *u = -z / ax; *v = -y / ax; }
        else          { *face = 1; *u =  z / ax; *v = -y / ax; }
    }
    else if (ay >= az)
    {
        if (y > 0.0f) { *face = 2; *u =  x / ay; *v =  z / ay; }
        else          { *face = 3; *u =  x / ay; *v = -z / ay; }
    }
    else
    {
        if (z > 0.0f) { *face = 4; *u =  x / az; *v = -y / az; }
        else          { *face = 5; *u = -x / az; *v = -y / az; }
    }
    *u = (*u + 1.0f) * 0.5f;
    *v = (*v + 1.0f) * 0.5f;
}   // dirToFace

// ----------------------------------------------------------------------------
/** Bilinear sample of an ARGB face image. Clamped at the edges, which softens
 *  the cube seams a little rather than wrapping across them. */
uint32_t sampleFace(const FaceImage& img, float u, float v)
{
    if (img.m_width == 0 || img.m_height == 0)
        return 0;
    float fx = u * (float)img.m_width - 0.5f;
    float fy = v * (float)img.m_height - 0.5f;
    int x0 = (int)floorf(fx);
    int y0 = (int)floorf(fy);
    float tx = fx - (float)x0;
    float ty = fy - (float)y0;
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    const int max_x = (int)img.m_width - 1;
    const int max_y = (int)img.m_height - 1;
    x0 = x0 < 0 ? 0 : (x0 > max_x ? max_x : x0);
    x1 = x1 < 0 ? 0 : (x1 > max_x ? max_x : x1);
    y0 = y0 < 0 ? 0 : (y0 > max_y ? max_y : y0);
    y1 = y1 < 0 ? 0 : (y1 > max_y ? max_y : y1);

    const uint32_t c00 = img.m_data[(size_t)y0 * img.m_width + x0];
    const uint32_t c10 = img.m_data[(size_t)y0 * img.m_width + x1];
    const uint32_t c01 = img.m_data[(size_t)y1 * img.m_width + x0];
    const uint32_t c11 = img.m_data[(size_t)y1 * img.m_width + x1];

    uint32_t out = 0;
    for (unsigned c = 0; c < 4; c++)
    {
        const unsigned shift = c * 8;
        const float a = (float)((c00 >> shift) & 0xFF);
        const float b = (float)((c10 >> shift) & 0xFF);
        const float d = (float)((c01 >> shift) & 0xFF);
        const float e = (float)((c11 >> shift) & 0xFF);
        const float top = a + (b - a) * tx;
        const float bottom = d + (e - d) * tx;
        float value = top + (bottom - top) * ty;
        if (value < 0.0f)
            value = 0.0f;
        else if (value > 255.0f)
            value = 255.0f;
        out |= ((uint32_t)(value + 0.5f)) << shift;
    }
    return out;
}   // sampleFace

// ----------------------------------------------------------------------------
/** Box reduces an ARGB panorama by half in each dimension. */
void reduce(const std::vector<uint32_t>& src, unsigned width, unsigned height,
            std::vector<uint32_t>* dest, unsigned* out_width,
            unsigned* out_height)
{
    const unsigned dw = width > 1 ? width / 2 : 1;
    const unsigned dh = height > 1 ? height / 2 : 1;
    dest->assign((size_t)dw * dh, 0);
    for (unsigned y = 0; y < dh; y++)
    {
        const unsigned sy0 = std::min(y * 2, height - 1);
        const unsigned sy1 = std::min(y * 2 + 1, height - 1);
        for (unsigned x = 0; x < dw; x++)
        {
            const unsigned sx0 = std::min(x * 2, width - 1);
            const unsigned sx1 = std::min(x * 2 + 1, width - 1);
            const uint32_t s[4] =
            {
                src[(size_t)sy0 * width + sx0], src[(size_t)sy0 * width + sx1],
                src[(size_t)sy1 * width + sx0], src[(size_t)sy1 * width + sx1]
            };
            uint32_t out = 0;
            for (unsigned c = 0; c < 4; c++)
            {
                const unsigned shift = c * 8;
                unsigned sum = 0;
                for (unsigned i = 0; i < 4; i++)
                    sum += (s[i] >> shift) & 0xFF;
                out |= ((sum + 2) / 4) << shift;
            }
            (*dest)[(size_t)y * dw + x] = out;
        }
    }
    *out_width = dw;
    *out_height = dh;
}   // reduce
}   // namespace

// ============================================================================
GEGXMEnvironmentMap::GEGXMEnvironmentMap(GEGXMDriver* driver)
{
    m_driver = driver;
    m_skybox = NULL;
    m_radiance = NULL;
    m_irradiance = NULL;
    m_skytop_color = irr::video::SColor(255, 128, 160, 200);
    m_specular_levels = 0.0f;
    m_ready = false;
    m_attempted = false;
}   // GEGXMEnvironmentMap

// ----------------------------------------------------------------------------
GEGXMEnvironmentMap::~GEGXMEnvironmentMap()
{
    reset();
}   // ~GEGXMEnvironmentMap

// ----------------------------------------------------------------------------
void GEGXMEnvironmentMap::reset()
{
    if (m_radiance != NULL)
    {
        m_radiance->drop();
        m_radiance = NULL;
    }
    if (m_irradiance != NULL)
    {
        m_irradiance->drop();
        m_irradiance = NULL;
    }
    m_skybox = NULL;
    m_ready = false;
    m_attempted = false;
    m_specular_levels = 0.0f;
}   // reset

// ----------------------------------------------------------------------------
void GEGXMEnvironmentMap::invalidate()
{
    // Keeps the textures but forgets which node they came from, so the next
    // addSkyBox() rebuilds with the current sRGB and PBR settings.
    m_skybox = NULL;
}   // invalidate

// ----------------------------------------------------------------------------
void GEGXMEnvironmentMap::addSkyBox(irr::scene::ISceneNode* node)
{
    if (node == NULL || node->getType() != irr::scene::ESNT_SKY_BOX)
        return;
    // "attempted", not "ready": a build that failed because the irradiance pass
    // ran out of memory must not be retried for every camera of every frame,
    // which would rebuild a 512x256 panorama and its cosine convolution at
    // frame rate.
    if (m_skybox == node && m_attempted)
        return;

    reset();
    m_skybox = node;
    m_attempted = true;

    std::vector<uint32_t> panorama;
    if (!buildRadiance(node, &panorama))
    {
        // The face textures may simply not be loaded yet, so this one is worth
        // retrying rather than giving up on.
        m_skybox = NULL;
        m_attempted = false;
        return;
    }
    buildIrradiance(panorama);
    m_ready = m_radiance != NULL && m_irradiance != NULL;

}   // addSkyBox

// ----------------------------------------------------------------------------
bool GEGXMEnvironmentMap::buildRadiance(irr::scene::ISceneNode* skybox,
                                        std::vector<uint32_t>* panorama)
{
    FaceImage faces[6];
    for (unsigned i = 0; i < 6; i++)
    {
        irr::video::ITexture* tex =
            skybox->getMaterial(FACE_ORDER[i]).getTexture(0);
        if (tex == NULL)
        {
            reportFailure("face has no texture", i, NULL);
            return false;
        }
        // GEGXMTexture::lock() re-decodes the source file rather than reading
        // the GPU copy back, which is exactly what is wanted here: the GPU copy
        // is swizzled and in slow uncached video memory.
        void* data = tex->lock(irr::video::ETLM_READ_ONLY);
        if (data == NULL)
        {
            reportFailure("face could not be read back", i,
                tex->getName().getPtr());
            return false;
        }
        const irr::core::dimension2du& size = tex->getSize();
        faces[i].m_width = size.Width;
        faces[i].m_height = size.Height;
        faces[i].m_data.assign((const uint32_t*)data,
            (const uint32_t*)data + (size_t)size.Width * size.Height);
        tex->unlock();
    }

    const unsigned w = GXM_ENV_WIDTH;
    const unsigned h = GXM_ENV_HEIGHT;
    panorama->assign((size_t)w * h, 0);

    // The inverse of dirToEquirect() in the shaders. Getting these two out of
    // step would rotate the sky relative to the reflections in it, so they are
    // written to mirror each other exactly.
    for (unsigned y = 0; y < h; y++)
    {
        const float v = ((float)y + 0.5f) / (float)h;
        const float theta = v * PI;
        const float sin_theta = sinf(theta);
        const float cy = cosf(theta);
        for (unsigned x = 0; x < w; x++)
        {
            const float u = ((float)x + 0.5f) / (float)w;
            const float phi = (u - 0.5f) * 2.0f * PI;
            const float cx = sin_theta * sinf(phi);
            const float cz = -sin_theta * cosf(phi);

            int face = 0;
            float fu = 0.0f, fv = 0.0f;
            dirToFace(cx, cy, cz, &face, &fu, &fv);
            (*panorama)[(size_t)y * w + x] = sampleFace(faces[face], fu, fv);
        }
    }

    // Average of the top row band, used as the ambient colour when image based
    // lighting is off.
    uint64_t sum[3] = { 0, 0, 0 };
    const unsigned band = std::max(1u, h / 16);
    for (unsigned y = 0; y < band; y++)
    {
        for (unsigned x = 0; x < w; x++)
        {
            const uint32_t c = (*panorama)[(size_t)y * w + x];
            sum[0] += (c >> 16) & 0xFF;
            sum[1] += (c >> 8) & 0xFF;
            sum[2] += c & 0xFF;
        }
    }
    const uint64_t count = (uint64_t)band * w;
    m_skytop_color = irr::video::SColor(255, (uint32_t)(sum[0] / count),
        (uint32_t)(sum[1] / count), (uint32_t)(sum[2] / count));

    // Handing it to GEGXMTexture gets the swizzled layout and the full mip
    // chain for free, and the mip chain is what the shader treats as roughness
    // levels. The image is ECF_A8R8G8B8, i.e. the same 0xAARRGGBB words the
    // panorama was built from.
    irr::video::IImage* image = m_driver->createImageFromData(
        irr::video::ECF_A8R8G8B8, irr::core::dimension2du(w, h),
        panorama->data(), false/*ownForeignMemory*/, true/*deleteMemory*/);
    if (image == NULL)
        return false;
    m_radiance = new GEGXMTexture(image, "gxm_env_radiance");
    if (m_radiance->getGXMTexture() == NULL)
    {
        m_radiance->drop();
        m_radiance = NULL;
        return false;
    }
    // A panorama wraps horizontally and must not wrap vertically: a v sample
    // past the pole belongs to the pole, not to the opposite pole.
    m_radiance->setAddressMode(SCE_GXM_TEXTURE_ADDR_REPEAT,
        SCE_GXM_TEXTURE_ADDR_CLAMP);
    m_radiance->setSRGB(getGEConfig()->m_pbr);

    // Highest mip index, which perceptual roughness 1.0 maps onto.
    unsigned levels = 1;
    unsigned mw = w, mh = h;
    while (mw > 1 || mh > 1)
    {
        mw = mw > 1 ? mw / 2 : 1;
        mh = mh > 1 ? mh / 2 : 1;
        levels++;
    }
    m_specular_levels = (float)(levels - 1);
    return true;
}   // buildRadiance

// ----------------------------------------------------------------------------
void GEGXMEnvironmentMap::buildIrradiance(
                                       const std::vector<uint32_t>& panorama)
{
    // Reduce the panorama down to something small enough to integrate over
    // every output texel without the cost exploding.
    std::vector<uint32_t> source = panorama;
    unsigned sw = GXM_ENV_WIDTH;
    unsigned sh = GXM_ENV_HEIGHT;
    for (unsigned i = 0; i < GXM_IRRADIANCE_SOURCE_LEVEL; i++)
    {
        std::vector<uint32_t> reduced;
        unsigned nw = 0, nh = 0;
        reduce(source, sw, sh, &reduced, &nw, &nh);
        source = std::move(reduced);
        sw = nw;
        sh = nh;
    }

    const unsigned w = GXM_IRRADIANCE_WIDTH;
    const unsigned h = GXM_IRRADIANCE_HEIGHT;
    std::vector<uint32_t> out((size_t)w * h, 0);

    // Precompute the source directions and their solid angles once; the
    // integral below reads them once per output texel.
    struct Sample
    {
        float m_dir[3];
        float m_weight;
        float m_color[3];
    };
    std::vector<Sample> samples;
    samples.reserve((size_t)sw * sh);
    for (unsigned y = 0; y < sh; y++)
    {
        const float v = ((float)y + 0.5f) / (float)sh;
        const float theta = v * PI;
        const float sin_theta = sinf(theta);
        for (unsigned x = 0; x < sw; x++)
        {
            const float u = ((float)x + 0.5f) / (float)sw;
            const float phi = (u - 0.5f) * 2.0f * PI;
            Sample s;
            s.m_dir[0] = sin_theta * sinf(phi);
            s.m_dir[1] = cosf(theta);
            s.m_dir[2] = -sin_theta * cosf(phi);
            // sin(theta) is the equirectangular projection's area distortion:
            // rows near the poles cover far less of the sphere than rows near
            // the equator, and weighting by it is what keeps the integral from
            // being dominated by the poles.
            s.m_weight = sin_theta;
            const uint32_t c = source[(size_t)y * sw + x];
            s.m_color[0] = (float)((c >> 16) & 0xFF);
            s.m_color[1] = (float)((c >> 8) & 0xFF);
            s.m_color[2] = (float)(c & 0xFF);
            samples.push_back(s);
        }
    }

    for (unsigned y = 0; y < h; y++)
    {
        const float v = ((float)y + 0.5f) / (float)h;
        const float theta = v * PI;
        const float sin_theta = sinf(theta);
        for (unsigned x = 0; x < w; x++)
        {
            const float u = ((float)x + 0.5f) / (float)w;
            const float phi = (u - 0.5f) * 2.0f * PI;
            const float nx = sin_theta * sinf(phi);
            const float ny = cosf(theta);
            const float nz = -sin_theta * cosf(phi);

            float acc[3] = { 0.0f, 0.0f, 0.0f };
            float total = 0.0f;
            for (const Sample& s : samples)
            {
                const float cosine = nx * s.m_dir[0] + ny * s.m_dir[1] +
                    nz * s.m_dir[2];
                if (cosine <= 0.0f)
                    continue;
                const float weight = cosine * s.m_weight;
                acc[0] += s.m_color[0] * weight;
                acc[1] += s.m_color[1] * weight;
                acc[2] += s.m_color[2] * weight;
                total += weight;
            }
            uint32_t color = 0xFF000000u;
            if (total > 0.0f)
            {
                for (unsigned c = 0; c < 3; c++)
                {
                    float value = acc[c] / total;
                    if (value > 255.0f)
                        value = 255.0f;
                    color |= ((uint32_t)(value + 0.5f)) << (16 - c * 8);
                }
            }
            out[(size_t)y * w + x] = color;
        }
    }

    irr::video::IImage* image = m_driver->createImageFromData(
        irr::video::ECF_A8R8G8B8, irr::core::dimension2du(w, h), out.data(),
        false/*ownForeignMemory*/, true/*deleteMemory*/);
    if (image == NULL)
        return;
    m_irradiance = new GEGXMTexture(image, "gxm_env_irradiance");
    if (m_irradiance->getGXMTexture() == NULL)
    {
        m_irradiance->drop();
        m_irradiance = NULL;
        return;
    }
    m_irradiance->setAddressMode(SCE_GXM_TEXTURE_ADDR_REPEAT,
        SCE_GXM_TEXTURE_ADDR_CLAMP);
    m_irradiance->setSRGB(getGEConfig()->m_pbr);
}   // buildIrradiance

// ----------------------------------------------------------------------------
const SceGxmTexture* GEGXMEnvironmentMap::getRadianceTexture() const
{
    return m_radiance == NULL ? NULL : m_radiance->getGXMTexture();
}   // getRadianceTexture

// ----------------------------------------------------------------------------
const SceGxmTexture* GEGXMEnvironmentMap::getIrradianceTexture() const
{
    return m_irradiance == NULL ? NULL : m_irradiance->getGXMTexture();
}   // getIrradianceTexture

}   // namespace GE

#endif   // _IRR_COMPILE_WITH_GXM_
