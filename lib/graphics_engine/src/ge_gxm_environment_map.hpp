#ifndef HEADER_GE_GXM_ENVIRONMENT_MAP_HPP
#define HEADER_GE_GXM_ENVIRONMENT_MAP_HPP

#include "IrrCompileConfig.h"

#ifdef _IRR_COMPILE_WITH_GXM_

#include <psp2/gxm.h>

#include <SColor.h>
#include <vector>

namespace irr
{
    namespace scene { class ISceneNode; }
    namespace video { class ITexture; }
}

namespace GE
{
class GEGXMDriver;
class GEGXMTexture;

/** Turns STK's six skybox images into the environment the PBR pass lights
 *  from, and into the sky the skybox pass draws.
 *
 *  All of it is built on the CPU at track load rather than on the GPU. The
 *  Vulkan backend prefilters its cube maps with compute shaders, which the SGX
 *  does not have; the alternative on this hardware would be a fullscreen
 *  fragment pass per cube face per mip, reading each result back to swizzle it
 *  into place. Doing the whole thing on the CPU is both less code and less
 *  guesswork, and it happens once per track, not per frame.
 *
 *  Two panoramas come out of it:
 *
 *  - The radiance panorama, which is the sky itself. Its automatically
 *    generated mip chain stands in for the roughness levels of a prefiltered
 *    specular cube map. A box reduction is not a GGX lobe, but it is the same
 *    approximation the GLSL renderer's degraded IBL path makes and it is free.
 *  - The irradiance panorama, cosine convolved so that one tap gives the whole
 *    diffuse ambient term for a normal.
 *
 *  Both are equirectangular 2D textures, not cube maps - see dirToEquirect()
 *  in the shader sources for why. */
class GEGXMEnvironmentMap
{
private:
    GEGXMDriver* m_driver;

    irr::scene::ISceneNode* m_skybox;

    GEGXMTexture* m_radiance;

    GEGXMTexture* m_irradiance;

    irr::video::SColor m_skytop_color;

    float m_specular_levels;

    bool m_ready;

    /** Whether a build has been attempted for m_skybox, successful or not. */
    bool m_attempted;

    // ------------------------------------------------------------------------
    /** Resamples the six cube faces into one equirectangular panorama. */
    bool buildRadiance(irr::scene::ISceneNode* skybox,
                       std::vector<uint32_t>* panorama);
    // ------------------------------------------------------------------------
    void buildIrradiance(const std::vector<uint32_t>& panorama);

public:
    // ------------------------------------------------------------------------
    GEGXMEnvironmentMap(GEGXMDriver* driver);
    // ------------------------------------------------------------------------
    ~GEGXMEnvironmentMap();
    // ------------------------------------------------------------------------
    /** Rebuilds from a skybox scene node, unless it is the one already loaded.
     *  Safe to call every frame; the work only happens on a change. */
    void addSkyBox(irr::scene::ISceneNode* node);
    // ------------------------------------------------------------------------
    void reset();
    // ------------------------------------------------------------------------
    /** Forces a rebuild on the next addSkyBox(), for when the sRGB or PBR
     *  setting changed underneath us. */
    void invalidate();
    // ------------------------------------------------------------------------
    bool isReady() const                                    { return m_ready; }
    // ------------------------------------------------------------------------
    const SceGxmTexture* getRadianceTexture() const;
    // ------------------------------------------------------------------------
    const SceGxmTexture* getIrradianceTexture() const;
    // ------------------------------------------------------------------------
    /** Index of the highest mip of the radiance panorama, which is what a
     *  perceptual roughness of 1 maps to in the shader. */
    float getSpecularLevels() const               { return m_specular_levels; }
    // ------------------------------------------------------------------------
    /** Average colour of the top of the sky. Used as the ambient term when
     *  image based lighting is switched off. */
    irr::video::SColor getSkytopColor() const     { return m_skytop_color; }
};   // GEGXMEnvironmentMap

}   // namespace GE

#endif   // _IRR_COMPILE_WITH_GXM_

#endif
