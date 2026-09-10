#include "ge_gxm_post_processing.hpp"

#ifdef _IRR_COMPILE_WITH_GXM_

#include "ge_gxm_driver.hpp"
#include "ge_gxm_limits.hpp"
#include "ge_gxm_render_target.hpp"
#include "ge_gxm_shader.hpp"
#include "ge_main.hpp"

#include "../source/Irrlicht/os.h"

#include <cstring>

namespace GE
{
namespace
{
/** Luminance above which a pixel starts contributing to bloom, and how much of
 *  the blurred result is added back. Matched by eye to the GLSL renderer's
 *  bloom so switching driver does not change how bright the game looks. */
const float BLOOM_THRESHOLD = 0.9f;
const float BLOOM_STRENGTH = 0.35f;
}   // namespace

// ============================================================================
GEGXMPostProcessing::GEGXMPostProcessing(GEGXMDriver* driver)
{
    m_driver = driver;
    m_bright_program = NULL;
    m_blur_program = NULL;
    m_composite_program = NULL;
    m_bright_patched = NULL;
    m_blur_patched = NULL;
    m_composite_patched = NULL;
    m_width = 0;
    m_height = 0;
    m_enabled = false;
    m_bloom_enabled = false;
}   // GEGXMPostProcessing

// ----------------------------------------------------------------------------
GEGXMPostProcessing::~GEGXMPostProcessing()
{
    destroy();
}   // ~GEGXMPostProcessing

// ----------------------------------------------------------------------------
void GEGXMPostProcessing::destroy()
{
    m_scene.reset();
    m_bloom[0].reset();
    m_bloom[1].reset();
    m_enabled = false;
    m_bloom_enabled = false;
}   // destroy

// ----------------------------------------------------------------------------
void GEGXMPostProcessing::init(unsigned width, unsigned height)
{
    m_driver->waitIdle();
    destroy();
    m_width = width;
    m_height = height;

    // Off unless asked for, and only meaningful with the PBR pipeline: without
    // it the shading output is already display referred and there is nothing
    // above white to bloom from.
    //
    // Leaving it off is not just skipping an effect. It is what lets the 3D
    // scene render straight into the display buffer, so the whole frame is one
    // scene and the colour never has to be resolved to memory and read back -
    // by some distance the cheapest thing this GPU can be asked to do.
    m_bloom_enabled = getGEConfig()->m_pbr && getGEConfig()->m_bloom;
    if (!m_bloom_enabled)
        return;

    m_bright_program =
        GEGXMShaderManager::getFragmentProgram("bloom_bright.frag");
    m_blur_program = GEGXMShaderManager::getFragmentProgram("blur.frag");
    m_composite_program = GEGXMShaderManager::getFragmentProgram(
        "composite.frag", "#define GXM_BLOOM 1");
    GEGXMProgram* vertex = m_driver->getFullscreenVertexProgram();
    if (m_bright_program == NULL || m_blur_program == NULL ||
        m_composite_program == NULL || vertex == NULL)
    {
        irr::os::Printer::log("GXM: post processing shaders unavailable, "
            "bloom will be disabled", irr::ELL_WARNING);
        m_bloom_enabled = false;
        return;
    }

    GEGXMProgramCache* cache = m_driver->getProgramCache();
    m_bright_patched = cache->getFragmentProgram(m_bright_program, vertex,
        GEGXMBlendState::opaque());
    m_blur_patched = cache->getFragmentProgram(m_blur_program, vertex,
        GEGXMBlendState::opaque());
    m_composite_patched = cache->getFragmentProgram(m_composite_program,
        vertex, GEGXMBlendState::opaque());
    if (m_bright_patched == NULL || m_blur_patched == NULL ||
        m_composite_patched == NULL)
    {
        m_bloom_enabled = false;
        return;
    }

    m_scene.reset(new GEGXMFBOTexture("gxm_scene", width, height,
        m_driver->getCurrentSurface()));
    const unsigned bw = std::max(1u, width / 4);
    const unsigned bh = std::max(1u, height / 4);
    m_bloom[0].reset(new GEGXMFBOTexture("gxm_bloom_0", bw, bh, NULL));
    m_bloom[1].reset(new GEGXMFBOTexture("gxm_bloom_1", bw, bh, NULL));

    if (!m_scene->isValid() || !m_bloom[0]->isValid() ||
        !m_bloom[1]->isValid())
    {
        irr::os::Printer::log("GXM: not enough video memory for the post "
            "processing targets, bloom will be disabled", irr::ELL_WARNING);
        destroy();
        return;
    }
    // The scene target is written in linear light and read back for the bright
    // pass, so it carries the sRGB encode on write to keep 8 bits of precision
    // usable in the shadows.
    m_scene->getSurface()->setSRGBWrite(getGEConfig()->m_pbr);
    m_enabled = true;
}   // init

// ----------------------------------------------------------------------------
GEGXMSurface* GEGXMPostProcessing::getSceneSurface() const
{
    return m_scene ? m_scene->getSurface() : NULL;
}   // getSceneSurface

// ----------------------------------------------------------------------------
void GEGXMPostProcessing::runPass(GEGXMSurface* target,
                                 const SceGxmTexture* source,
                                 SceGxmFragmentProgram* program,
                                 GEGXMProgram* reflection,
                                 const float* texel_uniform)
{
    if (target == NULL || source == NULL || program == NULL)
        return;
    // Each pass is its own scene: on a tile based GPU the previous pass's
    // output does not exist in memory until its scene has ended.
    if (!m_driver->beginSurfaceScene(target, false,
        irr::video::SColor(255, 0, 0, 0)))
        return;

    SceGxmContext* ctx = m_driver->getContext();
    sceGxmSetFrontDepthFunc(ctx, SCE_GXM_DEPTH_FUNC_ALWAYS);
    sceGxmSetFrontDepthWriteEnable(ctx, SCE_GXM_DEPTH_WRITE_DISABLED);
    sceGxmSetCullMode(ctx, SCE_GXM_CULL_NONE);

    sceGxmSetVertexProgram(ctx,
        m_driver->getFullscreenPatchedVertexProgram());
    sceGxmSetFragmentProgram(ctx, program);
    sceGxmSetFragmentTexture(ctx, GTU_MATERIAL_0, source);

    if (texel_uniform != NULL && reflection != NULL)
    {
        // Out of the frame arena and bound explicitly, not
        // sceGxmReserveFragmentDefaultUniformBuffer(): writes into a reserved
        // buffer do not reach the shader here. See GEGXMDrawCall::renderSkyBox().
        void* uniforms = m_driver->allocateFrameMemory(
            reflection->getDefaultUniformBufferSize(), 16);
        if (uniforms != NULL)
        {
            memset(uniforms, 0, reflection->getDefaultUniformBufferSize());
            const SceGxmProgramParameter* p =
                reflection->getParameter("u_texel");
            if (p == NULL)
                p = reflection->getParameter("u_settings");
            if (p != NULL)
                sceGxmSetUniformDataF(uniforms, p, 0, 4, texel_uniform);
            sceGxmSetFragmentDefaultUniformBuffer(ctx, uniforms);
        }
    }
    m_driver->drawFullscreenQuad();
    m_driver->endCurrentScene();
}   // runPass

// ----------------------------------------------------------------------------
void GEGXMPostProcessing::render(GEGXMSurface* display)
{
    if (!m_enabled || display == NULL)
        return;

    const unsigned bw = m_bloom[0]->getSize().Width;
    const unsigned bh = m_bloom[0]->getSize().Height;

    // Bright pass, downsampling to a quarter in each dimension as it goes. The
    // four taps are placed half a source texel apart so the pass is a 2x2 box
    // filter and a threshold at the same time.
    const float bright_uniform[4] =
    {
        0.5f / (float)m_width, 0.5f / (float)m_height, BLOOM_THRESHOLD, 0.0f
    };
    runPass(m_bloom[0]->getSurface(), m_scene->getGXMTexture(),
        m_bright_patched, m_bright_program, bright_uniform);

    // Separable gaussian: horizontal into pong, vertical back into ping.
    const float horizontal[4] = { 1.0f / (float)bw, 0.0f, 0.0f, 0.0f };
    runPass(m_bloom[1]->getSurface(), m_bloom[0]->getGXMTexture(),
        m_blur_patched, m_blur_program, horizontal);
    const float vertical[4] = { 0.0f, 1.0f / (float)bh, 0.0f, 0.0f };
    runPass(m_bloom[0]->getSurface(), m_bloom[1]->getGXMTexture(),
        m_blur_patched, m_blur_program, vertical);

    // Composite into the display buffer. This scene is deliberately left open:
    // the GUI is drawn into it next, which saves resolving the display buffer
    // twice.
    if (!m_driver->beginSurfaceScene(display, false,
        irr::video::SColor(255, 0, 0, 0)))
        return;

    SceGxmContext* ctx = m_driver->getContext();
    sceGxmSetFrontDepthFunc(ctx, SCE_GXM_DEPTH_FUNC_ALWAYS);
    sceGxmSetFrontDepthWriteEnable(ctx, SCE_GXM_DEPTH_WRITE_DISABLED);
    sceGxmSetCullMode(ctx, SCE_GXM_CULL_NONE);
    sceGxmSetVertexProgram(ctx,
        m_driver->getFullscreenPatchedVertexProgram());
    sceGxmSetFragmentProgram(ctx, m_composite_patched);
    sceGxmSetFragmentTexture(ctx, GTU_MATERIAL_0, m_scene->getGXMTexture());
    sceGxmSetFragmentTexture(ctx, GTU_MATERIAL_1,
        m_bloom[0]->getGXMTexture());

    void* uniforms = m_driver->allocateFrameMemory(
        m_composite_program->getDefaultUniformBufferSize(), 16);
    if (uniforms != NULL)
    {
        memset(uniforms, 0, m_composite_program->getDefaultUniformBufferSize());
        const SceGxmProgramParameter* p =
            m_composite_program->getParameter("u_settings");
        if (p != NULL)
        {
            const float settings[4] = { BLOOM_STRENGTH, 0.0f, 0.0f, 0.0f };
            sceGxmSetUniformDataF(uniforms, p, 0, 4, settings);
        }
        sceGxmSetFragmentDefaultUniformBuffer(ctx, uniforms);
    }
    m_driver->drawFullscreenQuad();
}   // render

}   // namespace GE

#endif   // _IRR_COMPILE_WITH_GXM_
