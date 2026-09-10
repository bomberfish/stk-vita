#ifndef HEADER_GE_GXM_POST_PROCESSING_HPP
#define HEADER_GE_GXM_POST_PROCESSING_HPP

#include "IrrCompileConfig.h"

#ifdef _IRR_COMPILE_WITH_GXM_

#include <psp2/gxm.h>

#include <memory>

namespace GE
{
class GEGXMDriver;
class GEGXMFBOTexture;
class GEGXMProgram;
class GEGXMSurface;

/** Bloom and the final composite.
 *
 *  Only built when the config asks for bloom, because it is not free in the way
 *  the rest of the pipeline is. Without it the 3D scene renders straight into
 *  the display buffer and the whole frame is one scene, so colour never leaves
 *  the tile buffer. Turning bloom on means the scene has to be resolved to
 *  memory so it can be sampled, which on a tile based GPU is the expensive part
 *  - more so than the blur passes themselves.
 *
 *  The chain is deliberately shallow for that reason: a combined
 *  downsample-and-threshold pass to quarter resolution, one separable gaussian
 *  there, and a composite that adds it back while writing to the display. Four
 *  scenes total, three of them at a sixteenth of the pixel count. */
class GEGXMPostProcessing
{
private:
    GEGXMDriver* m_driver;

    /** Full resolution target the 3D scene renders into when this is on. */
    std::unique_ptr<GEGXMFBOTexture> m_scene;

    /** Quarter resolution ping and pong buffers for the separable blur. */
    std::unique_ptr<GEGXMFBOTexture> m_bloom[2];

    GEGXMProgram* m_bright_program;

    GEGXMProgram* m_blur_program;

    GEGXMProgram* m_composite_program;

    SceGxmFragmentProgram* m_bright_patched;

    SceGxmFragmentProgram* m_blur_patched;

    SceGxmFragmentProgram* m_composite_patched;

    unsigned m_width, m_height;

    bool m_enabled;

    bool m_bloom_enabled;

    // ------------------------------------------------------------------------
    /** Runs one fullscreen fragment pass from \p source into \p target. */
    void runPass(GEGXMSurface* target, const SceGxmTexture* source,
                 SceGxmFragmentProgram* program, GEGXMProgram* reflection,
                 const float* texel_uniform);

public:
    // ------------------------------------------------------------------------
    GEGXMPostProcessing(GEGXMDriver* driver);
    // ------------------------------------------------------------------------
    ~GEGXMPostProcessing();
    // ------------------------------------------------------------------------
    /** Creates or tears down the offscreen targets to match the current
     *  config. Safe to call repeatedly. */
    void init(unsigned width, unsigned height);
    // ------------------------------------------------------------------------
    void destroy();
    // ------------------------------------------------------------------------
    /** True when the 3D scene has to render offscreen. */
    bool isEnabled() const                                { return m_enabled; }
    // ------------------------------------------------------------------------
    GEGXMSurface* getSceneSurface() const;
    // ------------------------------------------------------------------------
    /** Runs the chain and composites into \p display, leaving that scene open
     *  so the GUI can be drawn into it. */
    void render(GEGXMSurface* display);
};   // GEGXMPostProcessing

}   // namespace GE

#endif   // _IRR_COMPILE_WITH_GXM_

#endif
