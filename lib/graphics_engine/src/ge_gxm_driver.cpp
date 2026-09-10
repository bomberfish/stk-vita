#include "ge_gxm_driver.hpp"

#ifdef _IRR_COMPILE_WITH_GXM_

#include "ge_gxm_2d_renderer.hpp"
#include "ge_gxm_camera_scene_node.hpp"
#include "ge_gxm_draw_call.hpp"
#include "ge_gxm_environment_map.hpp"
#include "ge_gxm_limits.hpp"
#include "ge_gxm_memory.hpp"
#include "ge_gxm_mesh_cache.hpp"
#include "ge_gxm_post_processing.hpp"
#include "ge_gxm_render_target.hpp"
#include "ge_gxm_shader.hpp"
#include "ge_gxm_texture.hpp"
#include "ge_main.hpp"
#include "ge_material_manager.hpp"
#include "ge_spm.hpp"
#include "ge_spm_buffer.hpp"
#include "ge_texture.hpp"
#include "mini_glm.hpp"

#include "../source/Irrlicht/os.h"

#include <psp2/display.h>
#include <psp2/kernel/sysmem.h>

#include <IAttributes.h>
#include <ISceneManager.h>
#include <IrrlichtDevice.h>
#include <SDL_video.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace GE
{
namespace
{
/** The Vita's panel is a fixed 960x544. 960 would be a legal pitch on its own
 *  (it is both a multiple of 32 pixels for the tile accelerator and 64 byte
 *  aligned), but 1024 is what Sony's samples use and is kept here for that
 *  reason. What matters is only that the colour surface and the display
 *  controller are told the same number - see GEGXMSurface::init(). */
const unsigned DISPLAY_WIDTH = 960;
const unsigned DISPLAY_HEIGHT = 544;
const unsigned DISPLAY_STRIDE = 1024;

struct DisplayQueueData
{
    void* m_address;
    int m_swap_interval;
};

// ----------------------------------------------------------------------------
/** Runs on sceGxm's own display queue thread once a frame's fragment work has
 *  landed in memory. This is the only place that talks to the display
 *  controller. */
void displayQueueCallback(const void* callback_data)
{
    const DisplayQueueData* data = (const DisplayQueueData*)callback_data;

    SceDisplayFrameBuf fb;
    memset(&fb, 0, sizeof(fb));
    fb.size = sizeof(fb);
    fb.base = data->m_address;
    fb.pitch = DISPLAY_STRIDE;
    fb.pixelformat = SCE_DISPLAY_PIXELFORMAT_A8B8G8R8;
    fb.width = DISPLAY_WIDTH;
    fb.height = DISPLAY_HEIGHT;
    sceDisplaySetFrameBuf(&fb, SCE_DISPLAY_SETBUF_NEXTFRAME);

    for (int i = 0; i < data->m_swap_interval; i++)
        sceDisplayWaitVblankStart();
}   // displayQueueCallback

/** Screen space quad covering the whole render target, in normalised device
 *  coordinates with the uv already flipped to framebuffer orientation. */
struct FullscreenVertex
{
    float m_position[2];
    float m_uv[2];
};
}   // namespace

// ============================================================================
GEGXMDriver::GEGXMDriver(const SIrrlichtCreationParameters& params,
                         io::IFileSystem* io, SDL_Window* window,
                         IrrlichtDevice* device)
           : CNullDriver(io, core::dimension2d<u32>(DISPLAY_WIDTH,
                                                    DISPLAY_HEIGHT))
{
    m_params = params;
    m_irrlicht_device = device;
    m_context = NULL;
    m_context_host_mem = NULL;
    m_vdm_ring_buffer = NULL;
    m_vertex_ring_buffer = NULL;
    m_fragment_ring_buffer = NULL;
    m_fragment_usse_ring_buffer = NULL;
    m_display_render_target = NULL;
    m_back_buffer_index = 0;
    m_front_buffer_index = GXM_DISPLAY_BUFFER_COUNT - 1;
    m_notification_base = NULL;
    m_scene_texture = NULL;
    m_current_surface = NULL;
    m_shadow_surface = NULL;
    m_shadow_resolution = 0;
    memset(&m_shadow_texture, 0, sizeof(m_shadow_texture));
    m_in_scene = false;
    m_in_frame = false;
    m_frame_needs_clear = false;
    m_active_rtt = NULL;
    m_clear_color = video::SColor(255, 0, 0, 0);
    m_rtt_clear_color = video::SColor(0, 0, 0, 0);
    m_ambient_light = SColorf(0.0f, 0.0f, 0.0f, 1.0f);
    m_white_texture = NULL;
    m_transparent_texture = NULL;
    m_program_cache = NULL;
    m_fullscreen_vertex = NULL;
    m_clear_fragment = NULL;
    m_fullscreen_vertex_patched = NULL;
    m_clear_fragment_patched = NULL;
    m_fullscreen_vertices = NULL;
    m_fullscreen_indices = NULL;
    m_environment_map = NULL;
    m_post_processing = NULL;
    m_billboard_quad = NULL;
    m_mesh_cache_source = NULL;
    m_mesh_cache = NULL;
    m_polycount = 0;
    m_disable_wait_idle = false;
    m_gxm_initialised = false;

    // The panel is a fixed size, so whatever the config asked for is
    // overridden here rather than in the caller.
    ScreenSize = core::dimension2d<u32>(DISPLAY_WIDTH, DISPLAY_HEIGHT);
    m_render_target_size = ScreenSize;
    m_clip = getFullscreenClip();
    m_viewport = getFullscreenClip();

    DriverAttributes->setAttribute("MaxTextures", (int)GXM_MATERIAL_TEXTURE_COUNT);
    DriverAttributes->setAttribute("MaxSupportedTextures",
        (int)GXM_MATERIAL_TEXTURE_COUNT);
    DriverAttributes->setAttribute("MaxAnisotropy", 1);
    DriverAttributes->setAttribute("MaxTextureSize", 4096);
    DriverAttributes->setAttribute("Version", 100);

    if (!initGXM())
    {
        destroyGXM();
        throw std::runtime_error("Failed to initialise the SceGxm driver");
    }
}   // GEGXMDriver

// ----------------------------------------------------------------------------
GEGXMDriver::~GEGXMDriver()
{
    destroyGXM();
}   // ~GEGXMDriver

// ----------------------------------------------------------------------------
bool GEGXMDriver::initGXM()
{
    // Publish the driver before anything else: GE::getGXMDriver(), the texture
    // factory in ge_texture.cpp, GEGXMMeshCache and GEMaterialManager's XML
    // load all resolve through GE::getDriver().
    GE::setVideoDriver(this);
    GEGXMMemory::init();

    SceGxmInitializeParams init_params;
    memset(&init_params, 0, sizeof(init_params));
    init_params.flags = 0;
    // Two pending swaps is what triple buffering needs: one on screen, one
    // queued, one being drawn into.
    init_params.displayQueueMaxPendingCount = GXM_DISPLAY_BUFFER_COUNT - 1;
    init_params.displayQueueCallback = &displayQueueCallback;
    init_params.displayQueueCallbackDataSize = sizeof(DisplayQueueData);
    init_params.parameterBufferSize = SCE_GXM_DEFAULT_PARAMETER_BUFFER_SIZE;
    int err = sceGxmInitialize(&init_params);
    // SCE_GXM_ERROR_ALREADY_INITIALIZED means something else in the process
    // (SDL's own GXM renderer, for instance) got there first. sceGxm is a
    // process wide singleton, so that is usable - but this driver must then not
    // terminate it on the way out, which is what m_gxm_initialised tracks.
    const bool already = err == (int)SCE_GXM_ERROR_ALREADY_INITIALIZED;
    if (err < 0 && !already)
    {
        char buf[64];
        snprintf(buf, sizeof(buf), "GXM: sceGxmInitialize failed (0x%08x)",
            (unsigned)err);
        irr::os::Printer::log(buf, irr::ELL_ERROR);
        return false;
    }
    m_gxm_initialised = !already;
    m_notification_base = sceGxmGetNotificationRegion();

    if (!createContext())
        return false;
    if (!GEGXMShaderManager::init())
        return false;
    m_program_cache = new GEGXMProgramCache();
    if (!createDisplayBuffers())
        return false;
    if (!createFullscreenQuad())
        return false;

    createPlaceholderTextures();

    // Shadows and post processing are optional; failing to create them costs
    // an effect, not the driver.
    createShadowSurface();
    m_environment_map = new GEGXMEnvironmentMap(this);
    m_post_processing = new GEGXMPostProcessing(this);
    m_post_processing->init(DISPLAY_WIDTH, DISPLAY_HEIGHT);

    GEGXM2dRenderer::init(this);

    // Reads data/shaders/ge_shaders/shader_settings.xml, which is what maps
    // irrlicht material types onto the material names GEGXMDrawCall looks up.
    // It throws if the file is missing, and without it the first 3D draw would
    // throw out of getShader() instead.
    try
    {
        GEMaterialManager::init();
    }
    catch (std::exception& e)
    {
        irr::os::Printer::log("GXM: could not load the material settings",
            e.what(), irr::ELL_ERROR);
        return false;
    }

    unsigned compiled = GEGXMShaderManager::getCompiledCount();
    if (compiled > 0)
    {
        char buf[128];
        snprintf(buf, sizeof(buf), "GXM: compiled %u shaders (the rest came "
            "from the on-disk cache)", compiled);
        irr::os::Printer::log(buf, irr::ELL_INFORMATION);
    }

    // One line saying what this driver actually decided to do. On a Vita the
    // log is the only feedback channel there is, so it is worth being able to
    // tell "PBR off, no shadows" from "PBR on, shadows at 1024" without
    // guessing from the picture.
    {
        char buf[192];
        snprintf(buf, sizeof(buf), "GXM: %ux%u, pbr %s, ibl %s, shadows %s, "
            "post processing %s",
            DISPLAY_WIDTH, DISPLAY_HEIGHT,
            getGEConfig()->m_pbr ? "on" : "off",
            getGEConfig()->m_ibl ? "on" : "off",
            m_shadow_surface != NULL ? "on" : "off",
            m_post_processing->isEnabled() ? "on" : "off");
        irr::os::Printer::log(buf, irr::ELL_INFORMATION);
        if (m_shadow_surface != NULL)
        {
            snprintf(buf, sizeof(buf), "GXM: shadow cascade %ux%u",
                m_shadow_resolution, m_shadow_resolution);
            irr::os::Printer::log(buf, irr::ELL_INFORMATION);
        }
    }
    GEGXMMemory::logStats();
    return true;
}   // initGXM

// ----------------------------------------------------------------------------
bool GEGXMDriver::createContext()
{
    // Sizes well above Sony's defaults: STK submits far more draws per frame
    // than the samples these numbers come from, and running a ring buffer dry
    // stalls the CPU on the GPU. LPDDR is the cheap resource here.
    const size_t VDM_RING_SIZE = 1024 * 1024;
    const size_t VERTEX_RING_SIZE = 16 * 1024 * 1024;
    const size_t FRAGMENT_RING_SIZE = 8 * 1024 * 1024;
    const size_t FRAGMENT_USSE_RING_SIZE = 64 * 1024;

    m_context_host_mem = malloc(SCE_GXM_MINIMUM_CONTEXT_HOST_MEM_SIZE);
    m_vdm_ring_buffer = GEGXMMemory::allocate(GGMP_LPDDR_UNCACHED,
        VDM_RING_SIZE, 4096);
    m_vertex_ring_buffer = GEGXMMemory::allocate(GGMP_LPDDR_UNCACHED,
        VERTEX_RING_SIZE, 4096);
    m_fragment_ring_buffer = GEGXMMemory::allocate(GGMP_LPDDR_UNCACHED,
        FRAGMENT_RING_SIZE, 4096);
    unsigned fragment_usse_offset = 0;
    m_fragment_usse_ring_buffer = GEGXMMemory::allocateFragmentUsse(
        FRAGMENT_USSE_RING_SIZE, &fragment_usse_offset);

    if (m_context_host_mem == NULL || m_vdm_ring_buffer == NULL ||
        m_vertex_ring_buffer == NULL || m_fragment_ring_buffer == NULL ||
        m_fragment_usse_ring_buffer == NULL)
    {
        irr::os::Printer::log("GXM: out of memory for the context ring "
            "buffers", irr::ELL_ERROR);
        return false;
    }

    SceGxmContextParams context_params;
    memset(&context_params, 0, sizeof(context_params));
    context_params.hostMem = m_context_host_mem;
    context_params.hostMemSize = SCE_GXM_MINIMUM_CONTEXT_HOST_MEM_SIZE;
    context_params.vdmRingBufferMem = m_vdm_ring_buffer;
    context_params.vdmRingBufferMemSize = VDM_RING_SIZE;
    context_params.vertexRingBufferMem = m_vertex_ring_buffer;
    context_params.vertexRingBufferMemSize = VERTEX_RING_SIZE;
    context_params.fragmentRingBufferMem = m_fragment_ring_buffer;
    context_params.fragmentRingBufferMemSize = FRAGMENT_RING_SIZE;
    context_params.fragmentUsseRingBufferMem = m_fragment_usse_ring_buffer;
    context_params.fragmentUsseRingBufferMemSize = FRAGMENT_USSE_RING_SIZE;
    context_params.fragmentUsseRingBufferOffset = fragment_usse_offset;

    int err = sceGxmCreateContext(&context_params, &m_context);
    if (err < 0)
    {
        char buf[192];
        if (err == (int)SCE_GXM_ERROR_ALREADY_INITIALIZED)
        {
            // Real hardware allows several contexts, but Vita3K models only
            // one, and vitaGL takes it as soon as anything creates a GL window.
            // Worth naming, because the symptom is otherwise just a driver that
            // refuses to start.
            snprintf(buf, sizeof(buf), "GXM: sceGxmCreateContext reports the "
                "context is already taken - another GXM user (vitaGL, if a GL "
                "window was created) holds it");
        }
        else
        {
            snprintf(buf, sizeof(buf), "GXM: sceGxmCreateContext failed "
                "(0x%08x)", (unsigned)err);
        }
        irr::os::Printer::log(buf, irr::ELL_ERROR);
        m_context = NULL;
        return false;
    }
    return true;
}   // createContext

// ----------------------------------------------------------------------------
bool GEGXMDriver::createDisplayBuffers()
{
    SceGxmRenderTargetParams rt_params;
    memset(&rt_params, 0, sizeof(rt_params));
    rt_params.width = DISPLAY_WIDTH;
    rt_params.height = DISPLAY_HEIGHT;
    // The shadow pass, the post processing chain and the main pass are each a
    // scene, and GXM wants to know the worst case up front so it can size the
    // tiling structures.
    rt_params.scenesPerFrame = 8;
    rt_params.multisampleMode = SCE_GXM_MULTISAMPLE_NONE;
    rt_params.driverMemBlock = -1;
    if (sceGxmCreateRenderTarget(&rt_params, &m_display_render_target) < 0)
    {
        irr::os::Printer::log("GXM: could not create the display render "
            "target", irr::ELL_ERROR);
        return false;
    }

    const size_t buffer_size = (size_t)DISPLAY_STRIDE * DISPLAY_HEIGHT * 4;
    m_display_buffers.resize(GXM_DISPLAY_BUFFER_COUNT);
    m_frame_arenas.resize(GXM_DISPLAY_BUFFER_COUNT);
    for (unsigned i = 0; i < GXM_DISPLAY_BUFFER_COUNT; i++)
    {
        DisplayBuffer& b = m_display_buffers[i];
        // The display controller reads these directly, so they have to be in
        // CDRAM. 256 KiB alignment is what every Sony sample and every homebrew
        // GL implementation ends up handing sceDisplaySetFrameBuf, because they
        // all use a dedicated CDRAM memblock; the headers do not state a
        // requirement, so match the precedent rather than guess lower.
        b.m_data = GEGXMMemory::allocate(GGMP_CDRAM, buffer_size,
            256 * 1024);
        if (b.m_data == NULL)
        {
            irr::os::Printer::log("GXM: out of video memory for the display "
                "buffers", irr::ELL_ERROR);
            return false;
        }
        memset(b.m_data, 0, buffer_size);

        b.m_sync = NULL;
        if (sceGxmSyncObjectCreate(&b.m_sync) < 0)
        {
            irr::os::Printer::log("GXM: sceGxmSyncObjectCreate failed",
                irr::ELL_ERROR);
            return false;
        }

        b.m_surface.reset(new GEGXMSurface());
        // Every display buffer shares the render target created above and,
        // for i > 0, the depth buffer allocated with the first one: only one
        // scene is ever in flight, so one depth buffer is enough for all
        // three colour buffers.
        if (!b.m_surface->init(DISPLAY_WIDTH, DISPLAY_HEIGHT,
            SCE_GXM_COLOR_FORMAT_A8B8G8R8, b.m_data, true,
            i == 0 ? NULL : m_display_buffers[0].m_surface.get(),
            DISPLAY_STRIDE, m_display_render_target))
        {
            return false;
        }

        // 2 MB of scratch per frame covers the instance streams and index
        // runs of a busy race; allocateFrameMemory() logs and drops draws
        // rather than growing it mid-frame, which would risk reusing memory
        // the GPU is reading.
        FrameArena& arena = m_frame_arenas[i];
        arena.m_size = 2 * 1024 * 1024;
        arena.m_data = (uint8_t*)GEGXMMemory::allocate(GGMP_LPDDR_UNCACHED,
            arena.m_size, 64);
        arena.m_used = 0;
        if (arena.m_data == NULL)
        {
            irr::os::Printer::log("GXM: out of memory for the per frame "
                "arenas", irr::ELL_ERROR);
            return false;
        }
    }

    // Put something on screen straight away so the LiveArea splash does not
    // linger while the shaders compile on a cold cache.
    SceDisplayFrameBuf fb;
    memset(&fb, 0, sizeof(fb));
    fb.size = sizeof(fb);
    fb.base = m_display_buffers[0].m_data;
    fb.pitch = DISPLAY_STRIDE;
    fb.pixelformat = SCE_DISPLAY_PIXELFORMAT_A8B8G8R8;
    fb.width = DISPLAY_WIDTH;
    fb.height = DISPLAY_HEIGHT;
    // Checked here, unlike in the display queue callback, because this is the
    // one place a rejected base address (alignment, pitch, format) can be
    // reported. Without it a refused framebuffer looks like a frozen screen
    // with nothing in the log.
    int err = sceDisplaySetFrameBuf(&fb, SCE_DISPLAY_SETBUF_IMMEDIATE);
    if (err < 0)
    {
        char buf[80];
        snprintf(buf, sizeof(buf), "GXM: sceDisplaySetFrameBuf rejected the "
            "display buffer (0x%08x)", (unsigned)err);
        irr::os::Printer::log(buf, irr::ELL_ERROR);
        return false;
    }
    return true;
}   // createDisplayBuffers

// ----------------------------------------------------------------------------
bool GEGXMDriver::createFullscreenQuad()
{
    m_fullscreen_vertex = GEGXMShaderManager::getVertexProgram(
        "fullscreen.vert");
    m_clear_fragment = GEGXMShaderManager::getFragmentProgram("clear.frag");
    if (m_fullscreen_vertex == NULL || m_clear_fragment == NULL)
        return false;

    SceGxmVertexStream stream;
    stream.stride = sizeof(FullscreenVertex);
    stream.indexSource = SCE_GXM_INDEX_SOURCE_INDEX_16BIT;
    const GEGXMAttributeDesc attributes[] =
    {
        { "a_position", 0, offsetof(FullscreenVertex, m_position),
          SCE_GXM_ATTRIBUTE_FORMAT_F32, 2 },
        { "a_uv", 0, offsetof(FullscreenVertex, m_uv),
          SCE_GXM_ATTRIBUTE_FORMAT_F32, 2 },
    };
    GEGXMVertexLayout layout;
    if (!buildVertexLayout(m_fullscreen_vertex, attributes, 2, &stream, 1,
        &layout))
        return false;
    m_fullscreen_vertex_patched = m_program_cache->getVertexProgram(
        m_fullscreen_vertex, layout);
    m_clear_fragment_patched = m_program_cache->getFragmentProgram(
        m_clear_fragment, m_fullscreen_vertex, GEGXMBlendState::opaque());
    if (m_fullscreen_vertex_patched == NULL ||
        m_clear_fragment_patched == NULL)
        return false;

    m_fullscreen_vertices = GEGXMMemory::allocate(GGMP_LPDDR_UNCACHED,
        sizeof(FullscreenVertex) * 4, 16);
    m_fullscreen_indices = GEGXMMemory::allocate(GGMP_LPDDR_UNCACHED,
        sizeof(uint16_t) * 6, 16);
    if (m_fullscreen_vertices == NULL || m_fullscreen_indices == NULL)
        return false;

    // Clip space corners with uv in framebuffer orientation: v = 0 at the top
    // of the screen, which is where clip y = +1 lands under the viewport this
    // driver sets.
    FullscreenVertex* v = (FullscreenVertex*)m_fullscreen_vertices;
    v[0] = { { -1.0f,  1.0f }, { 0.0f, 0.0f } };
    v[1] = { {  1.0f,  1.0f }, { 1.0f, 0.0f } };
    v[2] = { { -1.0f, -1.0f }, { 0.0f, 1.0f } };
    v[3] = { {  1.0f, -1.0f }, { 1.0f, 1.0f } };
    uint16_t* idx = (uint16_t*)m_fullscreen_indices;
    idx[0] = 0; idx[1] = 1; idx[2] = 2;
    idx[3] = 2; idx[4] = 1; idx[5] = 3;
    return true;
}   // createFullscreenQuad

// ----------------------------------------------------------------------------
bool GEGXMDriver::createShadowSurface()
{
    // A single cascade, sized from STK's shadow setting. Two or four cascades
    // would look better but each one is another full geometry pass, and on this
    // GPU the shadow pass is already a meaningful slice of the frame.
    unsigned resolution = getGEConfig()->m_shadow_resolution;
    if (resolution == 0)
        return false;
    // The cascade is sampled as a linear strided texture, so keep it to a power
    // of two within what the hardware and the memory budget allow.
    if (resolution < 512)
        resolution = 512;
    else if (resolution > 2048)
        resolution = 2048;
    m_shadow_resolution = resolution;
    std::unique_ptr<GEGXMSurface> surface(new GEGXMSurface());
    if (!surface->initDepthOnly(m_shadow_resolution, m_shadow_resolution,
        true/*sample_depth*/))
    {
        irr::os::Printer::log("GXM: shadow cascade unavailable, shadows will "
            "be disabled", irr::ELL_WARNING);
        m_shadow_resolution = 0;
        return false;
    }
    // The depth surface doubles as a sampled texture. Nearest filtering is
    // deliberate: with a manual depth comparison, bilinear filtering would
    // interpolate depths rather than shadow test results, which biases the
    // comparison instead of softening it. The 3x3 kernel in the shader is what
    // does the softening.
    if (sceGxmTextureInitLinearStrided(&m_shadow_texture,
        surface->getDepthData(), SCE_GXM_TEXTURE_FORMAT_F32M_R,
        m_shadow_resolution, m_shadow_resolution,
        ((m_shadow_resolution + SCE_GXM_TILE_SIZEX - 1) /
        SCE_GXM_TILE_SIZEX) * SCE_GXM_TILE_SIZEX * 4) < 0)
    {
        irr::os::Printer::log("GXM: could not create a sampled view of the "
            "shadow cascade, shadows will be disabled", irr::ELL_WARNING);
        m_shadow_resolution = 0;
        return false;
    }
    sceGxmTextureSetMinFilter(&m_shadow_texture,
        SCE_GXM_TEXTURE_FILTER_POINT);
    sceGxmTextureSetMagFilter(&m_shadow_texture,
        SCE_GXM_TEXTURE_FILTER_POINT);
    sceGxmTextureSetMipFilter(&m_shadow_texture,
        SCE_GXM_TEXTURE_MIP_FILTER_DISABLED);
    // Clamping matters: a sample taken past the cascade must read the far
    // plane rather than wrapping to the other side of the map and shadowing
    // geometry that nothing is above.
    sceGxmTextureSetUAddrMode(&m_shadow_texture,
        SCE_GXM_TEXTURE_ADDR_CLAMP);
    sceGxmTextureSetVAddrMode(&m_shadow_texture,
        SCE_GXM_TEXTURE_ADDR_CLAMP);

    m_shadow_surface = surface.release();
    return true;
}   // createShadowSurface

// ----------------------------------------------------------------------------
void GEGXMDriver::createPlaceholderTextures()
{
    // A 2x2 opaque white and a 2x2 fully transparent texture. Every sampler a
    // shader declares has to have something bound, so these stand in for
    // material layers a mesh does not provide.
    video::IImage* white = createImage(video::ECF_A8R8G8B8,
        core::dimension2du(2, 2));
    uint32_t* white_data = (uint32_t*)white->lock();
    for (unsigned i = 0; i < 4; i++)
        white_data[i] = 0xFFFFFFFF;
    white->unlock();
    m_white_texture = new GEGXMTexture(white, "ge_gxm_white");

    video::IImage* transparent = createImage(video::ECF_A8R8G8B8,
        core::dimension2du(2, 2));
    uint32_t* transparent_data = (uint32_t*)transparent->lock();
    for (unsigned i = 0; i < 4; i++)
        transparent_data[i] = 0;
    transparent->unlock();
    m_transparent_texture = new GEGXMTexture(transparent,
        "ge_gxm_transparent");
}   // createPlaceholderTextures

// ----------------------------------------------------------------------------
void GEGXMDriver::destroyGXM()
{
    if (m_context != NULL)
    {
        if (m_in_scene)
        {
            sceGxmEndScene(m_context, NULL, NULL);
            m_in_scene = false;
        }
        sceGxmFinish(m_context);
    }
    sceGxmDisplayQueueFinish();

    GEGXM2dRenderer::destroy();
    m_draw_calls_cache.clear();

    delete m_post_processing;
    m_post_processing = NULL;
    delete m_environment_map;
    m_environment_map = NULL;
    delete m_shadow_surface;
    m_shadow_surface = NULL;
    m_scene_surface.reset();
    if (m_scene_texture != NULL)
    {
        m_scene_texture->drop();
        m_scene_texture = NULL;
    }
    // Owned by the scene manager's mesh cache, which the device clears; only
    // the borrowed pointer is dropped here.
    m_billboard_quad = NULL;
    if (m_white_texture != NULL)
    {
        m_white_texture->drop();
        m_white_texture = NULL;
    }
    if (m_transparent_texture != NULL)
    {
        m_transparent_texture->drop();
        m_transparent_texture = NULL;
    }

    delete m_program_cache;
    m_program_cache = NULL;
    GEGXMShaderManager::destroy();

    for (DisplayBuffer& b : m_display_buffers)
    {
        b.m_surface.reset();
        if (b.m_sync != NULL)
            sceGxmSyncObjectDestroy(b.m_sync);
        if (b.m_data != NULL)
            GEGXMMemory::free(b.m_data);
    }
    m_display_buffers.clear();
    for (FrameArena& a : m_frame_arenas)
    {
        if (a.m_data != NULL)
            GEGXMMemory::free(a.m_data);
    }
    m_frame_arenas.clear();

    if (m_display_render_target != NULL)
    {
        sceGxmDestroyRenderTarget(m_display_render_target);
        m_display_render_target = NULL;
    }
    if (m_fullscreen_vertices != NULL)
    {
        GEGXMMemory::free(m_fullscreen_vertices);
        m_fullscreen_vertices = NULL;
    }
    if (m_fullscreen_indices != NULL)
    {
        GEGXMMemory::free(m_fullscreen_indices);
        m_fullscreen_indices = NULL;
    }
    if (m_context != NULL)
    {
        sceGxmDestroyContext(m_context);
        m_context = NULL;
    }
    if (m_fragment_usse_ring_buffer != NULL)
    {
        GEGXMMemory::freeFragmentUsse(m_fragment_usse_ring_buffer);
        m_fragment_usse_ring_buffer = NULL;
    }
    if (m_vdm_ring_buffer != NULL)
        GEGXMMemory::free(m_vdm_ring_buffer);
    if (m_vertex_ring_buffer != NULL)
        GEGXMMemory::free(m_vertex_ring_buffer);
    if (m_fragment_ring_buffer != NULL)
        GEGXMMemory::free(m_fragment_ring_buffer);
    m_vdm_ring_buffer = NULL;
    m_vertex_ring_buffer = NULL;
    m_fragment_ring_buffer = NULL;
    if (m_context_host_mem != NULL)
    {
        free(m_context_host_mem);
        m_context_host_mem = NULL;
    }

    if (m_gxm_initialised)
    {
        GEGXMMemory::destroy();
        sceGxmTerminate();
        m_gxm_initialised = false;
    }
}   // destroyGXM

// ----------------------------------------------------------------------------
bool GEGXMDriver::queryFeature(E_VIDEO_DRIVER_FEATURE feature) const
{
    switch (feature)
    {
    case EVDF_TEXTURE_NPOT:
        // Deliberately false. Saying no makes GE resize every texture to a
        // power of two, which is what lets the upload path use the swizzled
        // layout, and the swizzled layout is the only one that can carry mip
        // maps. Undersampled textures alias badly at 960x544 and the mip chain
        // is also the single biggest bandwidth saving available on this GPU,
        // so the wasted memory on an odd sized image is a good trade.
        return false;
    case EVDF_MULTIPLE_RENDER_TARGETS:
    case EVDF_MRT_BLEND:
    case EVDF_MRT_COLOR_MASK:
    case EVDF_MRT_BLEND_FUNC:
        // The tile accelerator has one colour output. This is the constraint
        // that makes the GXM renderer forward rather than deferred.
        return false;
    case EVDF_GEOMETRY_SHADER:
    case EVDF_OCCLUSION_QUERY:
    case EVDF_TEXTURE_MATRIX:
        return false;
    case EVDF_STENCIL_BUFFER:
        // Supported by the hardware but not wired up: nothing in the GXM
        // pipeline uses stencil, and the depth surfaces are D16 without one.
        return false;
    case EVDF_TEXTURE_COMPRESSED_DXT:
        // UBC1/2/3 are supported by the texture unit, but the upload path only
        // produces uncompressed textures so far.
        return false;
    default:
        return true;
    }
}   // queryFeature

// ----------------------------------------------------------------------------
void GEGXMDriver::waitIdle()
{
    if (m_context == NULL || m_disable_wait_idle)
        return;
    if (m_in_scene)
    {
        // Closing the frame's scene loses its contents - this GPU has no colour
        // force-load, so whatever reopens it starts from undefined tiles. Arm
        // the clear so the rest of the frame composites over a known colour
        // instead of garbage. Reachable because a texture or render target
        // destroyed from inside GUIEngine::render() drains through here.
        if (m_in_frame && m_active_rtt == NULL)
            m_frame_needs_clear = true;
        endCurrentScene();
    }
    sceGxmFinish(m_context);
    sceGxmDisplayQueueFinish();
}   // waitIdle

// ----------------------------------------------------------------------------
void* GEGXMDriver::allocateFrameMemory(size_t size, size_t alignment)
{
    if (m_frame_arenas.empty())
        return NULL;
    FrameArena& arena = m_frame_arenas[m_back_buffer_index];
    size_t offset = (arena.m_used + alignment - 1) & ~(alignment - 1);
    if (offset + size > arena.m_size)
    {
        static bool warned = false;
        if (!warned)
        {
            warned = true;
            irr::os::Printer::log("GXM: the per frame scratch arena is full, "
                "some geometry will be dropped this frame", irr::ELL_WARNING);
        }
        return NULL;
    }
    arena.m_used = offset + size;
    return arena.m_data + offset;
}   // allocateFrameMemory

// ----------------------------------------------------------------------------
void GEGXMDriver::resetFrameMemory()
{
    if (m_frame_arenas.empty())
        return;
    m_frame_arenas[m_back_buffer_index].m_used = 0;
}   // resetFrameMemory

// ----------------------------------------------------------------------------
void GEGXMDriver::applyViewport(const core::rect<s32>& area)
{
    if (m_context == NULL || !m_in_scene)
        return;
    const int surface_w = (int)m_render_target_size.Width;
    const int surface_h = (int)m_render_target_size.Height;
    int x = std::max(0, area.UpperLeftCorner.X);
    int y = std::max(0, area.UpperLeftCorner.Y);
    int w = std::min(area.getWidth(), surface_w - x);
    int h = std::min(area.getHeight(), surface_h - y);
    if (w <= 0 || h <= 0)
        return;

    // X and Y map clip space onto the surface with y growing downwards, which is
    // what a GXM colour surface expects while irrlicht's matrices put y up.
    //
    // Z passes through unchanged (offset 0, scale 1). Irrlicht builds its
    // projections the Direct3D way - buildProjectionMatrixPerspectiveFovLH and
    // buildProjectionMatrixOrthoLH both map the view volume to z in 0..1, not
    // -1..1 - which is exactly the range the depth buffer stores, so no
    // remapping is needed and the whole 16-bit depth buffer is used. (The
    // Vulkan backend's camera node halves it instead, which costs it a bit of
    // precision; there is no reason to copy that here.)
    sceGxmSetViewport(m_context,
        (float)x + (float)w * 0.5f, (float)w * 0.5f,
        (float)y + (float)h * 0.5f, -(float)h * 0.5f,
        0.0f, 1.0f);
    sceGxmSetRegionClip(m_context, SCE_GXM_REGION_CLIP_OUTSIDE, x, y,
        x + w - 1, y + h - 1);
}   // applyViewport

// ----------------------------------------------------------------------------
void GEGXMDriver::setViewPort(const core::rect<s32>& area)
{
    core::rect<s32> vp = area;
    vp.clipAgainst(core::rect<s32>(0, 0, m_render_target_size.Width,
        m_render_target_size.Height));
    if (vp.getWidth() <= 0 || vp.getHeight() <= 0)
        return;
    m_viewport = vp;
    ViewPort = vp;
    // The draw call reads the viewport off the camera rather than off the
    // driver, because with split screen each camera has its own and they are
    // rendered in sequence within one scene. Same forwarding the Vulkan driver
    // does.
    if (m_irrlicht_device != NULL &&
        m_irrlicht_device->getSceneManager() != NULL &&
        m_irrlicht_device->getSceneManager()->getActiveCamera() != NULL)
    {
        GEGXMCameraSceneNode* cam = dynamic_cast<GEGXMCameraSceneNode*>(
            m_irrlicht_device->getSceneManager()->getActiveCamera());
        if (cam != NULL)
            cam->setViewPort(area);
    }
    applyViewport(vp);
}   // setViewPort

// ----------------------------------------------------------------------------
void GEGXMDriver::enableScissorTest(const core::rect<s32>& r)
{
    m_clip = r;
}   // enableScissorTest

// ----------------------------------------------------------------------------
void GEGXMDriver::clearColorSurface(SColor color)
{
    if (!m_in_scene)
        return;

    // Depth needs no clearing: the tile buffer starts at the depth surface's
    // background value at the top of every scene. Colour has to be painted,
    // which on a tile based GPU never leaves the tile and so costs only fill.
    sceGxmSetFrontDepthFunc(m_context, SCE_GXM_DEPTH_FUNC_ALWAYS);
    sceGxmSetFrontDepthWriteEnable(m_context, SCE_GXM_DEPTH_WRITE_DISABLED);
    sceGxmSetCullMode(m_context, SCE_GXM_CULL_NONE);
    // The clear has to cover the whole surface, not the current viewport.
    // Deliberately goes through applyViewport() rather than
    // sceGxmSetDefaultRegionClipAndViewport(): this driver's viewport has a
    // negative Y scale so that irrlicht's OpenGL style projections work, and
    // mixing in the default setup would leave the two out of step.
    applyViewport(core::rect<s32>(0, 0, m_render_target_size.Width,
        m_render_target_size.Height));

    sceGxmSetVertexProgram(m_context, m_fullscreen_vertex_patched);
    sceGxmSetFragmentProgram(m_context, m_clear_fragment_patched);

    void* uniforms = NULL;
    if (sceGxmReserveFragmentDefaultUniformBuffer(m_context, &uniforms) >= 0 &&
        uniforms != NULL)
    {
        const SceGxmProgramParameter* p =
            m_clear_fragment->getParameter("u_color");
        if (p != NULL)
        {
            const float rgba[4] =
            {
                color.getRed() / 255.0f, color.getGreen() / 255.0f,
                color.getBlue() / 255.0f, color.getAlpha() / 255.0f
            };
            sceGxmSetUniformDataF(uniforms, p, 0, 4, rgba);
        }
    }
    drawFullscreenQuad();

    sceGxmSetFrontDepthFunc(m_context, SCE_GXM_DEPTH_FUNC_LESS_EQUAL);
    sceGxmSetFrontDepthWriteEnable(m_context, SCE_GXM_DEPTH_WRITE_ENABLED);
    applyViewport(m_viewport);
}   // clearColorSurface

// ----------------------------------------------------------------------------
void GEGXMDriver::drawFullscreenQuad()
{
    if (!m_in_scene)
        return;
    sceGxmSetVertexStream(m_context, 0, m_fullscreen_vertices);
    sceGxmDraw(m_context, SCE_GXM_PRIMITIVE_TRIANGLES,
        SCE_GXM_INDEX_FORMAT_U16, m_fullscreen_indices, 6);
}   // drawFullscreenQuad

// ----------------------------------------------------------------------------
bool GEGXMDriver::beginSurfaceScene(GEGXMSurface* surface, bool clear,
                                    SColor clear_color)
{
    if (m_context == NULL || surface == NULL)
        return false;
    endCurrentScene();

    SceGxmSyncObject* fragment_sync = NULL;
    SceGxmRenderTarget* render_target = surface->getRenderTarget();
    // When rendering into a display buffer, the GPU has to be held off until
    // that buffer has left the screen; the sync object is what expresses that.
    for (DisplayBuffer& b : m_display_buffers)
    {
        if (b.m_surface.get() == surface)
        {
            fragment_sync = b.m_sync;
            render_target = m_display_render_target;
            break;
        }
    }

    int err = sceGxmBeginScene(m_context, 0, render_target, NULL, NULL,
        fragment_sync, surface->getColorSurface(), surface->getDepthSurface());
    if (err < 0)
    {
        char buf[64];
        snprintf(buf, sizeof(buf), "GXM: sceGxmBeginScene failed (0x%08x)",
            (unsigned)err);
        irr::os::Printer::log(buf, irr::ELL_ERROR);
        return false;
    }
    m_in_scene = true;
    m_current_surface = surface;
    m_render_target_size = surface->getSize();
    m_viewport = core::rect<s32>(0, 0, m_render_target_size.Width,
        m_render_target_size.Height);
    m_clip = m_viewport;

    sceGxmSetTwoSidedEnable(m_context, SCE_GXM_TWO_SIDED_DISABLED);
    sceGxmSetFrontDepthFunc(m_context, SCE_GXM_DEPTH_FUNC_LESS_EQUAL);
    sceGxmSetFrontDepthWriteEnable(m_context, SCE_GXM_DEPTH_WRITE_ENABLED);
    sceGxmSetCullMode(m_context, SCE_GXM_CULL_NONE);
    applyViewport(m_viewport);

    if (clear)
        clearColorSurface(clear_color);
    return true;
}   // beginSurfaceScene

// ----------------------------------------------------------------------------
void GEGXMDriver::endCurrentScene()
{
    if (!m_in_scene)
        return;
    sceGxmEndScene(m_context, NULL, NULL);
    m_in_scene = false;
    m_current_surface = NULL;
    // Do not leave the target size describing a surface that is no longer
    // bound; getCurrentRenderTargetSize() is public and callers outside a scene
    // would otherwise see the shadow cascade's or a render target's dimensions.
    m_render_target_size = ScreenSize;
    m_viewport = getFullscreenClip();
    m_clip = m_viewport;
}   // endCurrentScene

// ----------------------------------------------------------------------------
bool GEGXMDriver::beginScene(bool back_buffer, bool z_buffer, SColor color,
                             const SExposedVideoData& video_data,
                             core::rect<s32>* source_rect)
{
    if (m_context == NULL)
        return false;

    GEMaterialManager::update();
    if (m_billboard_quad == NULL)
        createBillboardQuad();

    m_clear_color = color;
    m_polycount = 0;
    m_in_frame = true;
    m_frame_needs_clear = back_buffer;
    GEGXM2dRenderer::clear();

    // No sceGxmBeginScene here: see ensureFrameScene().
    return true;
}   // beginScene

// ----------------------------------------------------------------------------
bool GEGXMDriver::ensureFrameScene()
{
    if (m_in_scene)
        return true;
    // A render target is active but its scene was closed: reopen that, not the
    // frame, or everything that follows would silently land on screen instead
    // of in the texture.
    if (m_active_rtt != NULL)
    {
        // Cleared, not resumed: a reopened scene starts from undefined tiles on
        // this GPU, so whatever the render target held is gone either way and a
        // clear at least makes the result deterministic. With the colour
        // setRenderTarget() was given, not the frame's - STK's kart previews
        // clear to transparent and composite with the alpha channel.
        return beginSurfaceScene(m_active_rtt->getSurface(), true,
            m_rtt_clear_color);
    }
    if (!m_in_frame || m_display_buffers.empty())
        return false;

    // With post processing on, the 3D scene renders offscreen so the bright
    // pass has something to sample; with it off it goes straight into the
    // display buffer and the whole frame is one scene from clear to swap, which
    // is the cheapest path this GPU has.
    GEGXMSurface* target = m_post_processing->isEnabled() ?
        m_post_processing->getSceneSurface() :
        m_display_buffers[m_back_buffer_index].m_surface.get();
    if (target == NULL)
        target = m_display_buffers[m_back_buffer_index].m_surface.get();

    const bool clear = m_frame_needs_clear;
    m_frame_needs_clear = false;
    return beginSurfaceScene(target, clear, m_clear_color);
}   // ensureFrameScene

// ----------------------------------------------------------------------------
bool GEGXMDriver::endScene()
{
    if (m_context == NULL || !m_in_frame)
        return false;

    GEGXMSurface* display =
        m_display_buffers[m_back_buffer_index].m_surface.get();

    // A render target left bound would send the composite and the GUI into a
    // texture instead of the screen. STK always unbinds, but the frame is over
    // either way, so drop it rather than depend on that.
    if (m_active_rtt != NULL)
    {
        endCurrentScene();
        m_active_rtt = NULL;
    }

    if (m_post_processing->isEnabled())
    {
        // Nothing may have drawn at all - a loading screen, for instance - in
        // which case the offscreen scene still has to be opened and cleared so
        // the bright pass reads the clear colour rather than the previous
        // frame.
        ensureFrameScene();
        // The scene has to reach memory before it can be sampled, so the
        // offscreen scene ends here; the bloom chain and the composite each
        // run as their own scene, the last of which targets the display and is
        // left open for the GUI.
        endCurrentScene();
        m_post_processing->render(display);
    }

    if (!m_in_scene)
    {
        const bool clear = m_frame_needs_clear;
        m_frame_needs_clear = false;
        beginSurfaceScene(display, clear, m_clear_color);
    }

    // The GUI is batched during the frame rather than drawn as it is
    // submitted, so that it lands on top of the composite rather than
    // underneath it.
    GEGXM2dRenderer::render();

    if (!m_in_scene)
    {
        // The display scene could not be opened, so there is nothing to
        // present; queuing the buffer anyway would show whatever it last held.
        m_in_frame = false;
        return false;
    }
    endCurrentScene();

    presentFrame();
    m_in_frame = false;
    PrimitivesDrawn = m_polycount;
    return true;
}   // endScene

// ----------------------------------------------------------------------------
void GEGXMDriver::presentFrame()
{
    DisplayBuffer& back = m_display_buffers[m_back_buffer_index];
    DisplayBuffer& front = m_display_buffers[m_front_buffer_index];

    // Keeps the "application is alive" watchdog happy while a long frame is in
    // flight, which matters on a cold shader cache.
    sceGxmPadHeartbeat(back.m_surface->getColorSurface(), back.m_sync);

    DisplayQueueData data;
    data.m_address = back.m_data;
    data.m_swap_interval = m_params.SwapInterval < 1 ? 0 :
        (m_params.SwapInterval > 2 ? 2 : m_params.SwapInterval);
    // Blocks once the queue is full, which is what paces the CPU to the
    // display rather than letting it run arbitrarily far ahead.
    sceGxmDisplayQueueAddEntry(front.m_sync, back.m_sync, &data);

    m_front_buffer_index = m_back_buffer_index;
    m_back_buffer_index = (m_back_buffer_index + 1) % m_display_buffers.size();
    // Reset the incoming arena here rather than at beginScene, because work is
    // submitted between the two: STK renders its kart previews to textures from
    // GUIEngine::update(), after this frame has been presented, and that work
    // has to allocate from the arena the next frame will use rather than from
    // one the next beginScene would then wipe underneath it. Safe because the
    // display queue is at most two frames deep, so the frame that last used
    // this arena has already been displayed.
    resetFrameMemory();
}   // presentFrame

// ----------------------------------------------------------------------------
bool GEGXMDriver::setRenderTarget(video::ITexture* texture,
                                  bool clear_back_buffer, bool clear_z_buffer,
                                  SColor color)
{
    if (m_context == NULL)
        return false;

    if (texture == NULL)
    {
        if (m_active_rtt == NULL)
            return true;
        m_active_rtt = NULL;
        // Back to whatever the frame was targeting. Ending the render target's
        // scene is what flushes its tiles to memory and makes it samplable.
        endCurrentScene();
        // Deliberately does not reopen the frame scene: ensureFrameScene() will
        // do that, with the clear still pending if nothing has drawn yet. That
        // matters because STK renders its kart previews to textures before the
        // frame's own content, and reopening-then-clearing here would be a
        // wasted full screen store and load.
        return true;
    }

    GEGXMFBOTexture* rtt = dynamic_cast<GEGXMFBOTexture*>(texture);
    if (rtt == NULL || !rtt->isValid())
        return false;

    // Switching away from a frame scene that already holds content destroys it:
    // this GPU has no colour force-load, so a reopened scene starts from
    // undefined tiles rather than from what was in the surface. Force a clear
    // when the frame scene comes back, which costs the frame's 3D content but
    // shows a clean image instead of garbage.
    if (m_in_scene && m_in_frame && m_active_rtt == NULL)
    {
        m_frame_needs_clear = true;
        static bool warned = false;
        if (!warned)
        {
            warned = true;
            irr::os::Printer::log("GXM: a render target was bound after the "
                "frame had already been drawn into; that frame's contents "
                "cannot be preserved on this GPU and will be re-cleared",
                irr::ELL_WARNING);
        }
    }
    m_active_rtt = rtt;
    m_rtt_clear_color = color;
    // Depth is cleared implicitly by starting a scene, which is why
    // clear_z_buffer is not consulted here.
    return beginSurfaceScene(rtt->getSurface(), clear_back_buffer, color);
}   // setRenderTarget

// ----------------------------------------------------------------------------
ITexture* GEGXMDriver::addRenderTargetTexture(
                                        const core::dimension2d<u32>& size,
                                        const io::path& name,
                                        const ECOLOR_FORMAT format,
                                        const bool use_stencil)
{
    GEGXMFBOTexture* rtt = new GEGXMFBOTexture(name.c_str(), size.Width,
        size.Height, m_display_buffers.empty() ? NULL :
        m_display_buffers[0].m_surface.get());
    if (!rtt->isValid())
    {
        rtt->drop();
        return NULL;
    }
    // Deliberately not added to the texture cache: STK's RenderTarget owns it
    // and drops it, matching what the Vulkan driver does.
    return rtt;
}   // addRenderTargetTexture

// ----------------------------------------------------------------------------
IImage* GEGXMDriver::createScreenShot(video::ECOLOR_FORMAT format,
                                      video::E_RENDER_TARGET target)
{
    if (m_display_buffers.empty())
        return NULL;
    // Everything queued has to have landed before the buffer is read, and the
    // most recently completed frame is the front buffer, not the back one.
    waitIdle();

    const DisplayBuffer& front = m_display_buffers[m_front_buffer_index];
    if (front.m_data == NULL)
        return NULL;

    IImage* image = createImage(video::ECF_A8R8G8B8,
        core::dimension2du(DISPLAY_WIDTH, DISPLAY_HEIGHT));
    if (image == NULL)
        return NULL;

    uint8_t* dest = (uint8_t*)image->lock();
    const uint8_t* src = (const uint8_t*)front.m_data;
    for (unsigned y = 0; y < DISPLAY_HEIGHT; y++)
    {
        const uint8_t* s = src + (size_t)y * DISPLAY_STRIDE * 4;
        uint8_t* d = dest + (size_t)y * DISPLAY_WIDTH * 4;
        for (unsigned x = 0; x < DISPLAY_WIDTH; x++)
        {
            // Display memory is R, G, B, A; irrlicht wants B, G, R, A.
            d[x * 4 + 0] = s[x * 4 + 2];
            d[x * 4 + 1] = s[x * 4 + 1];
            d[x * 4 + 2] = s[x * 4 + 0];
            d[x * 4 + 3] = 255;
        }
    }
    image->unlock();
    return image;
}   // createScreenShot

// ----------------------------------------------------------------------------
void GEGXMDriver::OnResize(const core::dimension2d<u32>& size)
{
    // The panel cannot change size, so this only ever arrives as a spurious
    // SDL event; accepting it would desynchronise every surface from the
    // display buffers.
    CNullDriver::OnResize(ScreenSize);
}   // OnResize

// ----------------------------------------------------------------------------
void GEGXMDriver::setFog(SColor color, E_FOG_TYPE fog_type, f32 start,
                         f32 end, f32 density, bool pixel_fog, bool range_fog)
{
    CNullDriver::setFog(color, fog_type, start, end, density, pixel_fog,
        range_fog);
}   // setFog

// ----------------------------------------------------------------------------
const SceGxmTexture* GEGXMDriver::getShadowTexture() const
{
    return m_shadow_surface == NULL ? NULL : &m_shadow_texture;
}   // getShadowTexture

// ----------------------------------------------------------------------------
/** The billboard and particle paths draw a unit quad through the same mesh
 *  pipeline as everything else. It cannot be built in the constructor because
 *  it has to be registered with the scene manager's mesh cache, and the scene
 *  manager is created after the driver. */
void GEGXMDriver::createBillboardQuad()
{
    GEGXMMeshCache* cache = getGXMMeshCache();
    if (cache == NULL)
        return;

    m_billboard_quad = new GESPM();
    GESPMBuffer* buffer = new GESPMBuffer();
    /* (-1, 1, 0) 2--1 (1, 1, 0)
                  |\ |
                  | \|
      (-1,-1, 0) 3--0 (1,-1, 0) */
    const short one_hf = 15360;   // 1.0 as a half float
    video::S3DVertexSkinnedMesh sp;
    sp.m_position = core::vector3df(1, -1, 0);
    sp.m_normal = MiniGLM::compressVector3(core::vector3df(0, 0, 1));
    sp.m_color = video::SColor((uint32_t)-1);
    sp.m_all_uvs[0] = one_hf;
    sp.m_all_uvs[1] = one_hf;
    buffer->getVerticesVector().push_back(sp);

    sp.m_position = core::vector3df(1, 1, 0);
    sp.m_all_uvs[0] = one_hf;
    sp.m_all_uvs[1] = 0;
    buffer->getVerticesVector().push_back(sp);

    sp.m_position = core::vector3df(-1, 1, 0);
    sp.m_all_uvs[0] = 0;
    sp.m_all_uvs[1] = 0;
    buffer->getVerticesVector().push_back(sp);

    sp.m_position = core::vector3df(-1, -1, 0);
    sp.m_all_uvs[0] = 0;
    sp.m_all_uvs[1] = one_hf;
    buffer->getVerticesVector().push_back(sp);

    const uint16_t indices[6] = { 2, 1, 0, 2, 0, 3 };
    for (unsigned i = 0; i < 6; i++)
        buffer->getIndicesVector().push_back(indices[i]);
    buffer->recalculateBoundingBox();
    m_billboard_quad->addMeshBuffer(buffer);
    m_billboard_quad->finalize();

    char name[32];
    snprintf(name, sizeof(name), "%llx",
        (unsigned long long)(uintptr_t)m_billboard_quad);
    cache->addMesh(name, m_billboard_quad);
    // The cache grabbed it; the driver keeps a borrowed pointer.
    m_billboard_quad->drop();
}   // createBillboardQuad

// ----------------------------------------------------------------------------
GEGXMMeshCache* GEGXMDriver::getGXMMeshCache() const
{
    if (m_irrlicht_device == NULL ||
        m_irrlicht_device->getSceneManager() == NULL)
        return NULL;
    irr::scene::IMeshCache* cache =
        m_irrlicht_device->getSceneManager()->getMeshCache();
    // Called once per draw batch, so the dynamic_cast is memoised against the
    // cache pointer rather than repeated a few hundred times a frame.
    if (cache != m_mesh_cache_source)
    {
        m_mesh_cache_source = cache;
        m_mesh_cache = dynamic_cast<GEGXMMeshCache*>(cache);
    }
    return m_mesh_cache;
}   // getGXMMeshCache

// ----------------------------------------------------------------------------
void GEGXMDriver::updateDriver(bool scale_changed, bool pbr_changed,
                               bool ibl_changed)
{
    waitIdle();
    // Every draw call holds patched programs that were built for the old
    // material set. The cached ones are simply dropped; the live ones, which the
    // scene manager owns, notice the generation change and rebuild.
    clearDrawCallsCache();
    GEGXMShaderManager::invalidateGeneration();
    GEMaterialManager::update();

    // The cascade is sized from the config, so a changed shadow setting means
    // rebuilding the surface - and turning shadows off or on changes which
    // shader variants the draw calls need, which the generation bump above
    // already forces.
    const unsigned wanted = getGEConfig()->m_shadow_resolution;
    const unsigned clamped = wanted == 0 ? 0u :
        (wanted < 512u ? 512u : (wanted > 2048u ? 2048u : wanted));
    if (clamped != m_shadow_resolution)
    {
        delete m_shadow_surface;
        m_shadow_surface = NULL;
        memset(&m_shadow_texture, 0, sizeof(m_shadow_texture));
        m_shadow_resolution = 0;
        createShadowSurface();
    }

    if (m_post_processing != NULL)
        m_post_processing->init(DISPLAY_WIDTH, DISPLAY_HEIGHT);
    if (ibl_changed && m_environment_map != NULL)
        m_environment_map->invalidate();
}   // updateDriver

// ----------------------------------------------------------------------------
void GEGXMDriver::clearDrawCallsCache()
{
    m_draw_calls_cache.clear();
}   // clearDrawCallsCache

// ----------------------------------------------------------------------------
void GEGXMDriver::addDrawCallToCache(std::unique_ptr<GEGXMDrawCall>& dc)
{
    m_draw_calls_cache.push_back(std::move(dc));
}   // addDrawCallToCache

// ----------------------------------------------------------------------------
std::unique_ptr<GEGXMDrawCall> GEGXMDriver::getDrawCallFromCache()
{
    if (m_draw_calls_cache.empty())
        return std::unique_ptr<GEGXMDrawCall>(new GEGXMDrawCall());
    std::unique_ptr<GEGXMDrawCall> dc = std::move(m_draw_calls_cache.back());
    m_draw_calls_cache.pop_back();
    return dc;
}   // getDrawCallFromCache

// ============================================================================
// 2D drawing. Every entry point funnels into the batching 2D renderer rather
// than issuing a draw, so that the GUI can be replayed after post processing.
// ----------------------------------------------------------------------------
void GEGXMDriver::draw2DVertexPrimitiveList(const void* vertices,
    u32 vertex_count, const void* index_list, u32 primitive_count,
    E_VERTEX_TYPE v_type, scene::E_PRIMITIVE_TYPE p_type, E_INDEX_TYPE i_type)
{
    if (v_type != EVT_STANDARD || i_type != EIT_16BIT ||
        p_type != scene::EPT_TRIANGLES)
        return;
    GEGXM2dRenderer::addVerticesIndices((S3DVertex*)vertices, vertex_count,
        (uint16_t*)index_list, primitive_count * 3, Material.getTexture(0));
}   // draw2DVertexPrimitiveList

// ----------------------------------------------------------------------------
void GEGXMDriver::draw2DImage(const video::ITexture* texture,
                              const core::position2d<s32>& dest_pos,
                              const core::rect<s32>& source_rect,
                              const core::rect<s32>* clip_rect, SColor color,
                              bool use_alpha_channel_of_texture)
{
    if (texture == NULL)
        return;
    core::rect<s32> dest_rect(dest_pos,
        core::dimension2d<s32>(source_rect.getWidth(),
        source_rect.getHeight()));
    const SColor colors[4] = { color, color, color, color };
    draw2DImage(texture, dest_rect, source_rect, clip_rect, colors,
        use_alpha_channel_of_texture);
}   // draw2DImage

// ----------------------------------------------------------------------------
void GEGXMDriver::draw2DImage(const video::ITexture* texture,
                              const core::rect<s32>& dest_rect,
                              const core::rect<s32>& source_rect,
                              const core::rect<s32>* clip_rect,
                              const video::SColor* const colors,
                              bool use_alpha_channel_of_texture)
{
    if (texture == NULL)
        return;
    GEGXM2dRenderer::addQuad(texture, dest_rect, source_rect, clip_rect,
        colors);
}   // draw2DImage

// ----------------------------------------------------------------------------
void GEGXMDriver::draw2DImageBatch(const video::ITexture* texture,
    const core::array<core::position2d<s32> >& positions,
    const core::array<core::rect<s32> >& source_rects,
    const core::rect<s32>* clip_rect, SColor color,
    bool use_alpha_channel_of_texture)
{
    const u32 count = std::min(positions.size(), source_rects.size());
    for (u32 i = 0; i < count; i++)
    {
        draw2DImage(texture, positions[i], source_rects[i], clip_rect, color,
            use_alpha_channel_of_texture);
    }
}   // draw2DImageBatch

// ----------------------------------------------------------------------------
void GEGXMDriver::draw2DRectangle(const core::rect<s32>& pos,
                                  SColor color_left_up, SColor color_right_up,
                                  SColor color_left_down,
                                  SColor color_right_down,
                                  const core::rect<s32>* clip)
{
    // The white placeholder makes an untextured gradient just another textured
    // quad, so the 2D renderer never has to switch programs.
    const SColor colors[4] =
    {
        color_left_up, color_left_down, color_right_down, color_right_up
    };
    GEGXM2dRenderer::addQuad(m_white_texture, pos, core::recti(0, 0, 2, 2),
        clip, colors);
}   // draw2DRectangle

// ----------------------------------------------------------------------------
void GEGXMDriver::draw2DLine(const core::position2d<s32>& start,
                             const core::position2d<s32>& end, SColor color)
{
    // Drawn as a one pixel quad: the 2D path only knows how to emit triangles,
    // and lines are rare enough (debug overlays) not to deserve a program of
    // their own.
    core::rect<s32> r(std::min(start.X, end.X), std::min(start.Y, end.Y),
        std::max(start.X, end.X) + 1, std::max(start.Y, end.Y) + 1);
    const SColor colors[4] = { color, color, color, color };
    GEGXM2dRenderer::addQuad(m_white_texture, r, core::recti(0, 0, 2, 2), NULL,
        colors);
}   // draw2DLine

}   // namespace GE

// ============================================================================
namespace irr
{
namespace video
{
// ----------------------------------------------------------------------------
IVideoDriver* createGXMDriver(const SIrrlichtCreationParameters& params,
                              io::IFileSystem* io, SDL_Window* win,
                              IrrlichtDevice* device)
{
    return new GE::GEGXMDriver(params, io, win, device);
}   // createGXMDriver
}   // namespace video
}   // namespace irr

#endif   // _IRR_COMPILE_WITH_GXM_
