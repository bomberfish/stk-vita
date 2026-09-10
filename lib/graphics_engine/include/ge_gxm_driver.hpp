#ifndef HEADER_GE_GXM_DRIVER_HPP
#define HEADER_GE_GXM_DRIVER_HPP

#include "IrrCompileConfig.h"

#ifdef _IRR_COMPILE_WITH_GXM_

#include <psp2/gxm.h>

#include "SDL_video.h"

#include "../source/Irrlicht/CNullDriver.h"
#include "SIrrCreationParameters.h"
#include "SColor.h"

#include <array>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

using namespace irr;
using namespace video;

namespace irr
{
    namespace scene { class IMeshCache; }
}

namespace GE
{
class GEGXMDrawCall;
class GEGXMEnvironmentMap;
class GEGXMFBOTexture;
class GEGXMMeshCache;
class GEGXMPostProcessing;
class GEGXMProgram;
class GEGXMProgramCache;
class GEGXMSurface;
class GESPM;

/** SuperTuxKart's native PlayStation Vita renderer.
 *
 *  The Vita's GPU is an SGX543MP4+, a tile based deferred rasteriser reached
 *  through libgxm. It is a very different machine from the ones the other STK
 *  backends target, and three of its limits decide the whole shape of this
 *  renderer:
 *
 *  - There are no multiple render targets. sceGxmBeginScene() takes one colour
 *    surface. A deferred G-buffer pass simply cannot be expressed, so this is a
 *    forward renderer: one geometry pass that lights as it shades, rather than
 *    the Vulkan backend's option of writing albedo and normals and lighting
 *    afterwards.
 *  - There are no compute shaders and no storage buffers. Per object data
 *    reaches the vertex shader through a vertex stream indexed by instance
 *    rather than a storage buffer, and the environment map prefiltering that
 *    the Vulkan backend does in compute runs as fullscreen fragment passes.
 *  - Uniforms live in a "default uniform buffer" that is DMA'd into a few
 *    hundred registers before each shader runs, so the per draw uniform budget
 *    is small and fixed. See ge_gxm_limits.hpp.
 *
 *  What it has instead is a large on-chip tile buffer. Depth is never written
 *  to memory at all in the main pass, and the whole scene is one
 *  begin/end scene pair so intermediate colour never round trips either. That
 *  is where the bandwidth for a PBR forward pass at 960x544 comes from.
 *
 *  The scene graph, mesh format, material definitions, culling and texture
 *  loading are all shared with the Vulkan backend through the rest of
 *  graphics_engine; only the GPU facing half is new. */
class GEGXMDriver : public video::CNullDriver
{
public:
    // ------------------------------------------------------------------------
    GEGXMDriver(const SIrrlichtCreationParameters& params,
                io::IFileSystem* io, SDL_Window* window,
                IrrlichtDevice* device);
    // ------------------------------------------------------------------------
    virtual ~GEGXMDriver();
    // ------------------------------------------------------------------------
    virtual bool beginScene(bool back_buffer = true, bool z_buffer = true,
        SColor color = SColor(255, 0, 0, 0),
        const SExposedVideoData& video_data = SExposedVideoData(),
        core::rect<s32>* source_rect = 0);
    // ------------------------------------------------------------------------
    virtual bool endScene();
    // ------------------------------------------------------------------------
    virtual bool queryFeature(E_VIDEO_DRIVER_FEATURE feature) const;
    // ------------------------------------------------------------------------
    virtual void setTransform(E_TRANSFORMATION_STATE state,
                              const core::matrix4& mat) {}
    // ------------------------------------------------------------------------
    virtual void setMaterial(const SMaterial& material) { Material = material; }
    // ------------------------------------------------------------------------
    virtual bool setRenderTarget(video::ITexture* texture,
        bool clear_back_buffer = true, bool clear_z_buffer = true,
        SColor color = video::SColor(0, 0, 0, 0));
    // ------------------------------------------------------------------------
    virtual bool setRenderTarget(const core::array<video::IRenderTarget>& texture,
        bool clear_back_buffer = true, bool clear_z_buffer = true,
        SColor color = video::SColor(0, 0, 0, 0)) { return false; }
    // ------------------------------------------------------------------------
    virtual void setViewPort(const core::rect<s32>& area);
    // ------------------------------------------------------------------------
    virtual bool updateHardwareBuffer(SHWBufferLink* hw_buffer)
                                                            { return false; }
    // ------------------------------------------------------------------------
    virtual SHWBufferLink* createHardwareBuffer(const scene::IMeshBuffer* mb)
                                                             { return NULL; }
    // ------------------------------------------------------------------------
    virtual void deleteHardwareBuffer(SHWBufferLink* hw_buffer)              {}
    // ------------------------------------------------------------------------
    virtual void drawHardwareBuffer(SHWBufferLink* hw_buffer)                {}
    // ------------------------------------------------------------------------
    virtual void addOcclusionQuery(scene::ISceneNode* node,
                                   const scene::IMesh* mesh = 0)             {}
    // ------------------------------------------------------------------------
    virtual void removeOcclusionQuery(scene::ISceneNode* node)               {}
    // ------------------------------------------------------------------------
    virtual void runOcclusionQuery(scene::ISceneNode* node,
                                   bool visible = false)                     {}
    // ------------------------------------------------------------------------
    virtual void updateOcclusionQuery(scene::ISceneNode* node,
                                      bool block = true)                     {}
    // ------------------------------------------------------------------------
    virtual u32 getOcclusionQueryResult(scene::ISceneNode* node) const
                                                                { return 0; }
    // ------------------------------------------------------------------------
    virtual void drawVertexPrimitiveList(const void* vertices, u32 vertex_count,
        const void* index_list, u32 primitive_count, E_VERTEX_TYPE v_type,
        scene::E_PRIMITIVE_TYPE p_type, E_INDEX_TYPE i_type)                 {}
    // ------------------------------------------------------------------------
    virtual void draw2DVertexPrimitiveList(const void* vertices,
        u32 vertex_count, const void* index_list, u32 primitive_count,
        E_VERTEX_TYPE v_type, scene::E_PRIMITIVE_TYPE p_type,
        E_INDEX_TYPE i_type);
    // ------------------------------------------------------------------------
    virtual void draw2DImage(const video::ITexture* texture,
        const core::position2d<s32>& dest_pos,
        const core::rect<s32>& source_rect,
        const core::rect<s32>* clip_rect = 0,
        SColor color = SColor(255, 255, 255, 255),
        bool use_alpha_channel_of_texture = false);
    // ------------------------------------------------------------------------
    virtual void draw2DImage(const video::ITexture* texture,
        const core::rect<s32>& dest_rect, const core::rect<s32>& source_rect,
        const core::rect<s32>* clip_rect = 0,
        const video::SColor* const colors = 0,
        bool use_alpha_channel_of_texture = false);
    // ------------------------------------------------------------------------
    virtual void draw2DImageBatch(const video::ITexture* texture,
        const core::array<core::position2d<s32> >& positions,
        const core::array<core::rect<s32> >& source_rects,
        const core::rect<s32>* clip_rect = 0,
        SColor color = SColor(255, 255, 255, 255),
        bool use_alpha_channel_of_texture = false);
    // ------------------------------------------------------------------------
    virtual void draw2DRectangle(const core::rect<s32>& pos,
        SColor color_left_up, SColor color_right_up, SColor color_left_down,
        SColor color_right_down, const core::rect<s32>* clip);
    // ------------------------------------------------------------------------
    virtual void draw2DLine(const core::position2d<s32>& start,
        const core::position2d<s32>& end,
        SColor color = SColor(255, 255, 255, 255));
    // ------------------------------------------------------------------------
    virtual void drawPixel(u32 x, u32 y, const SColor& color)                {}
    // ------------------------------------------------------------------------
    virtual void draw3DLine(const core::vector3df& start,
        const core::vector3df& end,
        SColor color = SColor(255, 255, 255, 255))                           {}
    // ------------------------------------------------------------------------
    virtual const wchar_t* getName() const              { return L"SceGxm"; }
    // ------------------------------------------------------------------------
    virtual void deleteAllDynamicLights()                                    {}
    // ------------------------------------------------------------------------
    virtual s32 addDynamicLight(const SLight& light)           { return -1; }
    // ------------------------------------------------------------------------
    virtual void turnLightOn(s32 light_index, bool turn_on)                  {}
    // ------------------------------------------------------------------------
    virtual u32 getMaximalDynamicLightAmount() const     { return (u32)-1; }
    // ------------------------------------------------------------------------
    /** CNullDriver discards the ambient colour, so it is kept here: the
     *  forward PBR pass needs it as a uniform every frame. */
    virtual void setAmbientLight(const SColorf& color)
    {
        m_ambient_light = color;
        CNullDriver::setAmbientLight(color);
    }
    // ------------------------------------------------------------------------
    const SColorf& getAmbientLightColor() const     { return m_ambient_light; }
    // ------------------------------------------------------------------------
    /** CNullDriver keeps the fog state protected; the fragment uniform block
     *  needs it, so it is republished here. */
    const SColor& getFogColor() const                   { return FogColor; }
    // ------------------------------------------------------------------------
    f32 getFogDensity() const                         { return FogDensity; }
    // ------------------------------------------------------------------------
    virtual void drawStencilShadowVolume(
        const core::array<core::vector3df>& triangles, bool zfail = true,
        u32 debug_data_visible = 0)                                          {}
    // ------------------------------------------------------------------------
    virtual void drawStencilShadow(bool clear_stencil_buffer = false,
        video::SColor left_up_edge = video::SColor(0, 0, 0, 0),
        video::SColor right_up_edge = video::SColor(0, 0, 0, 0),
        video::SColor left_down_edge = video::SColor(0, 0, 0, 0),
        video::SColor right_down_edge = video::SColor(0, 0, 0, 0))           {}
    // ------------------------------------------------------------------------
    virtual u32 getMaximalPrimitiveCount() const         { return (u32)-1; }
    // ------------------------------------------------------------------------
    virtual void setTextureCreationFlag(E_TEXTURE_CREATION_FLAG flag,
                                        bool enabled)                        {}
    // ------------------------------------------------------------------------
    virtual void setFog(SColor color, E_FOG_TYPE fog_type, f32 start, f32 end,
                        f32 density, bool pixel_fog, bool range_fog);
    // ------------------------------------------------------------------------
    virtual void OnResize(const core::dimension2d<u32>& size);
    // ------------------------------------------------------------------------
    virtual E_DRIVER_TYPE getDriverType() const { return video::EDT_GXM; }
    // ------------------------------------------------------------------------
    virtual const core::matrix4& getTransform(E_TRANSFORMATION_STATE state) const
    {
        static core::matrix4 unused;
        return unused;
    }
    // ------------------------------------------------------------------------
    virtual ITexture* addRenderTargetTexture(
        const core::dimension2d<u32>& size, const io::path& name,
        const ECOLOR_FORMAT format = ECF_UNKNOWN,
        const bool use_stencil = false);
    // ------------------------------------------------------------------------
    virtual void clearZBuffer()                                              {}
    // ------------------------------------------------------------------------
    virtual IImage* createScreenShot(
        video::ECOLOR_FORMAT format = video::ECF_UNKNOWN,
        video::E_RENDER_TARGET target = video::ERT_FRAME_BUFFER);
    // ------------------------------------------------------------------------
    virtual bool setClipPlane(u32 index, const core::plane3df& plane,
                              bool enable = false)              { return true; }
    // ------------------------------------------------------------------------
    virtual void enableClipPlane(u32 index, bool enable)                     {}
    // ------------------------------------------------------------------------
    virtual core::stringc getVendorInfo()             { return "ImgTec"; }
    // ------------------------------------------------------------------------
    virtual void enableMaterial2D(bool enable = true)                        {}
    // ------------------------------------------------------------------------
    virtual bool checkDriverReset()                          { return false; }
    // ------------------------------------------------------------------------
    virtual ECOLOR_FORMAT getColorFormat() const      { return ECF_A8R8G8B8; }
    // ------------------------------------------------------------------------
    virtual core::dimension2du getMaxTextureSize() const
                                       { return core::dimension2du(4096, 4096); }
    // ------------------------------------------------------------------------
    virtual void enableScissorTest(const core::rect<s32>& r);
    // ------------------------------------------------------------------------
    virtual void disableScissorTest()  { enableScissorTest(getFullscreenClip()); }
    // ------------------------------------------------------------------------
    core::rect<s32> getFullscreenClip() const
    {
        return core::rect<s32>(0, 0, ScreenSize.Width, ScreenSize.Height);
    }
    // ------------------------------------------------------------------------
    virtual const core::dimension2d<u32>& getCurrentRenderTargetSize() const
                                              { return m_render_target_size; }
    // ========================================================================
    // GXM specific interface, used by the rest of the GXM backend.
    // ------------------------------------------------------------------------
    /** Tears the GPU down. Separate from the destructor because the SDL window
     *  has to still exist while it happens, which is the same reason
     *  GEVulkanDriver has destroyVulkan(). */
    void destroyGXM();
    // ------------------------------------------------------------------------
    SceGxmContext* getContext() const                     { return m_context; }
    // ------------------------------------------------------------------------
    GEGXMProgramCache* getProgramCache() const      { return m_program_cache; }
    // ------------------------------------------------------------------------
    /** The surface the current sceGxmBeginScene() is rendering into. */
    GEGXMSurface* getCurrentSurface() const       { return m_current_surface; }
    // ------------------------------------------------------------------------
    bool isInScene() const                               { return m_in_scene; }
    // ------------------------------------------------------------------------
    /** True between setRenderTarget(texture) and setRenderTarget(NULL). The
     *  draw call skips its shadow pass while this holds: the shadow pass needs a
     *  scene of its own, which would tear down the render target's scene, and
     *  the render target users (kart previews in the menus, the minimap) do not
     *  want shadows anyway. */
    bool isRenderingToTexture() const           { return m_active_rtt != NULL; }
    // ------------------------------------------------------------------------
    /** Opens the frame's main scene if it is not open already, clearing it the
     *  first time.
     *
     *  beginScene() deliberately does not open a GXM scene, because the shadow
     *  cascade has to be rendered first and in a scene of its own. Opening the
     *  frame scene up front and closing it again for the shadow pass would mean
     *  storing and reloading the entire colour buffer, which on this GPU is the
     *  single most expensive thing a frame can do. So the scene is opened by
     *  whoever draws first. */
    bool ensureFrameScene();
    // ------------------------------------------------------------------------
    /** Ends the running scene, if any, without presenting. The pipeline needs
     *  this whenever a pass has to read what a previous pass wrote, because on
     *  a tile based GPU nothing is in memory until the scene ends. */
    void endCurrentScene();
    // ------------------------------------------------------------------------
    /** Starts a scene on \p surface. \p clear_color is drawn as a fullscreen
     *  quad when \p clear is set: GXM can clear depth for free through the
     *  tile background value, but colour has to be painted. */
    bool beginSurfaceScene(GEGXMSurface* surface, bool clear,
                           SColor clear_color);
    // ------------------------------------------------------------------------
    /** Draws the two triangle fullscreen quad in normalised device
     *  coordinates. The caller has already bound its own programs, textures
     *  and uniforms. */
    void drawFullscreenQuad();
    // ------------------------------------------------------------------------
    void* getFullscreenQuadVertices() const  { return m_fullscreen_vertices; }
    // ------------------------------------------------------------------------
    void* getFullscreenQuadIndices() const    { return m_fullscreen_indices; }
    // ------------------------------------------------------------------------
    GEGXMProgram* getFullscreenVertexProgram() const
                                             { return m_fullscreen_vertex; }
    // ------------------------------------------------------------------------
    SceGxmVertexProgram* getFullscreenPatchedVertexProgram() const
                                     { return m_fullscreen_vertex_patched; }
    // ------------------------------------------------------------------------
    video::SColor getClearColor() const              { return m_clear_color; }
    // ------------------------------------------------------------------------
    const core::rect<s32>& getCurrentClip() const           { return m_clip; }
    // ------------------------------------------------------------------------
    const core::rect<s32>& getViewPort() const          { return m_viewport; }
    // ------------------------------------------------------------------------
    video::ITexture* getWhiteTexture() const       { return m_white_texture; }
    // ------------------------------------------------------------------------
    video::ITexture* getTransparentTexture() const
                                             { return m_transparent_texture; }
    // ------------------------------------------------------------------------
    GEGXMMeshCache* getGXMMeshCache() const;
    // ------------------------------------------------------------------------
    GEGXMEnvironmentMap* getEnvironmentMap() const  { return m_environment_map; }
    // ------------------------------------------------------------------------
    GEGXMPostProcessing* getPostProcessing() const  { return m_post_processing; }
    // ------------------------------------------------------------------------
    GEGXMSurface* getShadowSurface() const        { return m_shadow_surface; }
    // ------------------------------------------------------------------------
    void createBillboardQuad();
    // ------------------------------------------------------------------------
    const SceGxmTexture* getShadowTexture() const;
    // ------------------------------------------------------------------------
    unsigned getShadowResolution() const     { return m_shadow_resolution; }
    // ------------------------------------------------------------------------
    /** Height of the cascade's world space extent, used to fit the light
     *  matrix; 0 disables shadows entirely. */
    bool hasShadows() const               { return m_shadow_surface != NULL; }
    // ------------------------------------------------------------------------
    GESPM* getBillboardQuad() const              { return m_billboard_quad; }
    // ------------------------------------------------------------------------
    IrrlichtDevice* getIrrlichtDevice() const     { return m_irrlicht_device; }
    // ------------------------------------------------------------------------
    SDL_Window* getSDLWindow() const     { return m_params.m_sdl_window; }
    // ------------------------------------------------------------------------
    /** Waits for every queued frame to finish and land on screen. Called
     *  before anything that frees GPU memory the GPU might still be reading. */
    void waitIdle();
    // ------------------------------------------------------------------------
    /** Suppresses waitIdle(). Removing a few thousand textures at once would
     *  otherwise drain the GPU per texture; the caller drains once up front and
     *  sets this for the duration, which is safe because nothing new is
     *  submitted while it holds. Mirrors GEVulkanDriver. */
    void setDisableWaitIdle(bool value)      { m_disable_wait_idle = value; }
    // ------------------------------------------------------------------------
    void addPolyCount(unsigned count)          { m_polycount += count; }
    // ------------------------------------------------------------------------
    void updateDriver(bool scale_changed = true, bool pbr_changed = false,
                      bool ibl_changed = false);
    // ------------------------------------------------------------------------
    void clearDrawCallsCache();
    // ------------------------------------------------------------------------
    void addDrawCallToCache(std::unique_ptr<GEGXMDrawCall>& dc);
    // ------------------------------------------------------------------------
    std::unique_ptr<GEGXMDrawCall> getDrawCallFromCache();
    // ------------------------------------------------------------------------
    /** Sets the GPU viewport and region clip from a rectangle in the current
     *  render target's pixel coordinates. */
    void applyViewport(const core::rect<s32>& area);
    // ------------------------------------------------------------------------
    /** Reserves per frame GPU memory that stays alive until the frame it was
     *  taken in has been displayed. Used for the per draw instance streams and
     *  index runs, which cannot be freed at the end of the call that made them
     *  because the GPU has not read them yet. */
    void* allocateFrameMemory(size_t size, size_t alignment);

private:
    // ------------------------------------------------------------------------
    virtual video::ITexture* createDeviceDependentTexture(IImage* surface,
                                 const io::path& name, void* mipmap_data = 0)
                                                             { return NULL; }
    // ------------------------------------------------------------------------
    virtual s32 addHighLevelShaderMaterial(
        const c8* vertex_shader_program, const c8* vertex_entry,
        E_VERTEX_SHADER_TYPE vs_target, const c8* pixel_shader_program,
        const c8* pixel_entry, E_PIXEL_SHADER_TYPE ps_target,
        const c8* geometry_shader_program, const c8* geometry_entry = "main",
        E_GEOMETRY_SHADER_TYPE gs_target = EGST_GS_4_0,
        scene::E_PRIMITIVE_TYPE in_type = scene::EPT_TRIANGLES,
        scene::E_PRIMITIVE_TYPE out_type = scene::EPT_TRIANGLE_STRIP,
        u32 vertices_out = 0, IShaderConstantSetCallBack* callback = 0,
        E_MATERIAL_TYPE base_material = video::EMT_SOLID, s32 user_data = 0,
        E_GPU_SHADING_LANGUAGE shading_lang = EGSL_DEFAULT) { return 0; }
    // ------------------------------------------------------------------------
    bool initGXM();
    // ------------------------------------------------------------------------
    bool createDisplayBuffers();
    // ------------------------------------------------------------------------
    bool createContext();
    // ------------------------------------------------------------------------
    bool createFullscreenQuad();
    // ------------------------------------------------------------------------
    bool createShadowSurface();
    // ------------------------------------------------------------------------
    void createPlaceholderTextures();
    // ------------------------------------------------------------------------
    void clearColorSurface(SColor color);
    // ------------------------------------------------------------------------
    void presentFrame();
    // ------------------------------------------------------------------------
    void resetFrameMemory();

    SIrrlichtCreationParameters m_params;

    SMaterial Material;

    IrrlichtDevice* m_irrlicht_device;

    SceGxmContext* m_context;

    /** Host memory the context keeps its own state in; plain malloc, not GPU
     *  visible. */
    void* m_context_host_mem;

    /** The three command ring buffers the context streams work through, all in
     *  uncached LPDDR so the CPU can write them without cache maintenance. */
    void* m_vdm_ring_buffer;
    void* m_vertex_ring_buffer;
    void* m_fragment_ring_buffer;
    void* m_fragment_usse_ring_buffer;

    /** One render target shared by all three display buffers: the tiling state
     *  only depends on the dimensions. */
    SceGxmRenderTarget* m_display_render_target;

    struct DisplayBuffer
    {
        void* m_data;
        SceGxmSyncObject* m_sync;
        std::unique_ptr<GEGXMSurface> m_surface;
    };
    std::vector<DisplayBuffer> m_display_buffers;

    unsigned m_back_buffer_index;
    unsigned m_front_buffer_index;

    /** Notification the CPU waits on to know a frame's fragment work is done.
     *  Points into the region sceGxmGetNotificationRegion() hands out. */
    volatile unsigned* m_notification_base;

    /** Offscreen colour target for the 3D scene when post processing is on;
     *  NULL when the scene renders straight into the display buffer, which is
     *  the fast path and what happens with post processing disabled. */
    std::unique_ptr<GEGXMSurface> m_scene_surface;

    GEGXMFBOTexture* m_scene_texture;

    GEGXMSurface* m_current_surface;

    GEGXMSurface* m_shadow_surface;

    /** A sampled view of the shadow cascade's depth surface. A DF32M depth
     *  surface can be read back as SCE_GXM_TEXTURE_FORMAT_F32M_R, which is
     *  what makes a sampled shadow map possible without an extra colour pass
     *  to encode depth. */
    SceGxmTexture m_shadow_texture;

    unsigned m_shadow_resolution;

    bool m_in_scene;

    /** Set between beginScene() and endScene(). Distinguishes a scene that is
     *  part of presenting a frame from one that is rendering to a texture. */
    bool m_in_frame;

    /** Whether the frame's main scene still owes a colour clear. */
    bool m_frame_needs_clear;

    GEGXMFBOTexture* m_active_rtt;

    core::dimension2d<u32> m_render_target_size;

    core::rect<s32> m_clip;

    core::rect<s32> m_viewport;

    video::SColor m_clear_color;

    /** Clear colour setRenderTarget() was given, kept because a render target's
     *  scene may have to be reopened and STK's previews rely on clearing to
     *  transparent. */
    video::SColor m_rtt_clear_color;

    SColorf m_ambient_light;

    video::ITexture* m_white_texture;

    video::ITexture* m_transparent_texture;

    GEGXMProgramCache* m_program_cache;

    GEGXMProgram* m_fullscreen_vertex;

    GEGXMProgram* m_clear_fragment;

    SceGxmVertexProgram* m_fullscreen_vertex_patched;

    SceGxmFragmentProgram* m_clear_fragment_patched;

    void* m_fullscreen_vertices;

    void* m_fullscreen_indices;

    GEGXMEnvironmentMap* m_environment_map;

    GEGXMPostProcessing* m_post_processing;

    GESPM* m_billboard_quad;

    /** Memoised result of getGXMMeshCache(), which is on the per draw path. */
    mutable irr::scene::IMeshCache* m_mesh_cache_source;
    mutable GEGXMMeshCache* m_mesh_cache;

    std::vector<std::unique_ptr<GEGXMDrawCall> > m_draw_calls_cache;

    /** Per frame scratch GPU memory, one arena per display buffer so that the
     *  arena being written is never one the GPU is still reading. */
    struct FrameArena
    {
        uint8_t* m_data;
        size_t m_size;
        size_t m_used;
    };
    std::vector<FrameArena> m_frame_arenas;

    unsigned m_polycount;

    bool m_disable_wait_idle;

    bool m_gxm_initialised;

};   // GEGXMDriver

}   // namespace GE

#endif   // _IRR_COMPILE_WITH_GXM_

#endif
