#include "ge_gxm_draw_call.hpp"

#ifdef _IRR_COMPILE_WITH_GXM_

#include "ge_culling_tool.hpp"
#include "ge_gxm_camera_scene_node.hpp"
#include "ge_gxm_driver.hpp"
#include "ge_gxm_environment_map.hpp"
#include "ge_gxm_mesh_cache.hpp"
#include "ge_gxm_render_target.hpp"
#include "ge_gxm_shader.hpp"
#include "ge_gxm_texture.hpp"
#include "ge_main.hpp"
#include "ge_material_manager.hpp"
#include "ge_render_info.hpp"
#include "ge_spm.hpp"
#include "ge_spm_buffer.hpp"
#include "ge_vulkan_animated_mesh_scene_node.hpp"
#include "mini_glm.hpp"

#include <IrrlichtDevice.h>
#include <ISceneManager.h>
#include "../source/Irrlicht/os.h"

#include "IBillboardSceneNode.h"
#include "ILightSceneNode.h"
#include "IMeshSceneNode.h"
#include "IParticleSystemSceneNode.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>

namespace GE
{
namespace
{
/** Which winding to discard for a back face culled material.
 *
 *  Not the same answer as the OpenGL renderer's. The GXM viewport this driver
 *  sets has a negative Y scale, because a GXM colour surface has its origin at
 *  the top left while irrlicht's projection matrices are built for OpenGL's
 *  bottom left. That mirrors screen space, so a triangle that rasterises
 *  counter-clockwise under OpenGL rasterises clockwise here, and the winding to
 *  cull is the opposite one. If models ever render inside out, this constant
 *  and nothing else is what to flip. */
const SceGxmCullMode BACKFACE_CULL = SCE_GXM_CULL_CCW;

/** Half-width of the shadow cascade in world units, and how far ahead of the
 *  camera it is centred. STK's tracks are hundreds of units across and a kart
 *  is about two, so a cascade this size keeps the texel density useful around
 *  the player while still catching the scenery that casts onto them. */
const float SHADOW_RANGE = 60.0f;
const float SHADOW_FORWARD_BIAS = 0.35f;

/** Materials that have no GXM fragment shader of their own and borrow one. */
struct ShaderMapping
{
    const char* m_ge_name;
    const char* m_fragment;
    const char* m_fragment_defines;
    const char* m_vertex_defines;
};

/** Maps GE's material names, which come from data/shaders/ge_shaders/
 *  shader_settings.xml, onto the Cg shaders in ge_gxm_shader_sources.cpp.
 *
 *  Several GE materials collapse onto one Cg shader with different defines,
 *  which keeps the number of runtime compilations - and so the cold cache
 *  startup time - down. "displace" has no entry: it needs to sample the scene
 *  colour while rendering into it, which without multiple render targets or a
 *  mid-scene resolve is not expressible here, so getShader() routes it to its
 *  non-PBR fallback instead. */
const ShaderMapping SHADER_MAPPINGS[] =
{
    { "solid",      "solid.frag",       "",                        ""             },
    { "normalmap",  "solid.frag",       "#define GXM_NORMAL_MAP 1", "#define GXM_NORMAL_MAP 1" },
    { "decal",      "decal.frag",       "",                        ""             },
    { "splatting",  "splatting.frag",   "",                        ""             },
    // GXM_ALPHA_TEST has to reach the fragment stage: that is where the discard
    // is. Without it an alpha tested material draws its cut-out texture opaque,
    // and since such textures are black where they are meant to be invisible,
    // every cut-out - foliage, railings, window frames - becomes a black
    // silhouette. GXM_UNLIT is read only by the fragment shader, and GXM_GRASS
    // only by the vertex shader, which animates the blade.
    { "alphatest",  "solid.frag",       "#define GXM_ALPHA_TEST 1",
                                        "#define GXM_ALPHA_TEST 1"             },
    { "unlit",      "solid.frag",       "#define GXM_ALPHA_TEST 1",
      "#define GXM_ALPHA_TEST 1\n#define GXM_UNLIT 1"                          },
    { "grass",      "solid.frag",
      "#define GXM_ALPHA_TEST 1\n#define GXM_GRASS 1",
                                        "#define GXM_ALPHA_TEST 1"             },
    { "alphablend", "transparent.frag", "",                        ""             },
    { "additive",   "transparent.frag", "",                        ""             },
};

// ----------------------------------------------------------------------------
const ShaderMapping* findMapping(const std::string& name)
{
    for (const ShaderMapping& m : SHADER_MAPPINGS)
    {
        if (name == m.m_ge_name)
            return &m;
    }
    return NULL;
}   // findMapping

// ----------------------------------------------------------------------------
/** Copies an irrlicht matrix into a shader uniform.
 *
 *  No transpose. The Cg side declares matrices as float4[4] and combines them
 *  with MUL4, which is defined to consume irrlicht's storage order directly -
 *  see the header comment in ge_gxm_shader_sources.cpp for why the shaders avoid
 *  float4x4 entirely. */
void setMatrixUniform(void* buffer, const SceGxmProgramParameter* param,
                      const irr::core::matrix4& matrix)
{
    if (param == NULL || buffer == NULL)
        return;
    sceGxmSetUniformDataF(buffer, param, 0, 16, matrix.pointer());
}   // setMatrixUniform

// ----------------------------------------------------------------------------
void setVec4Uniform(void* buffer, const SceGxmProgramParameter* param,
                    float x, float y, float z, float w)
{
    if (param == NULL || buffer == NULL)
        return;
    const float v[4] = { x, y, z, w };
    sceGxmSetUniformDataF(buffer, param, 0, 4, v);
}   // setVec4Uniform

// ----------------------------------------------------------------------------
const SceGxmTexture* resolveTexture(GEGXMDriver* driver,
                                    const irr::video::ITexture* texture,
                                    unsigned layer)
{
    if (texture != NULL)
    {
        const GEGXMTexture* gxm = dynamic_cast<const GEGXMTexture*>(texture);
        if (gxm != NULL && gxm->getGXMTexture() != NULL)
            return gxm->getGXMTexture();
        const GEGXMFBOTexture* fbo =
            dynamic_cast<const GEGXMFBOTexture*>(texture);
        if (fbo != NULL && fbo->getGXMTexture() != NULL)
            return fbo->getGXMTexture();
    }
    // Every sampler a program declares must have something bound or the draw is
    // undefined. Which placeholder matters: layer 0 is albedo, where a missing
    // texture means "use the vertex colour", so white. Every layer above it is
    // data - normals, and the packed gloss/metal/emissive map - where white
    // means fully metallic and fully emissive, which blows the surface out and
    // then renders it black for want of anything to reflect. Those get
    // transparent black, matching GEVulkanTextureDescriptor.
    const irr::video::ITexture* placeholder = layer == 0 ?
        driver->getWhiteTexture() : driver->getTransparentTexture();
    const GEGXMTexture* gxm =
        static_cast<const GEGXMTexture*>(placeholder);
    return gxm == NULL ? NULL : gxm->getGXMTexture();
}   // resolveTexture
}   // namespace

// ============================================================================
GEGXMPipeline::GEGXMPipeline()
{
    m_vertex = NULL;
    m_fragment = NULL;
    m_patched_vertex = NULL;
    m_patched_fragment = NULL;
    m_depth_vertex = NULL;
    m_depth_fragment = NULL;
    m_patched_depth_vertex = NULL;
    m_patched_depth_fragment = NULL;
    m_priority = 0;
    m_skinning = false;
    m_valid = false;
}   // GEGXMPipeline

// ============================================================================
GEGXMDrawCall::GEGXMDrawCall()
{
    m_culling_tool = new GECullingTool();
    m_shadow_culling_tool = new GECullingTool();
    m_skybox = NULL;
    m_has_sun = false;
    m_sun_direction = irr::core::vector3df(0.0f, 1.0f, 0.0f);
    m_sun_color = irr::video::SColorf(1.0f, 1.0f, 1.0f, 1.0f);
    m_last_sun_direction = m_sun_direction;
    m_had_sun = false;
    m_shadow_fitted = false;
    m_shadow_valid = false;
    m_skybox_program = NULL;
    m_patched_skybox = NULL;
    m_polycount = 0;
    m_pipeline_generation = 0;
}   // GEGXMDrawCall

// ----------------------------------------------------------------------------
GEGXMDrawCall::~GEGXMDrawCall()
{
    delete m_culling_tool;
    delete m_shadow_culling_tool;
}   // ~GEGXMDrawCall

// ----------------------------------------------------------------------------
void GEGXMDrawCall::reset()
{
    m_batches.clear();
    m_batch_lookup.clear();
    m_lights.clear();
    m_skybox = NULL;
    // Hand this frame's sun on to m_last_sun_direction before clearing it:
    // prepare() fits the cascade before the scene graph is walked, so it has to
    // use the previous frame's sun. The sun moves imperceptibly between frames,
    // and fitting after gathering would be too late to cull casters against.
    if (m_has_sun)
    {
        m_last_sun_direction = m_sun_direction;
        m_had_sun = true;
    }
    m_has_sun = false;
    m_shadow_valid = false;
    m_shared_vertex_uniforms.clear();
    m_shared_fragment_uniforms.clear();
}   // reset

// ----------------------------------------------------------------------------
void GEGXMDrawCall::clearRetainedSun()
{
    m_has_sun = false;
    m_had_sun = false;
    m_sun_direction = irr::core::vector3df(0.0f, 1.0f, 0.0f);
    m_last_sun_direction = m_sun_direction;
    m_sun_color = irr::video::SColorf(1.0f, 1.0f, 1.0f, 1.0f);
    m_shadow_fitted = false;
    m_shadow_valid = false;
}   // clearRetainedSun

// ----------------------------------------------------------------------------
void GEGXMDrawCall::prepare(GEGXMCameraSceneNode* cam)
{
    // Alpha scales it, matching GEVulkanLightHandler::prepare().
    m_ambient_color = irr::video::SColorf(0.0f, 0.0f, 0.0f, 1.0f);
    GEGXMDriver* gxm_driver = getGXMDriver();
    if (gxm_driver != NULL && gxm_driver->getIrrlichtDevice() != NULL &&
        gxm_driver->getIrrlichtDevice()->getSceneManager() != NULL)
    {
        const irr::video::SColorf c = gxm_driver->getIrrlichtDevice()
            ->getSceneManager()->getAmbientLight();
        m_ambient_color = irr::video::SColorf(c.r * c.a, c.g * c.a,
            c.b * c.a, 1.0f);
    }

    reset();
    m_polycount = 0;
    m_culling_tool->init(cam->getPVM(),
        cam->getViewFrustum()->getBoundingBox());
    m_view_position = cam->getAbsolutePosition();
    m_billboard_rotation = MiniGLM::getBulletQuaternion(cam->getViewMatrix());

    // Fit the cascade now, so that addNode() can test candidates against it as
    // well as against the camera. Uses the sun retained from the previous frame,
    // which reset() leaves alone for exactly this reason.
    m_shadow_fitted = false;
    GEGXMDriver* driver = getGXMDriver();
    // Same condition renderShadowPass() uses. Without the hasShadows() test the
    // very ordinary "dynamic lights on, shadows off" combination would gather
    // and sort caster-only batches that no pass ever draws.
    if (driver != NULL && driver->hasShadows() && getGEConfig()->m_pbr &&
        computeShadowMatrix(cam))
    {
        // The cascade is an orthographic box, so its frustum planes bound it
        // exactly and an infinite AABB is the right "no extra bound" value.
        irr::core::aabbox3df everything(-FLT_MAX, -FLT_MAX, -FLT_MAX,
            FLT_MAX, FLT_MAX, FLT_MAX);
        m_shadow_culling_tool->init(m_shadow_matrix, everything);
        m_shadow_fitted = true;
    }
}   // prepare

// ----------------------------------------------------------------------------
std::string GEGXMDrawCall::getShader(const irr::video::SMaterial& m) const
{
    std::string shader = GEMaterialManager::getShader(m.MaterialType);
    auto material = GEMaterialManager::getMaterial(shader);
    if (!material)
        return "solid";
    // Fall back when PBR is off, and always for "displace", which needs to read
    // the scene colour it is writing into.
    if ((!getGEConfig()->m_pbr && !material->m_nonpbr_fallback.empty()) ||
        shader == "displace")
    {
        if (!material->m_nonpbr_fallback.empty())
        {
            shader = material->m_nonpbr_fallback;
            material = GEMaterialManager::getMaterial(shader);
        }
    }
    // A render info marked transparent overrides an opaque material, which is
    // how karts fade in and out.
    auto& ri = m.getRenderInfo();
    if (material && !material->isTransparent() && ri && ri->isTransparent())
        return "alphablend";
    return shader;
}   // getShader

// ----------------------------------------------------------------------------
GEGXMPipeline* GEGXMDrawCall::getPipeline(GEGXMDriver* driver,
                                          const std::string& shader,
                                          bool skinning)
{
    const std::string key = shader + (skinning ? "|s" : "");
    auto it = m_pipelines.find(key);
    if (it != m_pipelines.end())
        return it->second.m_valid ? &it->second : NULL;

    GEGXMPipeline& pipeline = m_pipelines[key];
    pipeline.m_skinning = skinning;
    pipeline.m_material = GEMaterialManager::getMaterial(shader);
    const ShaderMapping* mapping = findMapping(shader);
    if (mapping == NULL || !pipeline.m_material)
    {
        irr::os::Printer::log("GXM: no shader mapping for material",
            shader.c_str(), irr::ELL_WARNING);
        return NULL;
    }

    const bool pbr = getGEConfig()->m_pbr;
    const bool transparent = pipeline.m_material->isTransparent();

    std::string vertex_defines = mapping->m_vertex_defines;
    std::string fragment_defines = mapping->m_fragment_defines;
    if (skinning)
        vertex_defines += "\n#define GXM_SKINNING 1";
    // The transparent shaders are unlit, so they need neither the PBR block nor
    // the interpolants that feed it.
    if (pbr && !transparent)
    {
        vertex_defines += "\n#define GXM_PBR 1";
        fragment_defines += "\n#define GXM_PBR 1";
        if (getGEConfig()->m_ibl)
            fragment_defines += "\n#define GXM_IBL 1";
        if (driver->hasShadows())
            fragment_defines += "\n#define GXM_SHADOW 1";
    }
    else if (pbr && transparent)
    {
        // Still needs the tone mapping curve so that transparents sit in the
        // same colour space as everything else - including which branch of it,
        // which convertColor() selects on GXM_IBL.
        fragment_defines += "\n#define GXM_PBR 1";
        if (getGEConfig()->m_ibl)
            fragment_defines += "\n#define GXM_IBL 1";
    }

    pipeline.m_vertex = GEGXMShaderManager::getVertexProgram("spm.vert",
        vertex_defines);
    pipeline.m_fragment = GEGXMShaderManager::getFragmentProgram(
        mapping->m_fragment, fragment_defines);
    if (pipeline.m_vertex == NULL || pipeline.m_fragment == NULL)
        return NULL;

    // The instance stream is indexed by instance rather than by vertex, which
    // is how GXM does instancing at all: sceGxmDrawInstanced() wraps the index
    // buffer every indexWrap indices and steps this stream on each wrap.
    SceGxmVertexStream streams[GVSI_COUNT];
    streams[GVSI_STATIC].stride = sizeof(GXMSPMVertex);
    streams[GVSI_STATIC].indexSource = SCE_GXM_INDEX_SOURCE_INDEX_16BIT;
    streams[GVSI_SKINNING].stride = sizeof(GXMSPMSkinning);
    streams[GVSI_SKINNING].indexSource = SCE_GXM_INDEX_SOURCE_INDEX_16BIT;
    streams[GVSI_INSTANCE].stride = sizeof(GXMObjectData);
    streams[GVSI_INSTANCE].indexSource = SCE_GXM_INDEX_SOURCE_INSTANCE_32BIT;

    const GEGXMAttributeDesc attributes[] =
    {
        { "a_position", GVSI_STATIC, offsetof(GXMSPMVertex, m_position),
          SCE_GXM_ATTRIBUTE_FORMAT_F32, 3 },
        { "a_normal", GVSI_STATIC, offsetof(GXMSPMVertex, m_normal),
          SCE_GXM_ATTRIBUTE_FORMAT_S16N, 4 },
        { "a_color", GVSI_STATIC, offsetof(GXMSPMVertex, m_color),
          SCE_GXM_ATTRIBUTE_FORMAT_U8N, 4 },
        { "a_uv", GVSI_STATIC, offsetof(GXMSPMVertex, m_uv),
          SCE_GXM_ATTRIBUTE_FORMAT_F16, 2 },
        { "a_uv_two", GVSI_STATIC, offsetof(GXMSPMVertex, m_uv_two),
          SCE_GXM_ATTRIBUTE_FORMAT_F16, 2 },
        { "a_tangent", GVSI_STATIC, offsetof(GXMSPMVertex, m_tangent),
          SCE_GXM_ATTRIBUTE_FORMAT_S16N, 4 },
        { "a_joint", GVSI_SKINNING, offsetof(GXMSPMSkinning, m_joint),
          SCE_GXM_ATTRIBUTE_FORMAT_S16, 4 },
        { "a_weight", GVSI_SKINNING, offsetof(GXMSPMSkinning, m_weight),
          SCE_GXM_ATTRIBUTE_FORMAT_F16, 4 },
        { "i_translation_hue", GVSI_INSTANCE,
          offsetof(GXMObjectData, m_translation_hue),
          SCE_GXM_ATTRIBUTE_FORMAT_F32, 4 },
        { "i_rotation", GVSI_INSTANCE, offsetof(GXMObjectData, m_rotation),
          SCE_GXM_ATTRIBUTE_FORMAT_F32, 4 },
        { "i_scale_skinning", GVSI_INSTANCE,
          offsetof(GXMObjectData, m_scale_skinning),
          SCE_GXM_ATTRIBUTE_FORMAT_F32, 4 },
        { "i_texture_trans", GVSI_INSTANCE,
          offsetof(GXMObjectData, m_texture_trans),
          SCE_GXM_ATTRIBUTE_FORMAT_F32, 4 },
        { "i_custom_color", GVSI_INSTANCE,
          offsetof(GXMObjectData, m_custom_color),
          SCE_GXM_ATTRIBUTE_FORMAT_F32, 4 },
    };
    const unsigned attribute_count =
        sizeof(attributes) / sizeof(attributes[0]);

    GEGXMVertexLayout layout;
    if (!buildVertexLayout(pipeline.m_vertex, attributes, attribute_count,
        streams, GVSI_COUNT, &layout))
        return NULL;

    GEGXMProgramCache* cache = driver->getProgramCache();
    pipeline.m_patched_vertex = cache->getVertexProgram(pipeline.m_vertex,
        layout);

    GEGXMBlendState blend = GEGXMBlendState::opaque();
    if (pipeline.m_material->m_additive)
        blend = GEGXMBlendState::additive();
    else if (pipeline.m_material->m_alphablend)
    {
        // transparent.frag outputs colour already multiplied by alpha, so the
        // blender must not multiply by it a second time.
        blend = GEGXMBlendState::premultipliedAlphaBlend();
    }
    pipeline.m_patched_fragment = cache->getFragmentProgram(
        pipeline.m_fragment, pipeline.m_vertex, blend);
    if (pipeline.m_patched_vertex == NULL ||
        pipeline.m_patched_fragment == NULL)
        return NULL;

    // Depth only forms for the shadow caster pass. Alpha tested materials need
    // a fragment shader there so the cut out holes do not cast; everything else
    // runs with no fragment shader at all, which is the cheapest thing this GPU
    // can do.
    const bool alpha_tested = !pipeline.m_material->texturelessDepth();
    if (!transparent)
    {
        std::string depth_vertex_defines = skinning ?
            "#define GXM_SKINNING 1" : "";
        // The caster has to be displaced the same way the visible geometry is,
        // or its shadow detaches from it.
        if (shader == "grass")
            depth_vertex_defines += "\n#define GXM_GRASS 1";
        std::string depth_fragment_defines;
        if (alpha_tested)
        {
            depth_vertex_defines += "\n#define GXM_ALPHA_TEST 1";
            depth_fragment_defines = "#define GXM_ALPHA_TEST 1";
        }
        pipeline.m_depth_vertex = GEGXMShaderManager::getVertexProgram(
            "spm_depth.vert", depth_vertex_defines);
        pipeline.m_depth_fragment = GEGXMShaderManager::getFragmentProgram(
            "depth_only.frag", depth_fragment_defines);
        if (pipeline.m_depth_vertex != NULL &&
            pipeline.m_depth_fragment != NULL)
        {
            GEGXMVertexLayout depth_layout;
            if (buildVertexLayout(pipeline.m_depth_vertex, attributes,
                attribute_count, streams, GVSI_COUNT, &depth_layout))
            {
                pipeline.m_patched_depth_vertex = cache->getVertexProgram(
                    pipeline.m_depth_vertex, depth_layout);
            }
            // The shadow pass writes no colour, so the output format is
            // irrelevant; a mask of none keeps the fragment shader from being
            // asked to produce one.
            GEGXMBlendState depth_blend;
            depth_blend.m_color_mask = SCE_GXM_COLOR_MASK_NONE;
            pipeline.m_patched_depth_fragment = cache->getFragmentProgram(
                pipeline.m_depth_fragment, pipeline.m_depth_vertex,
                depth_blend);
        }
    }

    // Draw order: opaque materials first, grouped so that the heavier shaders
    // come last and are more likely to be depth rejected; transparents after
    // all of them.
    pipeline.m_priority = transparent ? 100 : 0;
    if (shader == "splatting")
        pipeline.m_priority = 1;
    else if (shader == "normalmap")
        pipeline.m_priority = 2;
    else if (shader == "alphatest" || shader == "unlit" || shader == "grass")
        pipeline.m_priority = 3;
    pipeline.m_valid = true;
    return &pipeline;
}   // getPipeline

// ----------------------------------------------------------------------------
GEGXMDrawCall::Batch& GEGXMDrawCall::getBatch(GESPMBuffer* mb,
                                              const TexturesList& textures,
                                              const std::string& shader,
                                              bool transparent)
{
    auto key = std::make_pair(mb, textures);
    auto& by_shader = m_batch_lookup[key];
    auto it = by_shader.find(shader);
    if (it != by_shader.end())
        return m_batches[it->second];

    by_shader[shader] = (unsigned)m_batches.size();
    m_batches.push_back(Batch());
    Batch& batch = m_batches.back();
    batch.m_mb = mb;
    batch.m_textures = textures;
    batch.m_shader = shader;
    batch.m_skinning_node = NULL;
    batch.m_dynamic_vertices = NULL;
    batch.m_dynamic_indices = NULL;
    batch.m_dynamic_index_count = 0;
    batch.m_visible_count = 0;
    batch.m_sort_depth = 0.0f;
    batch.m_transparent = transparent;
    return batch;
}   // getBatch

// ----------------------------------------------------------------------------
void GEGXMDrawCall::addInstance(GESPMBuffer* mb, const TexturesList& textures,
                                const std::string& shader, bool transparent,
                                const GXMObjectData& data,
                                irr::scene::ISceneNode* skinning_node,
                                bool caster_only)
{
    if (skinning_node != NULL)
    {
        // A skinned node cannot share a batch: its joint palette is per draw
        // vertex uniform data, and two nodes with different skeletons cannot be
        // instanced together. So each one becomes its own single instance batch.
        m_batches.push_back(Batch());
        Batch& batch = m_batches.back();
        batch.m_mb = mb;
        batch.m_textures = textures;
        batch.m_shader = shader;
        batch.m_skinning_node = skinning_node;
        batch.m_dynamic_vertices = NULL;
        batch.m_dynamic_indices = NULL;
        batch.m_dynamic_index_count = 0;
        batch.m_transparent = transparent;
        batch.m_visible_count = caster_only ? 0u : 1u;
        if (caster_only)
            batch.m_caster_instances.push_back(data);
        else
            batch.m_instances.push_back(data);
        batch.m_sort_depth = (irr::core::vector3df(data.m_translation_hue[0],
            data.m_translation_hue[1], data.m_translation_hue[2]) -
            m_view_position).getLengthSQ();
        return;
    }

    Batch& batch = getBatch(mb, textures, shader, transparent);
    if (batch.m_instances.empty() && batch.m_caster_instances.empty())
    {
        batch.m_sort_depth = (irr::core::vector3df(data.m_translation_hue[0],
            data.m_translation_hue[1], data.m_translation_hue[2]) -
            m_view_position).getLengthSQ();
    }
    // Kept in two lists so that a mesh buffer shared between visible and
    // off-screen nodes - which is the normal case for track objects and items,
    // where one GESPMBuffer serves every instance of a model - does not drag its
    // off-screen instances through the lit pass just because one of them is on
    // screen.
    if (caster_only)
        batch.m_caster_instances.push_back(data);
    else
        batch.m_instances.push_back(data);
}   // addInstance

// ----------------------------------------------------------------------------
void GEGXMDrawCall::addDynamicBuffer(GEGXMDriver* driver, GESPMBuffer* mb,
                                     const TexturesList& textures,
                                     const std::string& shader,
                                     bool transparent,
                                     const GXMObjectData& data,
                                     bool caster_only)
{
    if (driver == NULL || mb == NULL)
        return;
    const unsigned vertex_count = mb->getVertexCount();
    const unsigned index_count = mb->getIndexCount();
    if (vertex_count == 0 || index_count == 0)
        return;

    void* vertices = driver->allocateFrameMemory(
        (size_t)vertex_count * sizeof(GXMSPMVertex), 16);
    void* indices = driver->allocateFrameMemory(
        (size_t)index_count * sizeof(uint16_t), 16);
    if (vertices == NULL || indices == NULL)
        return;

    // Same expansion the mesh cache does: the packed 10-bit normal and tangent
    // have no equivalent SceGxmAttributeFormat, so they become S16N here.
    const irr::video::S3DVertexSkinnedMesh* src =
        (const irr::video::S3DVertexSkinnedMesh*)mb->getVertices();
    GXMSPMVertex* dest = (GXMSPMVertex*)vertices;
    for (unsigned i = 0; i < vertex_count; i++)
        convertSPMVertex(src[i], dest + i);
    memcpy(indices, mb->getIndices(), index_count * sizeof(uint16_t));

    m_batches.push_back(Batch());
    Batch& batch = m_batches.back();
    batch.m_mb = mb;
    batch.m_textures = textures;
    batch.m_shader = shader;
    batch.m_skinning_node = NULL;
    batch.m_dynamic_vertices = vertices;
    batch.m_dynamic_indices = indices;
    batch.m_dynamic_index_count = index_count;
    batch.m_transparent = transparent;
    batch.m_visible_count = caster_only ? 0u : 1u;
    if (caster_only)
        batch.m_caster_instances.push_back(data);
    else
        batch.m_instances.push_back(data);
    batch.m_sort_depth = (irr::core::vector3df(data.m_translation_hue[0],
        data.m_translation_hue[1], data.m_translation_hue[2]) -
        m_view_position).getLengthSQ();
}   // addDynamicBuffer

// ----------------------------------------------------------------------------
void GEGXMDrawCall::fillObjectData(irr::scene::ISceneNode* node,
                                   int material_id,
                                   GXMObjectData* out) const
{
    memset(out, 0, sizeof(*out));

    const irr::core::matrix4& model_mat = node->getAbsoluteTransformation();
    irr::core::quaternion rotation(0.0f, 0.0f, 0.0f, 1.0f);
    irr::core::vector3df scale = model_mat.getScale();
    if (scale.X != 0.0f && scale.Y != 0.0f && scale.Z != 0.0f)
    {
        // Strip the scale out so what is left is a pure rotation that can be
        // expressed as a quaternion; the shader then applies translation,
        // rotation and scale separately. Same decomposition the Vulkan
        // backend's ObjectData::init() does.
        irr::core::matrix4 local_mat = model_mat;
        local_mat[0] = local_mat[0] / scale.X / local_mat[15];
        local_mat[1] = local_mat[1] / scale.X / local_mat[15];
        local_mat[2] = local_mat[2] / scale.X / local_mat[15];
        local_mat[4] = local_mat[4] / scale.Y / local_mat[15];
        local_mat[5] = local_mat[5] / scale.Y / local_mat[15];
        local_mat[6] = local_mat[6] / scale.Y / local_mat[15];
        local_mat[8] = local_mat[8] / scale.Z / local_mat[15];
        local_mat[9] = local_mat[9] / scale.Z / local_mat[15];
        local_mat[10] = local_mat[10] / scale.Z / local_mat[15];
        rotation = MiniGLM::getQuaternion(local_mat);
        // rotateVector() in the shader expects the conjugate, matching the
        // GLSL renderer.
        rotation.W = -rotation.W;
    }

    out->m_translation_hue[0] = model_mat[12];
    out->m_translation_hue[1] = model_mat[13];
    out->m_translation_hue[2] = model_mat[14];
    out->m_rotation[0] = rotation.X;
    out->m_rotation[1] = rotation.Y;
    out->m_rotation[2] = rotation.Z;
    out->m_rotation[3] = rotation.W;
    out->m_scale_skinning[0] = scale.X;
    out->m_scale_skinning[1] = scale.Y;
    out->m_scale_skinning[2] = scale.Z;
    out->m_scale_skinning[3] = 0.0f;

    const irr::video::SMaterial& material = node->getMaterial(material_id);
    const irr::core::matrix4& texture_matrix = material.getTextureMatrix(0);
    out->m_texture_trans[0] = texture_matrix[8];
    out->m_texture_trans[1] = texture_matrix[9];

    auto& ri = material.getRenderInfo();
    out->m_translation_hue[3] = (ri && ri->getHue() > 0.0f) ? ri->getHue() :
        0.0f;

    irr::video::SColor custom_color((uint32_t)-1);
    if (ri)
    {
        custom_color = ri->getVertexColor();
        // Vertex colours are authored in sRGB; with PBR on, everything the BRDF
        // sees has to be linear first.
        if (getGEConfig()->m_pbr)
            custom_color = srgb255ToLinearFromSColor(custom_color);
    }
    out->m_custom_color[0] = custom_color.getRed() / 255.0f;
    out->m_custom_color[1] = custom_color.getGreen() / 255.0f;
    out->m_custom_color[2] = custom_color.getBlue() / 255.0f;
    out->m_custom_color[3] = custom_color.getAlpha() / 255.0f;
}   // fillObjectData

// ----------------------------------------------------------------------------
void GEGXMDrawCall::addNode(irr::scene::ISceneNode* node)
{
    irr::scene::IMesh* mesh = NULL;
    GEVulkanAnimatedMeshSceneNode* animated = NULL;
    if (node->getType() == irr::scene::ESNT_ANIMATED_MESH)
    {
        animated = static_cast<GEVulkanAnimatedMeshSceneNode*>(node);
        mesh = animated->getMesh();
    }
    else if (node->getType() == irr::scene::ESNT_MESH)
    {
        mesh = static_cast<irr::scene::IMeshSceneNode*>(node)->getMesh();
        for (unsigned i = 0; i < mesh->getMeshBufferCount(); i++)
        {
            // Anything that is not an SPM buffer never made it into the packed
            // mesh cache and so has nothing to draw from.
            if (mesh->getMeshBuffer(i)->getVertexType() !=
                irr::video::EVT_SKINNED_MESH)
                return;
        }
    }
    else
        return;
    if (mesh == NULL)
        return;

    const bool has_skinning = animated != NULL &&
        !animated->getSkinningMatrices().empty();

    for (unsigned i = 0; i < mesh->getMeshBufferCount(); i++)
    {
        GESPMBuffer* buffer =
            static_cast<GESPMBuffer*>(mesh->getMeshBuffer(i));
        if (buffer->getIndexCount() == 0)
            continue;
        bool caster_only = false;
        if (m_culling_tool->isCulled(buffer, node))
        {
            // Not visible. Still worth keeping if it is inside the cascade,
            // because it may cast into something that is.
            if (!m_shadow_fitted ||
                m_shadow_culling_tool->isCulled(buffer, node))
                continue;
            caster_only = true;
        }
        const irr::video::SMaterial& material = node->getMaterial(i);
        const std::string shader = getShader(material);
        auto ge_material = GEMaterialManager::getMaterial(shader);
        const bool transparent = ge_material && ge_material->isTransparent();

        GXMObjectData data;
        fillObjectData(node, i, &data);

        // A transparent surface casts no shadow here, so an off-screen one has
        // nothing left to contribute. Checked before the streamed path below,
        // because converting a streamed buffer's geometry costs an upload into
        // the frame arena and all three streamed users are alpha blended.
        if (caster_only && transparent)
            continue;

        // Streamed buffers - skid marks, kart shadows, the rubber band - are
        // rebuilt every frame, so they are not in the packed mesh cache and
        // have to carry their own geometry.
        if (buffer->getHardwareMappingHint_Vertex() ==
            irr::scene::EHM_STREAM ||
            buffer->getHardwareMappingHint_Index() == irr::scene::EHM_STREAM)
        {
            addDynamicBuffer(getGXMDriver(), buffer, getTexturesList(material),
                shader, transparent, data, caster_only);
            continue;
        }

        // gxmSupportsSkinning() is false when no joint palette size compiled;
        // the mesh is then drawn through the static path, i.e. in bind pose.
        const bool skinned = has_skinning && buffer->hasSkinning() &&
            gxmSupportsSkinning();
        addInstance(buffer, getTexturesList(material), shader, transparent,
            data, skinned ? node : NULL, caster_only);
    }
}   // addNode

// ----------------------------------------------------------------------------
void GEGXMDrawCall::addBillboardNode(irr::scene::ISceneNode* node,
                                     irr::scene::ESCENE_NODE_TYPE node_type)
{
    GEGXMDriver* driver = getGXMDriver();
    if (driver == NULL || driver->getBillboardQuad() == NULL)
        return;
    irr::core::aabbox3df bb = node->getTransformedBoundingBox();
    if (m_culling_tool->isCulled(bb))
        return;

    // Billboards and particles are instances of one shared unit quad rather
    // than geometry of their own, which is what lets a whole particle system be
    // a single instanced draw.
    GESPMBuffer* quad = static_cast<GESPMBuffer*>(
        driver->getBillboardQuad()->getMeshBuffer(0u));
    if (quad == NULL)
        return;

    const irr::video::SMaterial& material = node->getMaterial(0);
    const std::string shader = getShader(material);
    auto ge_material = GEMaterialManager::getMaterial(shader);
    const bool transparent = ge_material && ge_material->isTransparent();
    const TexturesList textures = getTexturesList(material);

    if (node_type == irr::scene::ESNT_BILLBOARD)
    {
        irr::scene::IBillboardSceneNode* billboard =
            static_cast<irr::scene::IBillboardSceneNode*>(node);
        GXMObjectData data;
        memset(&data, 0, sizeof(data));
        const irr::core::matrix4& m = node->getAbsoluteTransformation();
        data.m_translation_hue[0] = m[12];
        data.m_translation_hue[1] = m[13];
        data.m_translation_hue[2] = m[14];
        for (unsigned i = 0; i < 4; i++)
            data.m_rotation[i] = m_billboard_rotation[i];
        const irr::core::vector2df size = billboard->getSize();
        data.m_scale_skinning[0] = size.X * 0.5f;
        data.m_scale_skinning[1] = size.Y * 0.5f;
        data.m_scale_skinning[2] = 0.0f;

        irr::video::SColor top, bottom;
        billboard->getColor(top, bottom);
        irr::video::SColor average;
        average.setAlpha((top.getAlpha() + bottom.getAlpha()) / 2);
        average.setRed((top.getRed() + bottom.getRed()) / 2);
        average.setGreen((top.getGreen() + bottom.getGreen()) / 2);
        average.setBlue((top.getBlue() + bottom.getBlue()) / 2);
        if (getGEConfig()->m_pbr)
            average = srgb255ToLinearFromSColor(average);
        data.m_custom_color[0] = average.getRed() / 255.0f;
        data.m_custom_color[1] = average.getGreen() / 255.0f;
        data.m_custom_color[2] = average.getBlue() / 255.0f;
        data.m_custom_color[3] = average.getAlpha() / 255.0f;
        addInstance(quad, textures, shader, transparent, data, NULL);
        return;
    }

    if (node_type != irr::scene::ESNT_PARTICLE_SYSTEM)
        return;
    irr::scene::IParticleSystemSceneNode* system =
        static_cast<irr::scene::IParticleSystemSceneNode*>(node);
    const irr::core::array<irr::scene::SParticle>& particles =
        system->getParticles();
    for (irr::u32 p = 0; p < particles.size(); p++)
    {
        const irr::scene::SParticle& particle = particles[p];
        GXMObjectData data;
        memset(&data, 0, sizeof(data));
        data.m_translation_hue[0] = particle.pos.X;
        data.m_translation_hue[1] = particle.pos.Y;
        data.m_translation_hue[2] = particle.pos.Z;
        for (unsigned i = 0; i < 4; i++)
            data.m_rotation[i] = m_billboard_rotation[i];
        data.m_scale_skinning[0] = particle.size.Width * 0.5f;
        data.m_scale_skinning[1] = particle.size.Height * 0.5f;
        data.m_scale_skinning[2] = 0.0f;
        irr::video::SColor color = particle.color;
        if (getGEConfig()->m_pbr)
            color = srgb255ToLinearFromSColor(color);
        data.m_custom_color[0] = color.getRed() / 255.0f;
        data.m_custom_color[1] = color.getGreen() / 255.0f;
        data.m_custom_color[2] = color.getBlue() / 255.0f;
        data.m_custom_color[3] = color.getAlpha() / 255.0f;
        addInstance(quad, textures, shader, transparent, data, NULL);
    }
}   // addBillboardNode

// ----------------------------------------------------------------------------
void GEGXMDrawCall::addSkyBox(irr::scene::ISceneNode* node)
{
    m_skybox = node;
    GEGXMDriver* driver = getGXMDriver();
    if (driver != NULL && driver->getEnvironmentMap() != NULL)
        driver->getEnvironmentMap()->addSkyBox(node);
}   // addSkyBox

// ----------------------------------------------------------------------------
void GEGXMDrawCall::addLightNode(irr::scene::ILightSceneNode* node)
{
    const irr::video::SLight& light = node->getLightData();
    if (node->getLightType() == irr::video::ELT_DIRECTIONAL)
    {
        // The sun. Its "position" is a direction for a directional light.
        m_has_sun = true;
        // irrlicht stores the direction the light travels in; every consumer
        // here - the BRDF, the most representative point approximation and the
        // shadow cascade's eye position - wants the direction towards the sun.
        // Matches ge_vulkan_light_handler.cpp.
        m_sun_direction = -light.Direction;
        if (m_sun_direction.getLengthSQ() < 1e-8f)
            m_sun_direction = irr::core::vector3df(0.0f, 1.0f, 0.0f);
        m_sun_direction.normalize();
        m_sun_color = light.DiffuseColor;
        return;
    }

    if (m_culling_tool->isCulled(light.Position, light.Radius))
        return;

    LightData data;
    memset(&data, 0, sizeof(data));
    data.m_position_radius[0] = light.Position.X;
    data.m_position_radius[1] = light.Position.Y;
    data.m_position_radius[2] = light.Position.Z;
    data.m_position_radius[3] = light.Radius;
    data.m_color_inverse_square_range[0] = light.DiffuseColor.r;
    data.m_color_inverse_square_range[1] = light.DiffuseColor.g;
    data.m_color_inverse_square_range[2] = light.DiffuseColor.b;
    const float range = std::max(light.Radius, 0.0001f);
    data.m_color_inverse_square_range[3] = 1.0f / (range * range);
    if (node->getLightType() == irr::video::ELT_SPOT)
    {
        // Same packing as the GLSL renderer: the direction's x and y with its z
        // reconstructed in the shader, plus the cone's scale and offset.
        irr::core::vector3df direction = light.Direction;
        if (direction.getLengthSQ() > 1e-8f)
            direction.normalize();
        const float outer = std::max(light.OuterCone, 0.0001f);
        const float inner = std::min(light.InnerCone, outer - 0.0001f);
        const float cos_outer = cosf(outer);
        const float cos_inner = cosf(inner);
        const float scale = 1.0f / std::max(cos_inner - cos_outer, 0.0001f);
        data.m_direction_scale_offset[0] = direction.X;
        data.m_direction_scale_offset[1] = direction.Y;
        data.m_direction_scale_offset[2] = direction.Z < 0.0f ? -scale : scale;
        data.m_direction_scale_offset[3] = -cos_outer * scale;
    }
    data.m_camera_distance =
        (light.Position - m_view_position).getLengthSQ();
    m_lights.push_back(data);
}   // addLightNode

// ----------------------------------------------------------------------------
void GEGXMDrawCall::generate(GEGXMDriver* driver)
{
    // Only GXM_MAX_LIGHT lights fit in the fragment uniform block, and each one
    // is a full BRDF evaluation per pixel, so keep the nearest and drop the
    // rest. Nearest rather than brightest because a distant light is attenuated
    // to nothing anyway.
    if (m_lights.size() > GXM_MAX_LIGHT)
    {
        std::partial_sort(m_lights.begin(), m_lights.begin() + GXM_MAX_LIGHT,
            m_lights.end(), [](const LightData& a, const LightData& b)
            {
                return a.m_camera_distance < b.m_camera_distance;
            });
        m_lights.resize(GXM_MAX_LIGHT);
    }

    // Fold the caster-only instances onto the end of each batch's list, so the
    // visible passes can draw the prefix and the caster pass the whole thing
    // from one uploaded stream.
    for (Batch& b : m_batches)
    {
        b.m_visible_count = (unsigned)b.m_instances.size();
        if (!b.m_caster_instances.empty())
        {
            b.m_instances.insert(b.m_instances.end(),
                b.m_caster_instances.begin(), b.m_caster_instances.end());
            b.m_caster_instances.clear();
        }
    }

    // Opaque batches sort by material so that consecutive draws share programs;
    // transparent ones sort back to front, which alpha blending requires.
    std::stable_sort(m_batches.begin(), m_batches.end(),
        [this, driver](const Batch& a, const Batch& b)
        {
            if (a.m_transparent != b.m_transparent)
                return b.m_transparent;
            if (a.m_transparent)
                return a.m_sort_depth > b.m_sort_depth;
            if (a.m_shader != b.m_shader)
                return a.m_shader < b.m_shader;
            return a.m_mb < b.m_mb;
        });
}   // generate

// ----------------------------------------------------------------------------
bool GEGXMDrawCall::computeShadowMatrix(GEGXMCameraSceneNode* cam)
{
    // The retained sun, not this frame's: see prepare().
    if (!m_had_sun)
        return false;
    const irr::core::vector3df sun_direction = m_last_sun_direction;

    // One cascade fitted around a point ahead of the camera. A proper cascaded
    // setup would look better but each extra cascade is another full geometry
    // pass, and the shadow pass is already a real slice of the frame here.
    const irr::core::vector3df camera_position = cam->getAbsolutePosition();
    irr::core::vector3df forward = cam->getTarget() - camera_position;
    if (forward.getLengthSQ() > 1e-8f)
        forward.normalize();
    const irr::core::vector3df center =
        camera_position + forward * (SHADOW_RANGE * SHADOW_FORWARD_BIAS);

    // The sun direction points from the surface towards the light, so the eye
    // goes along it, away from the scene.
    const irr::core::vector3df eye = center + sun_direction * SHADOW_RANGE;
    irr::core::vector3df up(0.0f, 1.0f, 0.0f);
    if (fabsf(sun_direction.Y) > 0.99f)
        up = irr::core::vector3df(0.0f, 0.0f, 1.0f);

    irr::core::matrix4 light_view;
    light_view.buildCameraLookAtMatrixLH(eye, center, up);
    irr::core::matrix4 light_projection;
    // The depth range has to reach behind the cascade centre as well as in
    // front of it, or casters between the sun and the visible ground are
    // clipped away and stop casting.
    light_projection.buildProjectionMatrixOrthoLH(SHADOW_RANGE * 2.0f,
        SHADOW_RANGE * 2.0f, 0.1f, SHADOW_RANGE * 3.0f);
    m_shadow_matrix = light_projection * light_view;
    return true;
}   // computeShadowMatrix

// ----------------------------------------------------------------------------
void GEGXMDrawCall::bindMaterialTextures(GEGXMDriver* driver,
                                         const Batch& batch,
                                         const GEGXMPipeline& pipeline)
{
    SceGxmContext* ctx = driver->getContext();
    const bool pbr = getGEConfig()->m_pbr;
    for (unsigned layer = 0; layer < GXM_MATERIAL_TEXTURE_COUNT; layer++)
    {
        const irr::video::ITexture* texture = layer < batch.m_textures.size() ?
            batch.m_textures[layer] : NULL;
        // Albedo layers are authored in sRGB and must be linearised by the
        // texture unit before the BRDF sees them; normal maps and the packed
        // gloss/metal/emissive map must not be. GEMaterial carries that per
        // layer, and on GXM it is a bit in the texture's own control word
        // rather than a separate view, so it is applied at bind time.
        if (pbr && pipeline.m_material &&
            layer < pipeline.m_material->m_srgb_settings.size())
        {
            GEGXMTexture* mutable_texture = dynamic_cast<GEGXMTexture*>(
                const_cast<irr::video::ITexture*>(texture));
            if (mutable_texture != NULL)
            {
                mutable_texture->setSRGB(
                    pipeline.m_material->m_srgb_settings[layer]);
            }
        }
        const SceGxmTexture* gxm = resolveTexture(driver, texture, layer);
        if (gxm != NULL)
            sceGxmSetFragmentTexture(ctx, GTU_MATERIAL_0 + layer, gxm);
    }
}   // bindMaterialTextures

// ----------------------------------------------------------------------------
void GEGXMDrawCall::bindEnvironmentTextures(GEGXMDriver* driver)
{
    SceGxmContext* ctx = driver->getContext();
    GEGXMEnvironmentMap* env = driver->getEnvironmentMap();
    const GEGXMTexture* white =
        static_cast<const GEGXMTexture*>(driver->getWhiteTexture());
    const SceGxmTexture* fallback = white == NULL ? NULL :
        white->getGXMTexture();

    const SceGxmTexture* irradiance = env == NULL ? NULL :
        env->getIrradianceTexture();
    const SceGxmTexture* radiance = env == NULL ? NULL :
        env->getRadianceTexture();
    sceGxmSetFragmentTexture(ctx, GTU_DIFFUSE_ENV,
        irradiance != NULL ? irradiance : fallback);
    sceGxmSetFragmentTexture(ctx, GTU_SPECULAR_ENV,
        radiance != NULL ? radiance : fallback);

    const SceGxmTexture* shadow = driver->getShadowTexture();
    // Binding white when there is no shadow map makes every depth comparison
    // pass, i.e. fully lit, which is the right no-shadow behaviour.
    sceGxmSetFragmentTexture(ctx, GTU_SHADOW_MAP,
        (shadow != NULL && m_shadow_valid) ? shadow : fallback);
}   // bindEnvironmentTextures

// ----------------------------------------------------------------------------
void* GEGXMDrawCall::getSharedVertexUniforms(GEGXMDriver* driver,
                                     GEGXMProgram* program,
                                     const irr::core::matrix4& projection_view)
{
    auto it = m_shared_vertex_uniforms.find(program);
    if (it != m_shared_vertex_uniforms.end())
        return it->second;

    const unsigned size = program->getDefaultUniformBufferSize();
    if (size == 0)
    {
        m_shared_vertex_uniforms[program] = NULL;
        return NULL;
    }
    // Allocated from the frame arena rather than reserved from the ring buffer,
    // so its lifetime is the frame and it can be rebound draw after draw with
    // sceGxmSetVertexDefaultUniformBuffer().
    void* buffer = driver->allocateFrameMemory(size, 16);
    if (buffer == NULL)
        return NULL;
    memset(buffer, 0, size);
    setMatrixUniform(buffer, program->getParameter("u_projection_view"),
        projection_view);

    // Grass sways along the wind direction the material manager keeps.
    const SceGxmProgramParameter* wind =
        program->getParameter("u_wind_direction");
    if (wind != NULL)
    {
        uint32_t constants_size = 0;
        void* constants_data = NULL;
        auto material = GEMaterialManager::getMaterial("grass");
        if (material && material->m_push_constants)
        {
            material->m_push_constants(&constants_size, &constants_data);
            if (constants_data != NULL &&
                constants_size >= sizeof(float) * 3)
            {
                const float* wind_direction = (const float*)constants_data;
                setVec4Uniform(buffer, wind, wind_direction[0],
                    wind_direction[1], wind_direction[2], 0.0f);
            }
        }
    }
    m_shared_vertex_uniforms[program] = buffer;
    return buffer;
}   // getSharedVertexUniforms

// ----------------------------------------------------------------------------
void* GEGXMDrawCall::getSharedFragmentUniforms(GEGXMDriver* driver,
                                               GEGXMProgram* program,
                                               GEGXMCameraSceneNode* cam)
{
    auto it = m_shared_fragment_uniforms.find(program);
    if (it != m_shared_fragment_uniforms.end())
        return it->second;

    const unsigned size = program->getDefaultUniformBufferSize();
    if (size == 0)
    {
        m_shared_fragment_uniforms[program] = NULL;
        return NULL;
    }
    void* buffer = driver->allocateFrameMemory(size, 16);
    if (buffer == NULL)
        return NULL;
    memset(buffer, 0, size);

    const GEGXMCameraMatrices* matrices = cam->getMatrices();
    setMatrixUniform(buffer, program->getParameter("u_view_matrix"),
        matrices->m_view_matrix);
    setMatrixUniform(buffer, program->getParameter("u_inverse_view_matrix"),
        matrices->m_inverse_view_matrix);
    setMatrixUniform(buffer, program->getParameter("u_shadow_matrix"),
        m_shadow_matrix);

    setVec4Uniform(buffer, program->getParameter("u_sun_color"),
        m_sun_color.r, m_sun_color.g, m_sun_color.b, 1.0f);
    // The sun's angular radius as a tangent, for the most representative point
    // approximation that softens its specular highlight.
    setVec4Uniform(buffer, program->getParameter("u_sun_direction"),
        m_sun_direction.X, m_sun_direction.Y, m_sun_direction.Z, 0.00465f);

    setVec4Uniform(buffer, program->getParameter("u_ambient_color"),
        m_ambient_color.r, m_ambient_color.g, m_ambient_color.b, 1.0f);

    GEGXMEnvironmentMap* env = driver->getEnvironmentMap();
    if (env != NULL)
    {
        const irr::video::SColor skytop = env->getSkytopColor();
        setVec4Uniform(buffer, program->getParameter("u_skytop_color"),
            skytop.getRed() / 255.0f, skytop.getGreen() / 255.0f,
            skytop.getBlue() / 255.0f, 1.0f);
    }
    else
    {
        // The same fallback GEVulkanLightHandler uses when a track has no
        // skybox to take a colour from.
        setVec4Uniform(buffer, program->getParameter("u_skytop_color"),
            0.325f, 0.35f, 0.375f, 1.0f);
    }

    const irr::video::SColor fog = driver->getFogColor();
    setVec4Uniform(buffer, program->getParameter("u_fog_color"),
        fog.getRed() / 255.0f, fog.getGreen() / 255.0f,
        fog.getBlue() / 255.0f, 1.0f);

    const float specular_levels = env == NULL ? 0.0f :
        env->getSpecularLevels();
    const float shadow_texel = driver->getShadowResolution() == 0 ? 0.0f :
        1.0f / (float)driver->getShadowResolution();
    setVec4Uniform(buffer, program->getParameter("u_misc"),
        driver->getFogDensity(),
        specular_levels, (float)m_lights.size(), shadow_texel);

    const SceGxmProgramParameter* lights = program->getParameter("u_lights");
    if (lights != NULL && !m_lights.empty())
    {
        // Three float4s per light, packed exactly as the GXM_LIGHT_* macros in
        // the shader read them.
        std::vector<float> data(m_lights.size() * 12, 0.0f);
        for (unsigned i = 0; i < m_lights.size(); i++)
        {
            memcpy(&data[i * 12 + 0], m_lights[i].m_position_radius,
                sizeof(float) * 4);
            memcpy(&data[i * 12 + 4],
                m_lights[i].m_color_inverse_square_range, sizeof(float) * 4);
            memcpy(&data[i * 12 + 8], m_lights[i].m_direction_scale_offset,
                sizeof(float) * 4);
        }
        sceGxmSetUniformDataF(buffer, lights, 0, (unsigned)data.size(),
            data.data());
    }

    m_shared_fragment_uniforms[program] = buffer;
    return buffer;
}   // getSharedFragmentUniforms

// ----------------------------------------------------------------------------
void GEGXMDrawCall::drawBatch(GEGXMDriver* driver, const Batch& batch,
                              const GEGXMPipeline& pipeline, bool depth_only,
                              unsigned instance_count)
{
    if (instance_count == 0 || batch.m_instances.empty() ||
        batch.m_mb == NULL)
        return;
    if (instance_count > batch.m_instances.size())
        instance_count = (unsigned)batch.m_instances.size();
    SceGxmContext* ctx = driver->getContext();

    const size_t instance_size = (size_t)instance_count *
        sizeof(GXMObjectData);
    void* instance_data = driver->allocateFrameMemory(instance_size, 16);
    if (instance_data == NULL)
        return;
    memcpy(instance_data, batch.m_instances.data(), instance_size);

    unsigned index_count = 0;
    void* vertex_stream = batch.m_dynamic_vertices;
    void* index_stream = batch.m_dynamic_indices;
    index_count = batch.m_dynamic_index_count;
    GEGXMMeshCache* cache = driver->getGXMMeshCache();
    if (vertex_stream == NULL)
    {
        // Base vertex rendering on GXM is pointer arithmetic: the mesh
        // buffer's offset into the packed cache becomes the stream address, so
        // the indices stay local to the mesh.
        if (cache == NULL || !cache->isValid())
            return;
        vertex_stream = cache->getVertexStream(batch.m_mb->getVBOOffset());
        index_stream = cache->getIndexStream(batch.m_mb->getIBOOffset());
        index_count = batch.m_mb->getIndexCount();
    }
    if (vertex_stream == NULL || index_stream == NULL || index_count == 0)
        return;

    sceGxmSetVertexStream(ctx, GVSI_STATIC, vertex_stream);
    if (pipeline.m_skinning && cache != NULL && cache->isValid())
    {
        sceGxmSetVertexStream(ctx, GVSI_SKINNING,
            cache->getSkinningStream(batch.m_mb->getVBOOffset()));
    }
    sceGxmSetVertexStream(ctx, GVSI_INSTANCE, instance_data);

    if (instance_count == 1)
    {
        // A stream with an instance index source reads element 0 for a
        // non-instanced draw, which is exactly right here and skips the
        // instancing path in the vertex pipeline.
        sceGxmDraw(ctx, SCE_GXM_PRIMITIVE_TRIANGLES, SCE_GXM_INDEX_FORMAT_U16,
            index_stream, index_count);
    }
    else
    {
        // indexWrap is the point at which the index buffer restarts and the
        // instance index steps, so the total index count is one mesh's worth
        // per instance.
        sceGxmDrawInstanced(ctx, SCE_GXM_PRIMITIVE_TRIANGLES,
            SCE_GXM_INDEX_FORMAT_U16, index_stream,
            index_count * instance_count, index_count);
    }
    if (!depth_only)
        m_polycount += (index_count / 3) * instance_count;
}   // drawBatch

// ----------------------------------------------------------------------------
void GEGXMDrawCall::renderShadowPass(GEGXMDriver* driver,
                                     GEGXMCameraSceneNode* cam)
{
    if (!driver->hasShadows() || !getGEConfig()->m_pbr)
        return;
    // Fitted in prepare(), which had to happen before gathering so casters could
    // be culled against it.
    if (!m_shadow_fitted)
        return;

    GEGXMSurface* shadow = driver->getShadowSurface();
    if (shadow == NULL)
        return;
    // No clear: the colour surface is disabled and depth is initialised from
    // the surface's background value at the top of every scene, so a depth only
    // pass starts clean for free.
    if (!driver->beginSurfaceScene(shadow, false,
        irr::video::SColor(255, 0, 0, 0)))
        return;

    SceGxmContext* ctx = driver->getContext();
    sceGxmSetFrontDepthFunc(ctx, SCE_GXM_DEPTH_FUNC_LESS_EQUAL);
    sceGxmSetFrontDepthWriteEnable(ctx, SCE_GXM_DEPTH_WRITE_ENABLED);
    // A constant slope scaled offset pushes casters away from the light, which
    // together with the shader's bias is what keeps surfaces from shadowing
    // themselves.
    sceGxmSetFrontDepthBias(ctx, 2, 4);
    // Casters are rendered without back face culling: a shadow caster's back
    // faces are the ones nearest the light for anything the camera sees the
    // front of, and culling them makes thin geometry stop casting.
    sceGxmSetCullMode(ctx, SCE_GXM_CULL_NONE);

    m_shared_vertex_uniforms.clear();
    m_shared_fragment_uniforms.clear();

    SceGxmVertexProgram* bound_vertex = NULL;
    SceGxmFragmentProgram* bound_fragment = NULL;
    for (const Batch& batch : m_batches)
    {
        if (batch.m_transparent)
            continue;
        GEGXMPipeline* pipeline = getPipeline(driver, batch.m_shader,
            batch.m_skinning_node != NULL);
        if (pipeline == NULL || pipeline->m_patched_depth_vertex == NULL ||
            pipeline->m_patched_depth_fragment == NULL)
            continue;

        if (bound_vertex != pipeline->m_patched_depth_vertex)
        {
            sceGxmSetVertexProgram(ctx, pipeline->m_patched_depth_vertex);
            bound_vertex = pipeline->m_patched_depth_vertex;
        }
        if (bound_fragment != pipeline->m_patched_depth_fragment)
        {
            sceGxmSetFragmentProgram(ctx,
                pipeline->m_patched_depth_fragment);
            bound_fragment = pipeline->m_patched_depth_fragment;
        }

        void* vertex_uniforms = NULL;
        if (batch.m_skinning_node != NULL)
        {
            // A skinned batch needs its own buffer because the joint palette
            // differs per node, so it cannot share the pass wide one.
            const unsigned size =
                pipeline->m_depth_vertex->getDefaultUniformBufferSize();
            vertex_uniforms = driver->allocateFrameMemory(size, 16);
            if (vertex_uniforms == NULL)
                continue;
            memset(vertex_uniforms, 0, size);
            setMatrixUniform(vertex_uniforms,
                pipeline->m_depth_vertex->getParameter("u_projection_view"),
                m_shadow_matrix);
            const std::vector<irr::core::matrix4>& matrices =
                static_cast<GEVulkanAnimatedMeshSceneNode*>(
                batch.m_skinning_node)->getSkinningMatrices();
            const SceGxmProgramParameter* joints =
                pipeline->m_depth_vertex->getParameter("u_joint_matrices");
            if (joints != NULL && !matrices.empty())
            {
                const unsigned count = std::min((unsigned)matrices.size(),
                    getGXMMaxJoints());
                std::vector<float> rows(count * 12, 0.0f);
                for (unsigned i = 0; i < count; i++)
                {
                    // Three rows of an affine transform, in the order MUL4
                    // consumes: row r is (m[r], m[4+r], m[8+r], m[12+r]).
                    const float* m = matrices[i].pointer();
                    for (unsigned r = 0; r < 3; r++)
                    {
                        rows[i * 12 + r * 4 + 0] = m[r];
                        rows[i * 12 + r * 4 + 1] = m[4 + r];
                        rows[i * 12 + r * 4 + 2] = m[8 + r];
                        rows[i * 12 + r * 4 + 3] = m[12 + r];
                    }
                }
                sceGxmSetUniformDataF(vertex_uniforms, joints, 0,
                    (unsigned)rows.size(), rows.data());
            }
        }
        else
        {
            vertex_uniforms = getSharedVertexUniforms(driver,
                pipeline->m_depth_vertex, m_shadow_matrix);
        }
        if (vertex_uniforms != NULL)
            sceGxmSetVertexDefaultUniformBuffer(ctx, vertex_uniforms);

        // Alpha tested casters sample their albedo to know where the holes are.
        if (!pipeline->m_material->texturelessDepth())
        {
            const SceGxmTexture* albedo = resolveTexture(driver,
                batch.m_textures.empty() ? NULL : batch.m_textures[0], 0);
            if (albedo != NULL)
                sceGxmSetFragmentTexture(ctx, GTU_MATERIAL_0, albedo);
        }
        // The caster pass draws every instance, visible or not.
        drawBatch(driver, batch, *pipeline, true/*depth_only*/,
            (unsigned)batch.m_instances.size());
    }

    sceGxmSetFrontDepthBias(ctx, 0, 0);
    driver->endCurrentScene();
    m_shadow_valid = true;
    m_shared_vertex_uniforms.clear();
    m_shared_fragment_uniforms.clear();
}   // renderShadowPass

// ----------------------------------------------------------------------------
/** KNOWN DEFECT: the sky renders as a flat wash of the panorama's ground band
 *  rather than the sky itself, under Vita3K at least.
 *
 *  What has been ruled out, so the next attempt does not repeat it: the pass
 *  runs and owns those pixels (forcing a constant colour fills the sky with
 *  it); the panorama the environment map builds is correct (dump it with the
 *  gxm_debug marker file and look); the corner rays computed below are correct
 *  and symmetric (they are logged); the ray uniforms do arrive (reading one
 *  back through the colour output gives a plausible direction, not zero); the
 *  sampler resolves to texture unit 0, which is the unit the radiance map is
 *  bound to; and it is unaffected by the texture being swizzled or linear, by
 *  tex2D versus tex2Dlod, by lerp versus an explicit mix, and by whether the
 *  uniform buffer is reserved or allocated from the frame arena.
 *
 *  What that leaves is the step in between: the direction the shader ends up
 *  with samples near v = 1, the bottom of the panorama, even though the corner
 *  rays going in do not point there. */
void GEGXMDrawCall::renderSkyBox(GEGXMDriver* driver,
                                 GEGXMCameraSceneNode* cam)
{
    GEGXMEnvironmentMap* env = driver->getEnvironmentMap();
    if (m_skybox == NULL || env == NULL || !env->isReady())
    {
        return;
    }

    if (m_skybox_program == NULL)
    {
        m_skybox_program =
            GEGXMShaderManager::getFragmentProgram("skybox.frag");
        if (m_skybox_program == NULL)
            return;
        m_patched_skybox = driver->getProgramCache()->getFragmentProgram(
            m_skybox_program, driver->getFullscreenVertexProgram(),
            GEGXMBlendState::opaque());
    }
    if (m_patched_skybox == NULL)
        return;

    SceGxmContext* ctx = driver->getContext();
    // Drawn first, before any geometry, with depth writes off. On a tile based
    // GPU a fullscreen write at the start of a scene is nearly free and it
    // means the geometry that follows simply overdraws the sky rather than
    // having to be depth tested against it.
    sceGxmSetFrontDepthFunc(ctx, SCE_GXM_DEPTH_FUNC_ALWAYS);
    sceGxmSetFrontDepthWriteEnable(ctx, SCE_GXM_DEPTH_WRITE_DISABLED);
    sceGxmSetCullMode(ctx, SCE_GXM_CULL_NONE);
    sceGxmSetVertexProgram(ctx,
        driver->getFullscreenPatchedVertexProgram());
    sceGxmSetFragmentProgram(ctx, m_patched_skybox);
    sceGxmSetFragmentTexture(ctx, GTU_MATERIAL_0, env->getRadianceTexture());

    // The four corner rays, unprojected here rather than in the shader. Doing it
    // on the CPU costs four matrix transforms a frame instead of two per pixel,
    // and - more to the point - it can be checked: irr::core::matrix4's own
    // transformVect() applies the matrix in exactly the convention the rest of
    // this driver uploads it in, so there is no second place for a clip space
    // convention to disagree.
    const irr::core::matrix4& inverse_pv =
        cam->getMatrices()->m_inverse_projection_view_matrix;
    float corners[16] = {};
    for (unsigned i = 0; i < 4; i++)
    {
        // o_uv order: (0,0), (1,0), (0,1), (1,1). Clip x is 2u-1; clip y is the
        // negation of that, because o_uv runs down the screen while clip y runs
        // up it.
        const float u = (i & 1) ? 1.0f : 0.0f;
        const float v = (i & 2) ? 1.0f : 0.0f;
        const float clip_x = 2.0f * u - 1.0f;
        const float clip_y = -(2.0f * v - 1.0f);
        float near_point[4];
        float far_point[4];
        // z = 0 is the near plane and z = 1 the far one in this clip
        // convention; any two distinct depths describe the same ray.
        inverse_pv.transformVect(near_point,
            irr::core::vector3df(clip_x, clip_y, 0.0f));
        inverse_pv.transformVect(far_point,
            irr::core::vector3df(clip_x, clip_y, 1.0f));
        irr::core::vector3df dir(0.0f, 0.0f, 1.0f);
        if (fabsf(near_point[3]) > 1e-9f && fabsf(far_point[3]) > 1e-9f)
        {
            dir = irr::core::vector3df(
                far_point[0] / far_point[3] - near_point[0] / near_point[3],
                far_point[1] / far_point[3] - near_point[1] / near_point[3],
                far_point[2] / far_point[3] - near_point[2] / near_point[3]);
            dir.normalize();
        }
        corners[i * 4 + 0] = dir.X;
        corners[i * 4 + 1] = dir.Y;
        corners[i * 4 + 2] = dir.Z;
        corners[i * 4 + 3] = 0.0f;
    }

    // Allocated out of the frame arena and bound explicitly, the same way
    // renderPass() does it, rather than through
    // sceGxmReserveFragmentDefaultUniformBuffer(): the arena route is the one
    // this driver has verified end to end.
    void* uniforms = driver->allocateFrameMemory(
        m_skybox_program->getDefaultUniformBufferSize(), 16);
    if (uniforms != NULL)
    {
        memset(uniforms, 0, m_skybox_program->getDefaultUniformBufferSize());
        static const char* const NAMES[4] =
            { "u_ray_tl", "u_ray_tr", "u_ray_bl", "u_ray_br" };
        for (unsigned i = 0; i < 4; i++)
        {
            const SceGxmProgramParameter* p =
                m_skybox_program->getParameter(NAMES[i]);
            if (p == NULL)
                continue;
            sceGxmSetUniformDataF(uniforms, p, 0, 4, corners + i * 4);
        }
        sceGxmSetFragmentDefaultUniformBuffer(ctx, uniforms);
    }
    driver->drawFullscreenQuad();

    sceGxmSetFrontDepthFunc(ctx, SCE_GXM_DEPTH_FUNC_LESS_EQUAL);
    sceGxmSetFrontDepthWriteEnable(ctx, SCE_GXM_DEPTH_WRITE_ENABLED);
}   // renderSkyBox

// ----------------------------------------------------------------------------
void GEGXMDrawCall::renderPass(GEGXMDriver* driver,
                               GEGXMCameraSceneNode* cam, bool transparent)
{
    SceGxmContext* ctx = driver->getContext();
    const irr::core::matrix4& projection_view =
        cam->getMatrices()->m_projection_view_matrix;

    SceGxmVertexProgram* bound_vertex = NULL;
    SceGxmFragmentProgram* bound_fragment = NULL;
    SceGxmCullMode bound_cull = SCE_GXM_CULL_NONE;
    bool bound_cull_valid = false;
    bool depth_write = true;

    bindEnvironmentTextures(driver);

    for (const Batch& batch : m_batches)
    {
        if (!isInPass(batch, transparent))
            continue;
        GEGXMPipeline* pipeline = getPipeline(driver, batch.m_shader,
            batch.m_skinning_node != NULL);
        if (pipeline == NULL)
            continue;

        if (bound_vertex != pipeline->m_patched_vertex)
        {
            sceGxmSetVertexProgram(ctx, pipeline->m_patched_vertex);
            bound_vertex = pipeline->m_patched_vertex;
        }
        if (bound_fragment != pipeline->m_patched_fragment)
        {
            sceGxmSetFragmentProgram(ctx, pipeline->m_patched_fragment);
            bound_fragment = pipeline->m_patched_fragment;
        }

        const SceGxmCullMode cull = pipeline->m_material->m_backface_culling ?
            BACKFACE_CULL : SCE_GXM_CULL_NONE;
        if (!bound_cull_valid || bound_cull != cull)
        {
            sceGxmSetCullMode(ctx, cull);
            bound_cull = cull;
            bound_cull_valid = true;
        }
        if (depth_write != pipeline->m_material->m_depth_write)
        {
            depth_write = pipeline->m_material->m_depth_write;
            sceGxmSetFrontDepthWriteEnable(ctx, depth_write ?
                SCE_GXM_DEPTH_WRITE_ENABLED : SCE_GXM_DEPTH_WRITE_DISABLED);
        }
        sceGxmSetFrontDepthFunc(ctx, pipeline->m_material->m_depth_test ?
            SCE_GXM_DEPTH_FUNC_LESS_EQUAL : SCE_GXM_DEPTH_FUNC_ALWAYS);

        void* vertex_uniforms = NULL;
        if (batch.m_skinning_node != NULL)
        {
            const unsigned size =
                pipeline->m_vertex->getDefaultUniformBufferSize();
            vertex_uniforms = driver->allocateFrameMemory(size, 16);
            if (vertex_uniforms == NULL)
                continue;
            memset(vertex_uniforms, 0, size);
            setMatrixUniform(vertex_uniforms,
                pipeline->m_vertex->getParameter("u_projection_view"),
                projection_view);
            const std::vector<irr::core::matrix4>& matrices =
                static_cast<GEVulkanAnimatedMeshSceneNode*>(
                batch.m_skinning_node)->getSkinningMatrices();
            const SceGxmProgramParameter* joints =
                pipeline->m_vertex->getParameter("u_joint_matrices");
            if (joints != NULL && !matrices.empty())
            {
                if (matrices.size() > getGXMMaxJoints())
                {
                    static bool warned = false;
                    if (!warned)
                    {
                        warned = true;
                        irr::os::Printer::log("GXM: a mesh has more joints "
                            "than the palette holds; it will be drawn with "
                            "the palette clamped", irr::ELL_WARNING);
                    }
                }
                const unsigned count = std::min((unsigned)matrices.size(),
                    getGXMMaxJoints());
                std::vector<float> rows(count * 12, 0.0f);
                for (unsigned i = 0; i < count; i++)
                {
                    const float* m = matrices[i].pointer();
                    for (unsigned r = 0; r < 3; r++)
                    {
                        rows[i * 12 + r * 4 + 0] = m[r];
                        rows[i * 12 + r * 4 + 1] = m[4 + r];
                        rows[i * 12 + r * 4 + 2] = m[8 + r];
                        rows[i * 12 + r * 4 + 3] = m[12 + r];
                    }
                }
                sceGxmSetUniformDataF(vertex_uniforms, joints, 0,
                    (unsigned)rows.size(), rows.data());
            }
        }
        else
        {
            vertex_uniforms = getSharedVertexUniforms(driver,
                pipeline->m_vertex, projection_view);
        }
        if (vertex_uniforms != NULL)
            sceGxmSetVertexDefaultUniformBuffer(ctx, vertex_uniforms);

        void* fragment_uniforms = getSharedFragmentUniforms(driver,
            pipeline->m_fragment, cam);
        if (fragment_uniforms != NULL)
            sceGxmSetFragmentDefaultUniformBuffer(ctx, fragment_uniforms);

        bindMaterialTextures(driver, batch, *pipeline);
        drawBatch(driver, batch, *pipeline, false/*depth_only*/,
            batch.m_visible_count);
    }

    if (!depth_write)
        sceGxmSetFrontDepthWriteEnable(ctx, SCE_GXM_DEPTH_WRITE_ENABLED);
}   // renderPass

// ----------------------------------------------------------------------------
void GEGXMDrawCall::syncPipelineGeneration()
{
    // Everything cached here is derived from programs the shader manager owns,
    // so a generation change means all of it is dangling. Done once per render
    // rather than inside getPipeline(), because renderSkyBox() uses its own
    // cached program before the first getPipeline() call of the frame.
    const unsigned generation = GEGXMShaderManager::getGeneration();
    if (m_pipeline_generation == generation)
        return;
    m_pipelines.clear();
    m_skybox_program = NULL;
    m_patched_skybox = NULL;
    m_pipeline_generation = generation;
}   // syncPipelineGeneration

// ----------------------------------------------------------------------------
void GEGXMDrawCall::render(GEGXMDriver* driver, GEGXMCameraSceneNode* cam)
{
    syncPipelineGeneration();

    // The shadow pass has to come first and in its own scene, before the frame
    // scene is opened: on a tile based GPU the cascade does not exist in memory
    // until its scene has ended, and reopening the frame scene afterwards would
    // mean storing and reloading the whole colour buffer.
    //
    // Skipped whenever a scene is already open - a render target texture, or a
    // second split screen camera drawing into the frame scene. Ending an open
    // scene to render the cascade would be unrecoverable: GXM has no colour
    // force-load, so whatever had been drawn into that surface would be gone
    // when it reopened. Those cameras fall back to m_shadow_valid == false and
    // bind the white fallback, i.e. they render unshadowed rather than wrong.
    // Both tests are needed. isInScene() catches the second and later split
    // screen cameras; isRenderingToTexture() catches a render target whose scene
    // was closed underneath us - GEGXMMeshCache::updateCache() drains the GPU
    // when a preview's meshes have just been added, which happens on exactly the
    // frame a kart preview first renders.
    if (!driver->isInScene() && !driver->isRenderingToTexture())
        renderShadowPass(driver, cam);

    // A no-op when a scene is already open, which is the render-to-texture case.
    if (!driver->ensureFrameScene())
        return;

    irr::core::rect<irr::s32> viewport = cam->getViewPort();
    if (viewport.getWidth() <= 0 || viewport.getHeight() <= 0)
    {
        // Render target cameras never get a viewport set on them, so fall back
        // to the whole target rather than rendering into a zero sized area.
        const irr::core::dimension2du& size =
            driver->getCurrentRenderTargetSize();
        viewport = irr::core::rect<irr::s32>(0, 0, size.Width, size.Height);
    }
    driver->applyViewport(viewport);

    m_shared_vertex_uniforms.clear();
    m_shared_fragment_uniforms.clear();

    renderSkyBox(driver, cam);
    renderPass(driver, cam, false/*transparent*/);
    renderPass(driver, cam, true/*transparent*/);

    driver->addPolyCount(m_polycount);
}   // render

}   // namespace GE

#endif   // _IRR_COMPILE_WITH_GXM_
