#ifndef HEADER_GE_GXM_DRAW_CALL_HPP
#define HEADER_GE_GXM_DRAW_CALL_HPP

#include "IrrCompileConfig.h"

#ifdef _IRR_COMPILE_WITH_GXM_

#include <psp2/gxm.h>

#include "ge_gxm_limits.hpp"

#include "LinearMath/btQuaternion.h"

#include "ESceneNodeTypes.h"
#include "SColor.h"
#include "SMaterial.h"
#include "matrix4.h"
#include "vector3d.h"

#include <array>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace irr
{
    namespace scene
    {
        class ISceneNode; class IBillboardSceneNode; class ILightSceneNode;
        class IMesh; struct SParticle;
    }
}

namespace GE
{
class GECullingTool;
class GEGXMCameraSceneNode;
class GEGXMDriver;
class GEGXMProgram;
class GESPMBuffer;
struct GEMaterial;

/** Everything needed to issue one material's draws, built lazily and cached.
 *
 *  On this hardware a "pipeline" is a patched vertex program plus a patched
 *  fragment program. The blend equation and the output format are baked into the
 *  fragment program at patch time rather than being dynamic state, so a material
 *  that draws both opaque and blended needs two of them; and the vertex program
 *  is patched against a specific attribute layout, so the skinned and static
 *  forms are separate objects too. */
struct GEGXMPipeline
{
    GEGXMProgram* m_vertex;
    GEGXMProgram* m_fragment;
    SceGxmVertexProgram* m_patched_vertex;
    SceGxmFragmentProgram* m_patched_fragment;

    /** Depth only forms used by the shadow caster pass. */
    GEGXMProgram* m_depth_vertex;
    GEGXMProgram* m_depth_fragment;
    SceGxmVertexProgram* m_patched_depth_vertex;
    SceGxmFragmentProgram* m_patched_depth_fragment;

    std::shared_ptr<const GEMaterial> m_material;

    /** Sort order within a pass. Solid materials are ordered to group program
     *  switches; transparent ones are drawn after and sorted by depth. */
    int m_priority;

    bool m_skinning;
    bool m_valid;

    GEGXMPipeline();
};   // GEGXMPipeline

/** Collects the visible scene for one camera and renders it.
 *
 *  The equivalent of GEVulkanDrawCall, but the differences are not cosmetic.
 *  There are no command buffers to build up front, no descriptor sets, no
 *  indirect draws and no storage buffers, so the structure is: gather visible
 *  mesh buffers into instance batches on the CPU, then walk them issuing
 *  sceGxmDraw calls with per instance data fed through a vertex stream.
 *
 *  The passes, in order:
 *
 *   1. Shadow casters, depth only, into a single orthographic cascade. Its own
 *      scene, with the colour surface disabled so nothing is written out but
 *      depth.
 *   2. Skybox, drawn first at maximum depth rather than last, because on a tile
 *      based GPU an early full screen write costs nothing and it means the
 *      geometry that follows overdraws it rather than being depth tested
 *      against it.
 *   3. Solid geometry, forward lit: sun with shadow, image based ambient, and
 *      the culled point lights, all in one pass.
 *   4. Transparent geometry, sorted back to front, depth tested but not
 *      written. */
class GEGXMDrawCall
{
private:
    typedef std::array<const irr::video::ITexture*,
        _IRR_MATERIAL_MAX_TEXTURES_> TexturesList;

    /** One drawable group: a mesh buffer, a set of textures and the instances
     *  that share both. */
    struct Batch
    {
        GESPMBuffer* m_mb;
        TexturesList m_textures;
        std::string m_shader;
        /** Instances to draw. generate() arranges these so the ones inside the
         *  camera frustum come first, which lets the visible passes draw a
         *  prefix of the list and the caster pass draw all of it - the same
         *  uploaded stream serves both. */
        std::vector<GXMObjectData> m_instances;
        /** Instances that are inside the shadow cascade but not the camera
         *  frustum. Concatenated onto m_instances by generate(). */
        std::vector<GXMObjectData> m_caster_instances;
        /** Size of the visible prefix of m_instances, valid after generate(). */
        unsigned m_visible_count;
        /** Non NULL when the batch is a single skinned node, whose joint
         *  palette has to be uploaded with it. */
        irr::scene::ISceneNode* m_skinning_node;
        /** Set for a streamed mesh buffer, whose geometry was converted into
         *  the frame arena when the node was gathered rather than living in the
         *  packed mesh cache. NULL means "look it up in the cache". */
        void* m_dynamic_vertices;
        void* m_dynamic_indices;
        unsigned m_dynamic_index_count;
        /** View space depth of the first instance, for sorting transparents. */
        float m_sort_depth;
        bool m_transparent;
    };

    struct LightData
    {
        float m_position_radius[4];
        float m_color_inverse_square_range[4];
        float m_direction_scale_offset[4];
        /** Distance to the camera, used to pick which lights survive the cull
         *  down to GXM_MAX_LIGHT. */
        float m_camera_distance;
    };

    GECullingTool* m_culling_tool;

    /** Second frustum, fitted to the shadow cascade. A caster does not have to
     *  be visible to matter, so gathering tests both and keeps whatever passes
     *  either. */
    GECullingTool* m_shadow_culling_tool;

    std::vector<Batch> m_batches;

    /** Batch index by (mesh buffer, textures, shader), so instances of the same
     *  drawable merge instead of becoming separate draws. */
    std::map<std::pair<GESPMBuffer*, TexturesList>,
        std::map<std::string, unsigned> > m_batch_lookup;

    std::vector<LightData> m_lights;

    irr::scene::ISceneNode* m_skybox;

    irr::core::vector3df m_view_position;

    btQuaternion m_billboard_rotation;

    /** Sun direction and colour, from the directional light in the scene. */
    irr::core::vector3df m_sun_direction;
    irr::video::SColorf m_sun_color;
    bool m_has_sun;

    /** The previous frame's sun, kept because prepare() has to fit the shadow
     *  cascade before the scene graph has been walked and so before this frame's
     *  sun is known. Held separately from m_sun_direction so that a track's sun
     *  cannot leak into a later scene that has none. */
    irr::core::vector3df m_last_sun_direction;
    bool m_had_sun;

    irr::core::matrix4 m_shadow_matrix;

    bool m_shadow_valid;

    /** Whether m_shadow_matrix (and so m_shadow_culling_tool) was fitted this
     *  frame. Computed in prepare(), before the lights are gathered, using the
     *  sun retained from the previous frame - see prepare(). */
    bool m_shadow_fitted;

    std::unordered_map<std::string, GEGXMPipeline> m_pipelines;

    /** Shader manager generation m_pipelines was built against. A recycled draw
     *  call whose generation is stale drops its cache rather than handing the
     *  GPU a released program. */
    unsigned m_pipeline_generation;

    /** This frame's ambient light, read from the scene manager in prepare().
     *
     *  It has to come from there rather than from the driver: STK sets it with
     *  ISceneManager::setAmbientLight(), which never reaches
     *  IVideoDriver::setAmbientLight(), so the driver's own copy stays black.
     *  Ambient is the only light a surface the sun does not reach gets, so a
     *  black one renders every wall, ceiling and shaded face pure black. */
    irr::video::SColorf m_ambient_color;

    GEGXMProgram* m_skybox_program;

    SceGxmFragmentProgram* m_patched_skybox;

    unsigned m_polycount;

    /** Per pass cache of filled uniform buffers, keyed by program. The lit
     *  uniform block is identical for every draw in a pass, so it is written
     *  once and rebound rather than refilled per draw. Cleared between passes,
     *  because the shadow pass uses a different projection. */
    std::unordered_map<const GEGXMProgram*, void*> m_shared_vertex_uniforms;
    std::unordered_map<const GEGXMProgram*, void*> m_shared_fragment_uniforms;

    // ------------------------------------------------------------------------
    std::string getShader(const irr::video::SMaterial& m) const;
    // ------------------------------------------------------------------------
    TexturesList getTexturesList(const irr::video::SMaterial& m) const
    {
        TexturesList textures;
        for (unsigned i = 0; i < textures.size(); i++)
            textures[i] = m.TextureLayer[i].Texture;
        return textures;
    }
    // ------------------------------------------------------------------------
    /** Builds, or returns from cache, the programs for a material. */
    GEGXMPipeline* getPipeline(GEGXMDriver* driver, const std::string& shader,
                               bool skinning);
    // ------------------------------------------------------------------------
    void syncPipelineGeneration();
    // ------------------------------------------------------------------------
    Batch& getBatch(GESPMBuffer* mb, const TexturesList& textures,
                    const std::string& shader, bool transparent);
    // ------------------------------------------------------------------------
    void addInstance(GESPMBuffer* mb, const TexturesList& textures,
                     const std::string& shader, bool transparent,
                     const GXMObjectData& data,
                     irr::scene::ISceneNode* skinning_node,
                     bool caster_only = false);
    // ------------------------------------------------------------------------
    /** Converts a streamed mesh buffer's geometry into the frame arena and adds
     *  it as its own single instance batch. */
    void addDynamicBuffer(GEGXMDriver* driver, GESPMBuffer* mb,
                          const TexturesList& textures,
                          const std::string& shader, bool transparent,
                          const GXMObjectData& data, bool caster_only);
    // ------------------------------------------------------------------------
    void fillObjectData(irr::scene::ISceneNode* node, int material_id,
                        GXMObjectData* out) const;
    // ------------------------------------------------------------------------
    /** Fits an orthographic cascade around the part of the view the shadows
     *  cover. Returns false when there is no sun to cast from. */
    bool computeShadowMatrix(GEGXMCameraSceneNode* cam);
    // ------------------------------------------------------------------------
    /** True when the batch has anything to draw in the pass \p transparent. A
     *  batch with no visible instances exists only to cast. */
    static bool isInPass(const Batch& batch, bool transparent)
    {
        return batch.m_visible_count > 0 &&
            batch.m_transparent == transparent;
    }

    // ------------------------------------------------------------------------
    void bindMaterialTextures(GEGXMDriver* driver, const Batch& batch,
                              const GEGXMPipeline& pipeline);
    // ------------------------------------------------------------------------
    void bindEnvironmentTextures(GEGXMDriver* driver);
    // ------------------------------------------------------------------------
    /** Writes the projection-view matrix, and for grass the wind, into a
     *  vertex uniform buffer shared by every draw using \p program. */
    void* getSharedVertexUniforms(GEGXMDriver* driver, GEGXMProgram* program,
                                  const irr::core::matrix4& projection_view);
    // ------------------------------------------------------------------------
    void* getSharedFragmentUniforms(GEGXMDriver* driver,
                                    GEGXMProgram* program,
                                    GEGXMCameraSceneNode* cam);
    // ------------------------------------------------------------------------
    /** Uploads a batch's per instance stream and issues the draw. */
    void drawBatch(GEGXMDriver* driver, const Batch& batch,
                   const GEGXMPipeline& pipeline, bool depth_only,
                   unsigned instance_count);
    // ------------------------------------------------------------------------
    void renderShadowPass(GEGXMDriver* driver, GEGXMCameraSceneNode* cam);
    // ------------------------------------------------------------------------
    void renderSkyBox(GEGXMDriver* driver, GEGXMCameraSceneNode* cam);
    // ------------------------------------------------------------------------
    void renderPass(GEGXMDriver* driver, GEGXMCameraSceneNode* cam,
                    bool transparent);

public:
    // ------------------------------------------------------------------------
    GEGXMDrawCall();
    // ------------------------------------------------------------------------
    ~GEGXMDrawCall();
    // ------------------------------------------------------------------------
    void prepare(GEGXMCameraSceneNode* cam);
    // ------------------------------------------------------------------------
    void addNode(irr::scene::ISceneNode* node);
    // ------------------------------------------------------------------------
    void addBillboardNode(irr::scene::ISceneNode* node,
                          irr::scene::ESCENE_NODE_TYPE node_type);
    // ------------------------------------------------------------------------
    void addSkyBox(irr::scene::ISceneNode* node);
    // ------------------------------------------------------------------------
    void addLightNode(irr::scene::ILightSceneNode* node);
    // ------------------------------------------------------------------------
    /** Sorts and culls what was gathered. Separate from render() so the scene
     *  manager can gather for every camera before any drawing starts. */
    void generate(GEGXMDriver* driver);
    // ------------------------------------------------------------------------
    void render(GEGXMDriver* driver, GEGXMCameraSceneNode* cam);
    // ------------------------------------------------------------------------
    void reset();
    // ------------------------------------------------------------------------
    /** Forgets the sun carried over between frames. Called when a draw call is
     *  recycled onto a different scene, so a track's sun cannot fit the cascade
     *  for a scene that has none. */
    void clearRetainedSun();
    // ------------------------------------------------------------------------
    unsigned getPolyCount() const                       { return m_polycount; }
};   // GEGXMDrawCall

}   // namespace GE

#endif   // _IRR_COMPILE_WITH_GXM_

#endif
