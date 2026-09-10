#ifndef HEADER_GE_GXM_LIMITS_HPP
#define HEADER_GE_GXM_LIMITS_HPP

#include <cstddef>
#include <cstdint>

namespace GE
{

/** Sizes shared between the Cg shaders and the C++ that feeds them.
 *
 *  Most of these are dictated by the GXM default uniform buffer, which is the
 *  only practical way to get uniforms to a shader on this hardware: it is
 *  DMA'd into the pipeline's secondary attribute registers before the shader
 *  runs, and there are 512 of those per stage once the shader compiler
 *  extension raises the limit from its default 128. One register is one float,
 *  so a stage's whole uniform footprint has to fit in 512 floats. */

/** Largest joint palette the shader compiler will accept, three float4 rows per
 *  joint. Not a constant: it depends on how many vertex uniform registers are
 *  available, which depends on whether SceShaccCgExt's module patches took
 *  effect - they do on hardware and they crash the compiler under Vita3K. So
 *  GEGXMShaderManager probes for it at startup and publishes the answer here.
 *
 *  40 joints is 480 of the 512 floats the raised limit gives, leaving room for
 *  the projection-view matrix. A mesh with more joints than the probe found is
 *  drawn with the palette clamped rather than dropped, and 0 means the compiler
 *  would not take a palette at all, in which case skinned meshes are drawn in
 *  bind pose. */
const unsigned GXM_MAX_JOINTS_CEILING = 40;

/** Result of the startup probe. Valid after GEGXMShaderManager::init(). */
unsigned getGXMMaxJoints();

/** False when no joint palette size compiled, so skinning is unavailable. */
inline bool gxmSupportsSkinning()          { return getGXMMaxJoints() > 0; }

/** Point and spot lights considered per draw. The Vulkan renderer allows 32,
 *  but each one is three float4s of fragment uniform and a full BRDF
 *  evaluation in a loop the SGX has to run per pixel, so the GXM renderer
 *  culls to the nearest few instead. */
const unsigned GXM_MAX_LIGHT = 8;

/** Resolution of the equirectangular environment panorama built from the
 *  skybox. Its mip chain doubles as the roughness levels for specular image
 *  based lighting, so the level count matters as much as the base size: 512x256
 *  gives ten levels, which is more roughness resolution than the eye can pick
 *  out. */
const unsigned GXM_ENV_WIDTH = 512;
const unsigned GXM_ENV_HEIGHT = 256;

/** Resolution of the cosine convolved irradiance panorama. Diffuse ambient is
 *  inherently smooth, so this is deliberately tiny - it is also what keeps the
 *  convolution, which is O(output x input), affordable on the CPU at track
 *  load. */
const unsigned GXM_IRRADIANCE_WIDTH = 32;
const unsigned GXM_IRRADIANCE_HEIGHT = 16;

/** Mip level of the panorama the irradiance convolution reads from. Integrating
 *  over a 32x16 reduction of the sky rather than the full 512x256 is the
 *  difference between a few million and a few billion operations, and the result
 *  is indistinguishable after a cosine lobe has been applied to it. */
const unsigned GXM_IRRADIANCE_SOURCE_LEVEL = 4;

/** Vertex layout the GXM renderer uploads meshes in.
 *
 *  It is not S3DVertexSkinnedMesh: that stores the normal and tangent as
 *  A2B10G10R10 packed into a u32, and SceGxmAttributeFormat has no packed
 *  10-bit format, so those two have to be expanded at upload time. S16N is
 *  used rather than S8N because 8-bit normals band visibly across the large
 *  smooth surfaces STK tracks are full of. */
struct GXMSPMVertex
{
    float m_position[3];        //  0: F32 x3
    int16_t m_normal[4];        // 12: S16N x4, w unused
    uint8_t m_color[4];         // 20: U8N x4, RGBA order
    int16_t m_uv[2];            // 24: F16 x2
    int16_t m_uv_two[2];        // 28: F16 x2
    int16_t m_tangent[4];       // 32: S16N x4, w carries the bitangent sign
};                              // 40 bytes

/** Per-vertex skinning data, kept in a second stream so that static meshes do
 *  not pay for it. Matches the layout GEGXMMeshCache writes. */
struct GXMSPMSkinning
{
    int16_t m_joint[4];         //  0: S16 x4
    int16_t m_weight[4];        //  8: F16 x4
};                              // 16 bytes

/** Per-instance data, read through a vertex stream with an instance index
 *  source rather than from a storage buffer, which GXM does not have.
 *
 *  Every field is a full float4 on purpose. A packed layout would save about
 *  40 bytes per instance, but the packing has to agree exactly with what the
 *  Cg compiler decides to do with a partially filled register, and a mistake
 *  there is invisible until it renders wrong on hardware. */
struct GXMObjectData
{
    float m_translation_hue[4];     //  0: xyz translation, w hue change
    float m_rotation[4];            // 16: rotation quaternion
    float m_scale_skinning[4];      // 32: xyz scale, w joint palette offset
    float m_texture_trans[4];       // 48: xy texture translation
    float m_custom_color[4];        // 64: custom vertex colour, 0..1
};                                  // 80 bytes

/** How the streams are numbered in every mesh drawing program. */
enum GXMVertexStreamIndex : unsigned
{
    GVSI_STATIC = 0,
    GVSI_SKINNING,
    GVSI_INSTANCE,
    GVSI_COUNT
};

/** Fragment texture units, fixed by the TEXUNITn semantics in the Cg sources.
 *  Units 0-4 are the material's own texture layers; the rest are the pipeline's
 *  own inputs, which is why materials may only use five layers on GXM against
 *  the eight the Vulkan renderer allows. */
enum GXMTextureUnit : unsigned
{
    GTU_MATERIAL_0 = 0,
    GTU_MATERIAL_1,
    GTU_MATERIAL_2,
    GTU_MATERIAL_3,
    GTU_MATERIAL_4,
    GTU_DIFFUSE_ENV,
    GTU_SPECULAR_ENV,
    GTU_SHADOW_MAP,
    GTU_COUNT
};

const unsigned GXM_MATERIAL_TEXTURE_COUNT = 5;

/** Number of display buffers. Three lets the GPU stay a frame ahead of the
 *  display without the CPU ever waiting on a buffer that is still on screen. */
const unsigned GXM_DISPLAY_BUFFER_COUNT = 3;

}   // namespace GE

#endif
