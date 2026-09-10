#include "ge_gxm_shader.hpp"

#ifdef _IRR_COMPILE_WITH_GXM_

#include "ge_gxm_limits.hpp"

#include <unordered_map>

/* The Cg source of every shader the GXM renderer uses.
 *
 * These live in the binary rather than in data/shaders for two reasons. The
 * shaders are compiled at runtime by the SceShaccCg module rather than ahead of
 * time (psp2cgc needs a licensed SDK), so they have to be available before any
 * file system path is known to be readable; and data/ on a Vita is copied to
 * ux0:data/stk/ by hand, so a shader that shipped as a data file would silently
 * be missing on every install that was not re-copied after an update.
 *
 * Conventions used throughout, all of them chosen to remove ambiguity rather
 * than for brevity, because this code cannot be stepped through on the target:
 *
 *  - Matrices are passed as float4[4] and combined by hand with MUL4/MUL3.
 *    A Cg float4x4 uniform would work too, but whether mul(v, m) or mul(m, v)
 *    is the right order then depends on the compiler's matrix storage order,
 *    and getting it wrong produces a black screen with no diagnostic. The
 *    arrays are laid out exactly as irrlicht's core::matrix4 stores them, so
 *    MUL4(v, m) is irrlicht's transformVect().
 *  - Vertex attributes are looked up by parameter name at runtime to fill in
 *    SceGxmVertexAttribute::regIndex, so the names below are load bearing:
 *    see GEGXMDrawCall::buildLayouts().
 *  - Varyings are matched between stages by semantic, so a fragment shader
 *    must declare the same semantic the vertex shader wrote.
 */

namespace GE
{
namespace
{
// ============================================================================
/** Shared helpers. Prepended to every shader. */
const char* COMMON = R"(
#define MUL4(v, m) (v.x * m[0] + v.y * m[1] + v.z * m[2] + v.w * m[3])
#define MUL3_POINT(v, m) (v.x * m[0].xyz + v.y * m[1].xyz + v.z * m[2].xyz + m[3].xyz)
#define MUL3_DIR(v, m) (v.x * m[0].xyz + v.y * m[1].xyz + v.z * m[2].xyz)

float3 rotateVector(float4 quat, float3 v)
{
    return v + 2.0 * cross(cross(v, quat.xyz) + quat.w * v, quat.xyz);
}

float4 getWorldPosition(float3 origin, float4 rotation, float3 scale,
                        float3 local_pos)
{
    float3 p = local_pos * scale;
    p = rotateVector(rotation, p);
    return float4(p + origin, 1.0);
}

/* Direction to equirectangular texture coordinate.
 *
 * The environment maps are 2D panoramas rather than cube maps. A cube map would
 * be the natural choice, but sceGxmTextureInitCube() only accepts the swizzled
 * layout and the order its six faces and their mip chains are expected in is
 * not something this code can verify on hardware; an equirectangular 2D texture
 * goes through exactly the same upload path as every other texture in the game,
 * mip chain included. The cost is some distortion at the poles, which at this
 * resolution is not visible, and one texel of seam where u wraps, which is why
 * every environment sample below uses an explicit level of detail rather than
 * letting the hardware derive one across the seam. */
float2 dirToEquirect(float3 d)
{
    d = normalize(d);
    float u = atan2(d.x, -d.z) * 0.15915494 + 0.5;
    float v = acos(clamp(d.y, -1.0, 1.0)) * 0.31830989;
    return float2(u, v);
}
)";

// ============================================================================
/** Tone mapping, hue rotation and the PBR BRDF, ported from the GE GLSL
 *  shaders so that the GXM renderer matches the Vulkan one's look. */
const char* COMMON_COLOR = R"(
float3 convertColor(float3 c)
{
#ifdef GXM_IBL
    return (c * (6.5 * c + 0.45)) / (c * (5.0 * c + 1.75) + 0.05);
#else
    return (c * (7.0 * c + 0.75)) / (c * (5.0 * c + 1.75) + 0.05);
#endif
}

float3 rgbToHsv(float3 c)
{
    float4 K = float4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    float4 p = lerp(float4(c.b, c.g, K.w, K.z), float4(c.g, c.b, K.x, K.y),
        step(c.b, c.g));
    float4 q = lerp(float4(p.x, p.y, p.w, c.r), float4(c.r, p.y, p.z, p.x),
        step(p.x, c.r));
    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return float3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

float3 hsvToRgb(float3 c)
{
    float4 K = float4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    float3 p = abs(frac(float3(c.x, c.x, c.x) + K.xyz) * 6.0 - K.www);
    return c.z * lerp(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

/* Rotates the hue of a texel towards target_hue, used for the per-kart colour
 * scheme. Mirrors solid.frag/alphatest.frag in the GLSL renderer, including the
 * different saturation constant between the PBR and non-PBR paths. */
float4 applyHueChange(float4 tex_color, float target_hue, bool use_alpha_mask)
{
    if (target_hue <= 0.0)
        return tex_color;
    float3 old_hsv = rgbToHsv(tex_color.rgb);
    if (use_alpha_mask)
    {
        float mask = tex_color.a;
        float mask_step = step(mask, 0.5);
#ifdef GXM_PBR
        float saturation = mask * 2.5;
#else
        float saturation = mask * 1.825;
#endif
        float2 new_xy = lerp(float2(old_hsv.x, old_hsv.y),
            float2(target_hue, max(old_hsv.y, saturation)),
            float2(mask_step, mask_step));
        float3 new_color = hsvToRgb(float3(new_xy.x, new_xy.y, old_hsv.z));
        return float4(new_color, 1.0);
    }
    float3 new_color = hsvToRgb(float3(target_hue, old_hsv.y, old_hsv.z));
    return float4(new_color, tex_color.a);
}
)";

// ============================================================================
const char* COMMON_PBR = R"(
float2 F_AB(float perceptual_roughness, float NdotV)
{
    float4 c0 = float4(-1.0, -0.0275, -0.572, 0.022);
    float4 c1 = float4(1.0, 0.0425, 1.04, -0.04);
    float4 r = perceptual_roughness * c0 + c1;
    float a004 = min(r.x * r.x, pow(2.0, -9.28 * NdotV)) * r.x + r.y;
    return float2(-1.04, 1.04) * a004 + r.zw;
}

float F_Schlick(float f0, float f90, float VdotH)
{
    return lerp(f0, f90, pow(1.0 - VdotH, 5.0));
}

float Fd_Burley(float roughness, float NdotV, float NdotL, float LdotH)
{
    float f90 = 0.5 + 2.0 * roughness * LdotH * LdotH;
    float light_scatter = F_Schlick(1.0, f90, NdotL);
    float view_scatter = F_Schlick(1.0, f90, NdotV);
    return light_scatter * view_scatter;
}

float D_GGX(float roughness, float NdotH)
{
    float one_minus_NdotH_squared = 1.0 - NdotH * NdotH;
    float a = NdotH * roughness;
    float k = roughness / (one_minus_NdotH_squared + a * a);
    return k * k * (1.0 / 3.14159265359);
}

float V_Smith_GGX_Correlated(float roughness, float NdotV, float NdotL)
{
    return 0.5 / lerp(2.0 * NdotL * NdotV, NdotL + NdotV, roughness);
}

float3 fresnel(float3 f0, float f90, float VdotH)
{
    return f0 + (f90 - f0) * pow(1.0 - VdotH, 5.0);
}

float3 envBRDFApprox(float3 F0, float2 F_ab)
{
    return F0 * F_ab.x + F_ab.y;
}

float perceptualRoughnessToRoughness(float perceptual_roughness)
{
    float roughness = clamp(perceptual_roughness, 0.089, 1.0);
    return roughness * roughness;
}

float3 environmentLight(float3 irradiance, float3 radiance, float roughness,
                        float3 diffuse_color, float2 F_ab, float3 F0,
                        float F90, float NdotV)
{
    float3 Fr = max(float3(1.0 - roughness, 1.0 - roughness,
        1.0 - roughness), F0) - F0;
    float3 kS = F0 + Fr * pow(1.0 - NdotV, 5.0);
    float Ess = F_ab.x + F_ab.y;
    float3 FssEss = kS * Ess * F90;
    float Ems = 1.0 - Ess;
    float3 Favg = F0 + (1.0 - F0) / 21.0;
    float3 Fms = FssEss * Favg / (1.0 - Ems * Favg);
    float3 FmsEms = Fms * Ems;
    float3 Edss = 1.0 - (FssEss + FmsEms);
    float3 kD = diffuse_color * Edss;
    return (FmsEms + kD) * irradiance + FssEss * radiance;
}

float3 PBRLight(float3 normal, float3 eyedir, float3 lightdir, float3 color,
                float perceptual_roughness, float metallic)
{
    float NdotV = max(dot(normal, eyedir), 0.0001);
    float NdotL = clamp(dot(normal, lightdir), 0.0, 1.0);
    float2 F_ab = F_AB(perceptual_roughness, NdotV);
    float3 H = normalize(eyedir + lightdir);
    float NdotH = clamp(dot(normal, H), 0.0, 1.0);
    float LdotH = clamp(dot(lightdir, H), 0.0, 1.0);
    float3 diffuse_color = color * (1.0 - metallic);
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), color, metallic);
    float F90 = clamp(dot(F0, float3(16.5, 16.5, 16.5)), 0.0, 1.0);
    float roughness = perceptualRoughnessToRoughness(perceptual_roughness);
    float3 diffuse = diffuse_color * Fd_Burley(roughness, NdotV, NdotL, NdotH);
    float D = D_GGX(roughness, NdotH);
    float V = V_Smith_GGX_Correlated(roughness, NdotV, NdotL);
    float3 F = fresnel(F0, F90, LdotH);
    float3 specular = D * V * F * (1.0 + F0 * (1.0 / F_ab.x - 1.0));
    return NdotL * (diffuse + specular);
}

/* Sun most representative point, from the Frostbite PBR paper. Softens the
 * sun's specular highlight into a disc of the right angular size. */
float3 sunDirection(float3 R, float3 sun_direction, float sun_angle_tan_half)
{
    float DdotR = dot(sun_direction, R);
    float3 S = normalize(R - DdotR * sun_direction);
    float t2 = 1.0 + sun_angle_tan_half * sun_angle_tan_half;
    float2 sin_cos = float2(2.0 * sun_angle_tan_half, 2.0 - t2) / t2;
    float factor = step(DdotR, sin_cos.y);
    return lerp(R, normalize(sun_direction * sin_cos.y + S * sin_cos.x),
        factor);
}
)";

// ============================================================================
/** The uniform block every lit fragment shader shares. Kept in one place so
 *  GEGXMDrawCall::uploadFragmentUniforms() has exactly one layout to write. */
const char* COMMON_LIT_UNIFORMS = R"(
#define GXM_LIT_UNIFORMS \
    uniform float4 u_view_matrix[4], \
    uniform float4 u_inverse_view_matrix[4], \
    uniform float4 u_sun_color, \
    uniform float4 u_sun_direction, \
    uniform float4 u_ambient_color, \
    uniform float4 u_skytop_color, \
    uniform float4 u_fog_color, \
    uniform float4 u_misc, \
    uniform float4 u_lights[GXM_MAX_LIGHT_VEC4], \
    uniform float4 u_shadow_matrix[4], \
    uniform sampler2D u_diffuse_env : TEXUNIT5, \
    uniform sampler2D u_specular_env : TEXUNIT6, \
    uniform sampler2D u_shadow_map : TEXUNIT7

/* u_misc packs the handful of scalars that would otherwise each waste a
 * float4 register: fog density, the highest specular mip index, how many of
 * u_lights are live, and one over the shadow map resolution. */
#define GXM_FOG_DENSITY     u_misc.x
#define GXM_SPECULAR_LEVELS u_misc.y
#define GXM_LIGHT_COUNT     u_misc.z
#define GXM_SHADOW_TEXEL    u_misc.w

/* Each light is three float4s: position+radius, colour+inverse square range,
 * and the spotlight direction/scale/offset. Same packing as LightData in the
 * GLSL renderer so GEGXMLightHandler can share its maths. */
#define GXM_LIGHT_POS(i)   u_lights[(i) * 3 + 0]
#define GXM_LIGHT_COLOR(i) u_lights[(i) * 3 + 1]
#define GXM_LIGHT_DIR(i)   u_lights[(i) * 3 + 2]

float3 accumulateLights(float3 diffuse_color, float3 normal, float3 xpos,
                        float3 eyedir, float perceptual_roughness,
                        float metallic, float4 u_lights[GXM_MAX_LIGHT_VEC4],
                        float4 u_view_matrix[4], float light_count)
{
    float3 accumulated = float3(0.0, 0.0, 0.0);
    for (int i = 0; i < GXM_MAX_LIGHT; i++)
    {
        if (float(i) >= light_count)
            break;
        float4 pos_radius = GXM_LIGHT_POS(i);
        float4 color_range = GXM_LIGHT_COLOR(i);
        float3 view_light_pos = MUL3_POINT(pos_radius.xyz, u_view_matrix);
        float3 light_to_frag = view_light_pos - xpos;
        float distance_sq = dot(light_to_frag, light_to_frag);
        if (distance_sq * color_range.w > 1.0)
            continue;
        float dist = sqrt(distance_sq);
        float3 L = light_to_frag / max(dist, 0.0001);
        float sattenuation = 1.0;
        float4 dir_scale_offset = GXM_LIGHT_DIR(i);
        float sscale = dir_scale_offset.z;
        if (sscale != 0.0)
        {
            float3 sdir = float3(dir_scale_offset.x, dir_scale_offset.y, 0.0);
            sdir.z = sqrt(max(1.0 - dot(sdir, sdir), 0.0)) * sign(sscale);
            sdir = MUL3_DIR(sdir, u_view_matrix);
            sattenuation = clamp(dot(-sdir, L) * abs(sscale) +
                dir_scale_offset.w, 0.0, 1.0);
        }
        float3 diffuse_specular = PBRLight(normal, eyedir, L, diffuse_color,
            perceptual_roughness, metallic);
        float attenuation = 20.0 / (1.0 + distance_sq);
        float radius = pos_radius.w;
        attenuation *= (radius - dist) / max(radius, 0.0001);
        attenuation *= sattenuation * sattenuation;
        accumulated += color_range.xyz * max(attenuation, 0.0) *
            diffuse_specular;
    }
    return accumulated;
}

/* One ortho shadow cascade with a 3x3 PCF kernel. The depth stored in the
 * shadow map is the GXM depth buffer value, so the comparison is against the
 * same 0..1 range after projecting the world position through the light's
 * matrix. A slope independent constant bias is enough here because the caster
 * pass already applies a polygon offset. */
float getShadowFactor(float3 world_position, float4 u_shadow_matrix[4],
                      sampler2D u_shadow_map, float texel, float NdotL)
{
#ifdef GXM_SHADOW
    float4 clip = MUL4(float4(world_position, 1.0), u_shadow_matrix);
    if (clip.w <= 0.0)
        return 1.0;
    float3 proj = clip.xyz / clip.w;
    // Outside the cascade's depth range there is nothing to compare against.
    // Without this a receiver beyond the far plane compares against the cleared
    // far value and comes out fully shadowed. The range is 0..1 because
    // irrlicht's orthographic projection is built the Direct3D way; the viewport
    // passes z through unchanged, so this is also the stored depth.
    if (proj.z < 0.0 || proj.z > 1.0)
        return 1.0;
    float2 uv = float2(proj.x * 0.5 + 0.5, 0.5 - proj.y * 0.5);
    /* Outside the cascade there is no shadow information, so fade out rather
     * than clamping, which would smear the border texel across the world. */
    float2 fade = saturate((0.5 - abs(uv - 0.5)) * 16.0);
    float edge = min(fade.x, fade.y);
    if (edge <= 0.0)
        return 1.0;
    // No remap: the caster pass stored exactly this value.
    float depth = proj.z;
    float bias = 0.0015 + 0.004 * (1.0 - NdotL);
    float reference = depth - bias;
    float lit = 0.0;
    for (int y = -1; y <= 1; y++)
    {
        for (int x = -1; x <= 1; x++)
        {
            float2 offset = float2(float(x), float(y)) * texel;
            float stored = tex2D(u_shadow_map, uv + offset).r;
            lit += step(reference, stored);
        }
    }
    lit *= (1.0 / 9.0);
    return lerp(1.0, lit, edge);
#else
    return 1.0;
#endif
}

/* The forward equivalent of the GLSL renderer's handlePBR(): sun, image based
 * ambient and every culled point light, tone mapped to display range.
 *  pbr.x = gloss, pbr.y = metalness, pbr.z = emissive. */
float3 handlePBR(float3 diffuse_color, float3 pbr, float3 world_position,
                 float3 world_normal, GXM_LIT_UNIFORMS)
{
    float3 xpos = MUL3_POINT(world_position, u_view_matrix);
    float3 eyedir = -normalize(xpos);
    float3 normal = normalize(MUL3_DIR(world_normal, u_view_matrix));
    float perceptual_roughness = 1.0 - pbr.x;
    float roughness = perceptualRoughnessToRoughness(perceptual_roughness);
    float metallic = pbr.y;
    float emissive = pbr.z;

    float3 reflection = reflect(-eyedir, normal);
    float3 irradiance = float3(0.0, 0.0, 0.0);
    float3 radiance = float3(0.0, 0.0, 0.0);
#ifdef GXM_IBL
    float3 world_reflection = MUL3_DIR(reflection, u_inverse_view_matrix);
    /* The irradiance panorama is already cosine convolved on the CPU, so a
     * single tap at its only level is the whole diffuse ambient term. */
    irradiance = tex2Dlod(u_diffuse_env,
        float4(dirToEquirect(world_normal), 0.0, 0.0)).rgb;
    /* Roughness selects a mip of the panorama. Its mip chain is a plain box
     * reduction rather than a GGX prefilter, which is the same approximation
     * the GLSL renderer's degraded IBL path makes and is a good deal cheaper
     * than importance sampling per pixel would be here. */
    radiance = tex2Dlod(u_specular_env, float4(dirToEquirect(world_reflection),
        0.0, perceptual_roughness * GXM_SPECULAR_LEVELS)).rgb;
#endif

    /* The sun direction arrives in world space; the lighting maths below is in
     * view space, so rotate it once here. */
    float3 view_sun_dir = normalize(MUL3_DIR(u_sun_direction.xyz,
        u_view_matrix));
    float3 lightdir = sunDirection(reflection, view_sun_dir,
        u_sun_direction.w);

    float NdotV = max(dot(normal, eyedir), 0.0001);
    float NdotL = clamp(dot(normal, lightdir), 0.0, 1.0);
    float2 F_ab = F_AB(perceptual_roughness, NdotV);
    float3 H = normalize(eyedir + lightdir);
    float NdotH = clamp(dot(normal, H), 0.0, 1.0);
    float LdotH = clamp(dot(lightdir, H), 0.0, 1.0);
    float3 base_diffuse = diffuse_color * (1.0 - metallic);
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), diffuse_color, metallic);
    float F90 = clamp(dot(F0, float3(16.5, 16.5, 16.5)), 0.0, 1.0);

    float3 diffuse = base_diffuse * Fd_Burley(roughness, NdotV, NdotL, NdotH);
    float D = D_GGX(roughness, NdotH);
    float V = V_Smith_GGX_Correlated(roughness, NdotV, NdotL);
    float3 F = fresnel(F0, F90, LdotH);
    float3 specular = D * V * F * (1.0 + F0 * (1.0 / F_ab.x - 1.0));
    float3 sunlight = NdotL * (diffuse + specular);

    float shadow = getShadowFactor(world_position, u_shadow_matrix,
        u_shadow_map, GXM_SHADOW_TEXEL, NdotL);
    sunlight *= shadow;

    float3 ambient = u_ambient_color.rgb * 0.4;
    float3 diffuse_ambient = envBRDFApprox(base_diffuse, F_AB(1.0, NdotV));
    float3 specular_ambient = F90 * envBRDFApprox(F0, F_ab);
    float3 environment;
#ifdef GXM_IBL
    environment = environmentLight(irradiance, radiance, roughness,
        base_diffuse, F_ab, F0, F90, NdotV);
#else
    environment = u_skytop_color.rgb * ambient * base_diffuse;
#endif

    // Not "emit": that is a reserved word in Cg (geometry shader stream output)
    // and naming a local after it fails the whole shader to compile.
    float3 emitted = emissive * diffuse_color * 4.0;
    float3 mixed_color = u_sun_color.rgb * sunlight + environment + emitted +
        (diffuse_ambient + specular_ambient) * ambient;
    mixed_color += accumulateLights(diffuse_color, normal, xpos, eyedir,
        perceptual_roughness, metallic, u_lights, u_view_matrix,
        GXM_LIGHT_COUNT);
    return convertColor(mixed_color);
}
)";

// ============================================================================
/** Mesh vertex shader. Covers static and skinned meshes and the grass wind
 *  animation, selected with GXM_SKINNING / GXM_GRASS. */
const char* VS_SPM = R"(
void main(
    float3 a_position          : POSITION,
    float4 a_normal            : NORMAL,
    float4 a_color             : COLOR0,
    float2 a_uv                : TEXCOORD0,
    float2 a_uv_two            : TEXCOORD1,
    float4 a_tangent           : TANGENT,
#ifdef GXM_SKINNING
    float4 a_joint             : BLENDINDICES,
    float4 a_weight            : BLENDWEIGHT,
    uniform float4 u_joint_matrices[GXM_MAX_JOINTS_VEC4],
#endif
#ifdef GXM_GRASS
    uniform float4 u_wind_direction,
#endif
    float4 i_translation_hue   : TEXCOORD2,
    float4 i_rotation          : TEXCOORD3,
    float4 i_scale_skinning    : TEXCOORD4,
    float4 i_texture_trans     : TEXCOORD5,
    float4 i_custom_color      : TEXCOORD6,
    uniform float4 u_projection_view[4],
    out float4 v_position      : POSITION,
    out float4 o_color         : COLOR0,
    out float4 o_uv            : TEXCOORD0,
    out float4 o_world_hue     : TEXCOORD1
#ifdef GXM_PBR
    , out float3 o_normal      : TEXCOORD2
#ifdef GXM_NORMAL_MAP
    , out float3 o_tangent     : TEXCOORD3
    , out float3 o_bitangent   : TEXCOORD4
#endif
#endif
    )
{
    float3 local_position = a_position;
    float3 local_normal = a_normal.xyz;
    float3 local_tangent = a_tangent.xyz;

#ifdef GXM_SKINNING
    /* The palette holds one affine transform per joint as three float4 rows;
     * blending the rows is equivalent to blending the matrices and costs three
     * multiply-adds per joint instead of four. */
    int base = int(i_scale_skinning.w + 0.5) * 3;
    float4 row0 = float4(0.0, 0.0, 0.0, 0.0);
    float4 row1 = float4(0.0, 0.0, 0.0, 0.0);
    float4 row2 = float4(0.0, 0.0, 0.0, 0.0);
    for (int j = 0; j < 4; j++)
    {
        float w = a_weight[j];
        if (w <= 0.0)
            continue;
        int idx = base + int(max(a_joint[j], 0.0) + 0.5) * 3;
        row0 += w * u_joint_matrices[idx + 0];
        row1 += w * u_joint_matrices[idx + 1];
        row2 += w * u_joint_matrices[idx + 2];
    }
    float4 hp = float4(local_position, 1.0);
    local_position = float3(dot(row0, hp), dot(row1, hp), dot(row2, hp));
    float4 hn = float4(local_normal, 0.0);
    local_normal = float3(dot(row0, hn), dot(row1, hn), dot(row2, hn));
    float4 ht = float4(local_tangent, 0.0);
    local_tangent = float3(dot(row0, ht), dot(row1, ht), dot(row2, ht));
#endif

    float3 origin = i_translation_hue.xyz;
#ifdef GXM_GRASS
    /* Bend the blade by an amount that depends on its height and on the red
     * vertex channel, which the track exporter uses as a stiffness mask. */
    float3 offset = sin(u_wind_direction.xyz * (a_position.y * 0.1));
    offset += cos(u_wind_direction.xyz) * 0.7;
    origin += offset * a_color.r;
#endif

    float4 world_position = getWorldPosition(origin, i_rotation,
        i_scale_skinning.xyz, local_position);
    v_position = MUL4(world_position, u_projection_view);
    o_world_hue = float4(world_position.xyz, i_translation_hue.w);
#ifdef GXM_GRASS
    o_color = float4(1.0, 1.0, 1.0, 1.0);
#else
    o_color = a_color * i_custom_color;
#endif
    o_uv = float4(a_uv + i_texture_trans.xy, a_uv_two);
#ifdef GXM_PBR
    float3 world_normal = rotateVector(i_rotation, local_normal);
    o_normal = world_normal;
#ifdef GXM_NORMAL_MAP
    float3 world_tangent = rotateVector(i_rotation, local_tangent);
    o_tangent = world_tangent;
    o_bitangent = cross(world_normal, world_tangent) * a_tangent.w;
#endif
#endif
}
)";

// ============================================================================
/** Depth only mesh vertex shader, used for the shadow caster pass. Writes
 *  nothing but clip position and, when the caster is alpha tested, the uv. */
const char* VS_SPM_DEPTH = R"(
void main(
    float3 a_position          : POSITION,
#ifdef GXM_ALPHA_TEST
    float2 a_uv                : TEXCOORD0,
#endif
#ifdef GXM_GRASS
    float4 a_color             : COLOR0,
    uniform float4 u_wind_direction,
#endif
#ifdef GXM_SKINNING
    float4 a_joint             : BLENDINDICES,
    float4 a_weight            : BLENDWEIGHT,
    uniform float4 u_joint_matrices[GXM_MAX_JOINTS_VEC4],
#endif
    float4 i_translation_hue   : TEXCOORD2,
    float4 i_rotation          : TEXCOORD3,
    float4 i_scale_skinning    : TEXCOORD4,
    float4 i_texture_trans     : TEXCOORD5,
    uniform float4 u_projection_view[4],
    out float4 v_position      : POSITION
#ifdef GXM_ALPHA_TEST
    , out float4 o_uv          : TEXCOORD0
#endif
    )
{
    float3 local_position = a_position;
#ifdef GXM_SKINNING
    int base = int(i_scale_skinning.w + 0.5) * 3;
    float4 row0 = float4(0.0, 0.0, 0.0, 0.0);
    float4 row1 = float4(0.0, 0.0, 0.0, 0.0);
    float4 row2 = float4(0.0, 0.0, 0.0, 0.0);
    for (int j = 0; j < 4; j++)
    {
        float w = a_weight[j];
        if (w <= 0.0)
            continue;
        int idx = base + int(max(a_joint[j], 0.0) + 0.5) * 3;
        row0 += w * u_joint_matrices[idx + 0];
        row1 += w * u_joint_matrices[idx + 1];
        row2 += w * u_joint_matrices[idx + 2];
    }
    float4 hp = float4(local_position, 1.0);
    local_position = float3(dot(row0, hp), dot(row1, hp), dot(row2, hp));
#endif
    float3 origin = i_translation_hue.xyz;
#ifdef GXM_GRASS
    // Identical to the wind in VS_SPM. Without it the blade casts its shadow
    // from where it would be standing still, so the shadow does not move with
    // the grass.
    float3 offset = sin(u_wind_direction.xyz * (a_position.y * 0.1));
    offset += cos(u_wind_direction.xyz) * 0.7;
    origin += offset * a_color.r;
#endif
    float4 world_position = getWorldPosition(origin, i_rotation,
        i_scale_skinning.xyz, local_position);
    v_position = MUL4(world_position, u_projection_view);
#ifdef GXM_ALPHA_TEST
    o_uv = float4(a_uv + i_texture_trans.xy, 0.0, 0.0);
#endif
}
)";

// ============================================================================
const char* FS_DEPTH_ONLY = R"(
float4 main(
#ifdef GXM_ALPHA_TEST
    float4 o_uv : TEXCOORD0,
    uniform sampler2D u_texture_0 : TEXUNIT0
#endif
    ) : COLOR
{
#ifdef GXM_ALPHA_TEST
    if (tex2D(u_texture_0, o_uv.xy).a < 0.5)
        discard;
#endif
    return float4(1.0, 1.0, 1.0, 1.0);
}
)";

// ============================================================================
/** solid, alphatest and unlit share one body; the differences are an alpha
 *  test and whether the PBR channels come from a texture or are fixed. */
const char* FS_SOLID = R"(
float4 main(
    float4 o_color     : COLOR0,
    float4 o_uv        : TEXCOORD0,
    float4 o_world_hue : TEXCOORD1,
#ifdef GXM_PBR
    float3 o_normal    : TEXCOORD2,
#ifdef GXM_NORMAL_MAP
    float3 o_tangent   : TEXCOORD3,
    float3 o_bitangent : TEXCOORD4,
    uniform sampler2D u_texture_3 : TEXUNIT3,
#endif
    uniform sampler2D u_texture_2 : TEXUNIT2,
    GXM_LIT_UNIFORMS,
#endif
    uniform sampler2D u_texture_0 : TEXUNIT0) : COLOR
{
    float4 tex_color = tex2D(u_texture_0, o_uv.xy);
#ifdef GXM_ALPHA_TEST
    if (tex_color.a * o_color.a < 0.5)
        discard;
    tex_color = applyHueChange(tex_color, o_world_hue.w, false);
#else
    tex_color = applyHueChange(tex_color, o_world_hue.w, true);
#endif

    float3 diffuse_color = tex_color.rgb * o_color.rgb;
#ifndef GXM_PBR
    return float4(diffuse_color, 1.0);
#else
    float3 normal = normalize(o_normal);
#ifdef GXM_NORMAL_MAP
    float3 tangent_space_normal =
        2.0 * tex2D(u_texture_3, o_uv.xy).xyz - 1.0;
    float3 t = normalize(o_tangent);
    float3 b = normalize(o_bitangent);
    normal = normalize(t * tangent_space_normal.x + b * tangent_space_normal.y +
        normal * tangent_space_normal.z);
#endif
#ifdef GXM_UNLIT
    float3 pbr = float3(0.0, 0.0, 0.4);
#else
    float3 pbr = tex2D(u_texture_2, o_uv.xy).xyz;
#endif
    return float4(handlePBR(diffuse_color, pbr, o_world_hue.xyz, normal,
        u_view_matrix, u_inverse_view_matrix, u_sun_color, u_sun_direction,
        u_ambient_color, u_skytop_color, u_fog_color, u_misc, u_lights,
        u_shadow_matrix, u_diffuse_env, u_specular_env, u_shadow_map), 1.0);
#endif
}
)";

// ============================================================================
const char* FS_DECAL = R"(
float4 main(
    float4 o_color     : COLOR0,
    float4 o_uv        : TEXCOORD0,
    float4 o_world_hue : TEXCOORD1,
#ifdef GXM_PBR
    float3 o_normal    : TEXCOORD2,
    uniform sampler2D u_texture_2 : TEXUNIT2,
    GXM_LIT_UNIFORMS,
#endif
    uniform sampler2D u_texture_0 : TEXUNIT0,
    uniform sampler2D u_texture_1 : TEXUNIT1) : COLOR
{
    float4 color = tex2D(u_texture_0, o_uv.xy);
    float4 layer_two = tex2D(u_texture_1, o_uv.zw);
    float3 final_color = layer_two.a * layer_two.rgb +
        color.rgb * (1.0 - layer_two.a);
#ifndef GXM_PBR
    return float4(final_color, 1.0);
#else
    float3 normal = normalize(o_normal);
    float3 pbr = tex2D(u_texture_2, o_uv.xy).xyz;
    return float4(handlePBR(final_color, pbr, o_world_hue.xyz, normal,
        u_view_matrix, u_inverse_view_matrix, u_sun_color, u_sun_direction,
        u_ambient_color, u_skytop_color, u_fog_color, u_misc, u_lights,
        u_shadow_matrix, u_diffuse_env, u_specular_env, u_shadow_map), 1.0);
#endif
}
)";

// ============================================================================
/** Terrain splatting: a control texture in layer 1 selects between four
 *  detail textures. The distance dependent blend towards a lower sampling
 *  frequency is what keeps the tiling from being obvious in the distance. */
const char* FS_SPLATTING = R"(
float4 main(
    float4 o_uv        : TEXCOORD0,
    float4 o_world_hue : TEXCOORD1,
    float3 o_normal    : TEXCOORD2,
    uniform sampler2D u_texture_1 : TEXUNIT1,
    uniform sampler2D u_texture_2 : TEXUNIT2,
    uniform sampler2D u_texture_3 : TEXUNIT3,
    uniform sampler2D u_texture_4 : TEXUNIT4,
    uniform sampler2D u_texture_0 : TEXUNIT0,
    GXM_LIT_UNIFORMS) : COLOR
{
    float3 view_position = MUL3_POINT(o_world_hue.xyz, u_view_matrix);
    float cam_dist = length(view_position);
    float mitigation = clamp(pow(cam_dist * 0.01, 2.0), 0.0, 1.0);

    float4 splatting = tex2D(u_texture_1, o_uv.zw);
    float2 uv_low = o_uv.xy * 0.5;
    float4 detail0 = lerp(tex2D(u_texture_2, o_uv.xy),
        tex2D(u_texture_2, uv_low), mitigation);
    float4 detail1 = lerp(tex2D(u_texture_3, o_uv.xy),
        tex2D(u_texture_3, uv_low), mitigation);
    float4 detail2 = lerp(tex2D(u_texture_4, o_uv.xy),
        tex2D(u_texture_4, uv_low), mitigation);
    float4 detail3 = lerp(tex2D(u_texture_0, o_uv.xy),
        tex2D(u_texture_0, uv_low), mitigation);

    float rest = max(0.0, 1.0 - splatting.r - splatting.g - splatting.b);
    float4 splatted = splatting.r * detail0 + splatting.g * detail1 +
        splatting.b * detail2 + rest * detail3;

    float3 normal = normalize(o_normal);
    return float4(handlePBR(splatted.rgb, float3(0.0, 0.0, 0.0),
        o_world_hue.xyz, normal,
        u_view_matrix, u_inverse_view_matrix, u_sun_color, u_sun_direction,
        u_ambient_color, u_skytop_color, u_fog_color, u_misc, u_lights,
        u_shadow_matrix, u_diffuse_env, u_specular_env, u_shadow_map), 1.0);
}
)";

// ============================================================================
/** Transparent and additive materials. Premultiplied so that one blend state
 *  covers both alpha blending and additive, exactly as the GLSL renderer
 *  does. */
const char* FS_TRANSPARENT = R"(
float4 main(
    float4 o_color : COLOR0,
    float4 o_uv    : TEXCOORD0,
    uniform sampler2D u_texture_0 : TEXUNIT0) : COLOR
{
    float4 color = tex2D(u_texture_0, o_uv.xy) * o_color;
    float3 mixed_color = color.rgb;
#ifdef GXM_PBR
    // Same curve as the opaque pass, which convertColor() switches on GXM_IBL,
    // so a transparent surface sits in the same colour space as what is behind
    // it rather than a slightly different one.
    mixed_color = convertColor(mixed_color);
#endif
    return float4(mixed_color * color.a, color.a);
}
)";

// ============================================================================
/** 2D pass: irrlicht hands us screen space pixel coordinates, which the shader
 *  turns into clip space so no projection matrix upload is needed per quad. */
const char* VS_2D = R"(
void main(
    float2 a_position : POSITION,
    float4 a_color    : COLOR0,
    float2 a_uv       : TEXCOORD0,
    uniform float4 u_screen,
    out float4 v_position : POSITION,
    out float4 o_color    : COLOR0,
    out float2 o_uv       : TEXCOORD0)
{
    v_position = float4(a_position.x * u_screen.x - 1.0,
        1.0 - a_position.y * u_screen.y, 0.0, 1.0);
    o_color = a_color;
    o_uv = a_uv;
}
)";

// ============================================================================
const char* FS_2D = R"(
float4 main(
    float4 o_color : COLOR0,
    float2 o_uv    : TEXCOORD0,
    uniform sampler2D u_texture_0 : TEXUNIT0) : COLOR
{
    return tex2D(u_texture_0, o_uv) * o_color;
}
)";

// ============================================================================
const char* FS_2D_UNTEXTURED = R"(
float4 main(float4 o_color : COLOR0) : COLOR
{
    return o_color;
}
)";

// ============================================================================
/** Fullscreen pass geometry. Two triangles in a static vertex buffer, so the
 *  vertex shader is a straight pass through. */
const char* VS_FULLSCREEN = R"(
void main(
    float2 a_position : POSITION,
    float2 a_uv       : TEXCOORD0,
    out float4 v_position : POSITION,
    out float2 o_uv       : TEXCOORD0)
{
    v_position = float4(a_position, 0.0, 1.0);
    o_uv = a_uv;
}
)";

// ============================================================================
/** Skybox. Reconstructs a world space view ray from the inverse
 *  projection-view matrix so the cube can be sampled without any geometry. */
const char* FS_SKYBOX = R"(
float4 main(
    float2 o_uv : TEXCOORD0,
    uniform float4 u_ray_tl,
    uniform float4 u_ray_tr,
    uniform float4 u_ray_bl,
    uniform float4 u_ray_br,
    uniform sampler2D u_skybox : TEXUNIT0) : COLOR
{
    /* The four corner rays come from the CPU, already unprojected, in o_uv
     * order: (0,0) top left, (1,0) top right, (0,1) bottom left, (1,1) bottom
     * right. A pinhole camera's image plane is planar, so the direction through
     * any pixel is the bilinear blend of the corner directions - no matrix, no
     * per pixel divide, and no chance of a clip space convention drifting
     * between the C++ and the shader. */
    float3 top = u_ray_tl.xyz + (u_ray_tr.xyz - u_ray_tl.xyz) * o_uv.x;
    float3 bottom = u_ray_bl.xyz + (u_ray_br.xyz - u_ray_bl.xyz) * o_uv.x;
    float3 dir = top + (bottom - top) * o_uv.y;
    /* Explicit level 0: the u wrap makes the derived level of detail collapse
     * along the seam, which would draw a one pixel wide blurred stripe down the
     * middle of the sky. */
    return float4(tex2Dlod(u_skybox,
        float4(dirToEquirect(dir), 0.0, 0.0)).rgb, 1.0);
}
)";

// ============================================================================
/** Bright pass for bloom: keep what is above the threshold, in quarter
 *  resolution. The four taps make it a downsample and a filter in one pass,
 *  which matters a lot for Vita fill rate. */
const char* FS_BLOOM_BRIGHT = R"(
float4 main(
    float2 o_uv : TEXCOORD0,
    uniform float4 u_texel,
    uniform sampler2D u_texture_0 : TEXUNIT0) : COLOR
{
    float3 sum = float3(0.0, 0.0, 0.0);
    sum += tex2D(u_texture_0, o_uv + float2(-u_texel.x, -u_texel.y)).rgb;
    sum += tex2D(u_texture_0, o_uv + float2( u_texel.x, -u_texel.y)).rgb;
    sum += tex2D(u_texture_0, o_uv + float2(-u_texel.x,  u_texel.y)).rgb;
    sum += tex2D(u_texture_0, o_uv + float2( u_texel.x,  u_texel.y)).rgb;
    float3 color = sum * 0.25;
    float luma = dot(color, float3(0.2126, 0.7152, 0.0722));
    float threshold = u_texel.z;
    float knee = max(threshold * 0.5, 0.0001);
    /* Soft knee so that a texel drifting across the threshold does not pop. */
    float weight = clamp((luma - threshold + knee) / (2.0 * knee), 0.0, 1.0);
    weight *= weight;
    return float4(color * weight, 1.0);
}
)";

// ============================================================================
/** Separable 9 tap gaussian, run once horizontally and once vertically.
 *  u_texel.xy is the step, so the same program serves both directions. */
const char* FS_BLUR = R"(
float4 main(
    float2 o_uv : TEXCOORD0,
    uniform float4 u_texel,
    uniform sampler2D u_texture_0 : TEXUNIT0) : COLOR
{
    float2 step_uv = u_texel.xy;
    float3 sum = tex2D(u_texture_0, o_uv).rgb * 0.2270270270;
    sum += tex2D(u_texture_0, o_uv + step_uv * 1.3846153846).rgb * 0.3162162162;
    sum += tex2D(u_texture_0, o_uv - step_uv * 1.3846153846).rgb * 0.3162162162;
    sum += tex2D(u_texture_0, o_uv + step_uv * 3.2307692308).rgb * 0.0702702703;
    sum += tex2D(u_texture_0, o_uv - step_uv * 3.2307692308).rgb * 0.0702702703;
    return float4(sum, 1.0);
}
)";

// ============================================================================
/** Final composite into the display buffer: scene plus bloom. */
const char* FS_COMPOSITE = R"(
float4 main(
    float2 o_uv : TEXCOORD0,
    uniform float4 u_settings,
    uniform sampler2D u_texture_0 : TEXUNIT0,
    uniform sampler2D u_texture_1 : TEXUNIT1) : COLOR
{
    float3 color = tex2D(u_texture_0, o_uv).rgb;
#ifdef GXM_BLOOM
    color += tex2D(u_texture_1, o_uv).rgb * u_settings.x;
#endif
    return float4(color, 1.0);
}
)";

// ============================================================================
/** Paints a flat colour. GXM can clear depth for free - the tile buffer starts
 *  at the depth surface's background value - but there is no equivalent for
 *  colour, so clearing the colour buffer means rasterising over it. On a tile
 *  based GPU that is cheap: the write never leaves the tile. */
const char* FS_CLEAR = R"(
float4 main(uniform float4 u_color) : COLOR
{
    return u_color;
}
)";

// ============================================================================
/** Copies a 2D texture, used to resolve a render target texture into the
 *  format STK's RTT consumers expect and for the screenshot path. */
const char* FS_COPY = R"(
float4 main(
    float2 o_uv : TEXCOORD0,
    uniform sampler2D u_texture_0 : TEXUNIT0) : COLOR
{
    return tex2D(u_texture_0, o_uv);
}
)";

// ----------------------------------------------------------------------------
/** Assembles the shader table on first use.
 *
 *  Each entry is the concatenation of the helper blocks a shader needs. The
 *  numeric limits come from ge_gxm_limits.hpp so that the C++ side that sizes
 *  the uniform uploads and the Cg side that declares the arrays can never
 *  disagree. */
const std::unordered_map<std::string, std::string>& getShaderTable()
{
    static std::unordered_map<std::string, std::string> table;
    if (!table.empty())
        return table;

    // No limits block here: it is prepended per request by getProgram(), because
    // the joint count is only known after the startup probe and is part of the
    // shader cache key.
    const std::string limits = "";

    const std::string common = limits + COMMON;
    const std::string color = common + COMMON_COLOR;
    const std::string lit = color + COMMON_PBR + COMMON_LIT_UNIFORMS;

    table["spm.vert"] = common + VS_SPM;
    table["spm_depth.vert"] = common + VS_SPM_DEPTH;
    table["depth_only.frag"] = limits + FS_DEPTH_ONLY;
    table["solid.frag"] = lit + FS_SOLID;
    table["decal.frag"] = lit + FS_DECAL;
    table["splatting.frag"] = lit + FS_SPLATTING;
    table["transparent.frag"] = color + FS_TRANSPARENT;
    table["2d.vert"] = limits + VS_2D;
    table["2d.frag"] = limits + FS_2D;
    table["2d_untextured.frag"] = limits + FS_2D_UNTEXTURED;
    table["fullscreen.vert"] = limits + VS_FULLSCREEN;
    table["skybox.frag"] = common + FS_SKYBOX;
    table["bloom_bright.frag"] = limits + FS_BLOOM_BRIGHT;
    table["blur.frag"] = limits + FS_BLUR;
    table["composite.frag"] = limits + FS_COMPOSITE;
    table["clear.frag"] = limits + FS_CLEAR;
    table["copy.frag"] = limits + FS_COPY;
    return table;
}   // getShaderTable
}   // namespace

// ----------------------------------------------------------------------------
std::string getGXMLimitsDefines()
{
    // Prepended to every shader. The joint count comes from the startup probe,
    // so it is part of the source and therefore part of the cache key: a cache
    // written when the extension worked is not reused when it does not.
    const unsigned joints = getGXMMaxJoints();
    return
        "#define GXM_MAX_JOINTS " + std::to_string(joints) + "\n"
        "#define GXM_MAX_JOINTS_VEC4 " + std::to_string(joints * 3) + "\n"
        "#define GXM_MAX_LIGHT " + std::to_string(GXM_MAX_LIGHT) + "\n"
        "#define GXM_MAX_LIGHT_VEC4 " +
            std::to_string(GXM_MAX_LIGHT * 3) + "\n";
}   // getGXMLimitsDefines

// ----------------------------------------------------------------------------
const std::string& getGXMShaderSource(const std::string& name)
{
    static const std::string empty;
    const auto& table = getShaderTable();
    auto it = table.find(name);
    if (it == table.end())
        return empty;
    return it->second;
}   // getGXMShaderSource

}   // namespace GE

#endif   // _IRR_COMPILE_WITH_GXM_
