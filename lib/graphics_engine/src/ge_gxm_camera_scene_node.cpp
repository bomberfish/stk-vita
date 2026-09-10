#include "ge_gxm_camera_scene_node.hpp"

#ifdef _IRR_COMPILE_WITH_GXM_

#include "ge_gxm_scene_manager.hpp"
#include "ge_main.hpp"

namespace GE
{
// ----------------------------------------------------------------------------
GEGXMCameraSceneNode::GEGXMCameraSceneNode(irr::scene::ISceneNode* parent,
                                           irr::scene::ISceneManager* mgr,
                                           irr::s32 id,
                                        const irr::core::vector3df& position,
                                        const irr::core::vector3df& lookat)
                    : CCameraSceneNode(parent, mgr, id, position, lookat)
{
    static_cast<GEGXMSceneManager*>(SceneManager)->addDrawCall(this);
}   // GEGXMCameraSceneNode

// ----------------------------------------------------------------------------
GEGXMCameraSceneNode::~GEGXMCameraSceneNode()
{
    static_cast<GEGXMSceneManager*>(SceneManager)->removeDrawCall(this);
}   // ~GEGXMCameraSceneNode

// ----------------------------------------------------------------------------
void GEGXMCameraSceneNode::render()
{
    irr::scene::CCameraSceneNode::render();

    m_matrices.m_view_matrix = ViewArea.getTransform(irr::video::ETS_VIEW);
    // No clip space fixup at all, unlike the Vulkan path which flips Y and
    // halves Z. Irrlicht builds its projections the Direct3D way, so z already
    // arrives in 0..1 - exactly what the depth buffer stores - and
    // GEGXMDriver::applyViewport() passes it through untouched while flipping Y
    // in the viewport transform instead of in the matrix. getShadowFactor() in
    // the shaders depends on this: it compares proj.z directly against the
    // stored depth.
    m_matrices.m_projection_matrix =
        ViewArea.getTransform(irr::video::ETS_PROJECTION);

    irr::core::matrix4 mat;
    m_matrices.m_view_matrix.getInverse(mat);
    m_matrices.m_inverse_view_matrix = mat;

    m_matrices.m_projection_view_matrix =
        m_matrices.m_projection_matrix * m_matrices.m_view_matrix;
    m_matrices.m_projection_view_matrix.getInverse(
        m_matrices.m_inverse_projection_view_matrix);
}   // render

// ----------------------------------------------------------------------------
irr::core::matrix4 GEGXMCameraSceneNode::getPVM() const
{
    // The unmodified matrices, for frustum culling.
    return ViewArea.getTransform(irr::video::ETS_PROJECTION) *
        ViewArea.getTransform(irr::video::ETS_VIEW);
}   // getPVM

}   // namespace GE

#endif   // _IRR_COMPILE_WITH_GXM_
