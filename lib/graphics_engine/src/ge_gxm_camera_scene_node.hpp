#ifndef HEADER_GE_GXM_CAMERA_SCENE_NODE_HPP
#define HEADER_GE_GXM_CAMERA_SCENE_NODE_HPP

#include "IrrCompileConfig.h"

#ifdef _IRR_COMPILE_WITH_GXM_

#include "../source/Irrlicht/CCameraSceneNode.h"

namespace GE
{

/** The camera matrices the GXM shaders read, in the exact order the uniform
 *  uploads expect.
 *
 *  Unlike the Vulkan backend this is not a uniform buffer object: GXM has no
 *  practical equivalent, so the draw code copies the members it needs into each
 *  draw's default uniform buffer. Keeping them precomputed here still saves
 *  recomputing an inverse per draw. */
struct GEGXMCameraMatrices
{
    irr::core::matrix4 m_view_matrix;
    irr::core::matrix4 m_projection_matrix;
    irr::core::matrix4 m_inverse_view_matrix;
    irr::core::matrix4 m_projection_view_matrix;
    irr::core::matrix4 m_inverse_projection_view_matrix;
};   // GEGXMCameraMatrices

class GEGXMCameraSceneNode : public irr::scene::CCameraSceneNode
{
private:
    GEGXMCameraMatrices m_matrices;

    irr::core::rect<irr::s32> m_viewport;

public:
    // ------------------------------------------------------------------------
    GEGXMCameraSceneNode(irr::scene::ISceneNode* parent,
                         irr::scene::ISceneManager* mgr, irr::s32 id,
          const irr::core::vector3df& position = irr::core::vector3df(0, 0, 0),
         const irr::core::vector3df& lookat = irr::core::vector3df(0, 0, 100));
    // ------------------------------------------------------------------------
    ~GEGXMCameraSceneNode();
    // ------------------------------------------------------------------------
    virtual void render();
    // ------------------------------------------------------------------------
    void setViewPort(const irr::core::rect<irr::s32>& area)
                                                         { m_viewport = area; }
    // ------------------------------------------------------------------------
    const irr::core::rect<irr::s32>& getViewPort() const { return m_viewport; }
    // ------------------------------------------------------------------------
    irr::core::matrix4 getPVM() const;
    // ------------------------------------------------------------------------
    const GEGXMCameraMatrices* getMatrices() const     { return &m_matrices; }
};   // GEGXMCameraSceneNode

}   // namespace GE

#endif   // _IRR_COMPILE_WITH_GXM_

#endif
