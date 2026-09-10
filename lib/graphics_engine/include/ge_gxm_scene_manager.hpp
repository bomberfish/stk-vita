#ifndef HEADER_GE_GXM_SCENE_MANAGER_HPP
#define HEADER_GE_GXM_SCENE_MANAGER_HPP

#include "IrrCompileConfig.h"

#ifdef _IRR_COMPILE_WITH_GXM_

#include "../source/Irrlicht/CSceneManager.h"

#include <map>
#include <memory>

namespace GE
{
class GEGXMCameraSceneNode;
class GEGXMDrawCall;

/** Replaces irrlicht's scene manager for the GXM driver.
 *
 *  Same job as GEVulkanSceneManager: irrlicht's own drawAll() walks the graph
 *  issuing a draw per mesh buffer per node with its own material state, which
 *  neither this renderer's material model nor its instancing can express. So
 *  the traversal is intercepted, every visible node is handed to a per camera
 *  GEGXMDrawCall to be batched, and the whole scene is issued at once.
 *
 *  The mesh and animated mesh node classes are shared with the Vulkan backend
 *  unchanged - they hold no Vulkan state, only the SPM mesh and, for animated
 *  ones, the skinning matrices. */
class GEGXMSceneManager : public irr::scene::CSceneManager
{
private:
    std::map<GEGXMCameraSceneNode*, std::unique_ptr<GEGXMDrawCall> >
        m_draw_calls;

    // ------------------------------------------------------------------------
    void drawAllInternal();

public:
    // ------------------------------------------------------------------------
    GEGXMSceneManager(irr::video::IVideoDriver* driver,
                      irr::io::IFileSystem* fs,
                      irr::gui::ICursorControl* cursor_control,
                      irr::gui::IGUIEnvironment* gui_environment);
    // ------------------------------------------------------------------------
    ~GEGXMSceneManager();
    // ------------------------------------------------------------------------
    virtual void clear();
    // ------------------------------------------------------------------------
    virtual irr::scene::ICameraSceneNode* addCameraSceneNode(
        irr::scene::ISceneNode* parent = 0,
        const irr::core::vector3df& position = irr::core::vector3df(0, 0, 0),
        const irr::core::vector3df& lookat = irr::core::vector3df(0, 0, 100),
        irr::s32 id = -1, bool make_active = true);
    // ------------------------------------------------------------------------
    virtual irr::scene::IAnimatedMeshSceneNode* addAnimatedMeshSceneNode(
        irr::scene::IAnimatedMesh* mesh, irr::scene::ISceneNode* parent = 0,
        irr::s32 id = -1,
        const irr::core::vector3df& position = irr::core::vector3df(0, 0, 0),
        const irr::core::vector3df& rotation = irr::core::vector3df(0, 0, 0),
        const irr::core::vector3df& scale =
            irr::core::vector3df(1.0f, 1.0f, 1.0f),
        bool also_add_if_mesh_pointer_zero = false);
    // ------------------------------------------------------------------------
    virtual irr::scene::IMeshSceneNode* addMeshSceneNode(
        irr::scene::IMesh* mesh, irr::scene::ISceneNode* parent = 0,
        irr::s32 id = -1,
        const irr::core::vector3df& position = irr::core::vector3df(0, 0, 0),
        const irr::core::vector3df& rotation = irr::core::vector3df(0, 0, 0),
        const irr::core::vector3df& scale =
            irr::core::vector3df(1.0f, 1.0f, 1.0f),
        bool also_add_if_mesh_pointer_zero = false);
    // ------------------------------------------------------------------------
    virtual void drawAll(irr::u32 flags = 0xFFFFFFFF);
    // ------------------------------------------------------------------------
    virtual irr::u32 registerNodeForRendering(irr::scene::ISceneNode* node,
        irr::scene::E_SCENE_NODE_RENDER_PASS pass =
            irr::scene::ESNRP_AUTOMATIC);
    // ------------------------------------------------------------------------
    void addDrawCall(GEGXMCameraSceneNode* cam);
    // ------------------------------------------------------------------------
    void removeDrawCall(GEGXMCameraSceneNode* cam);
};   // GEGXMSceneManager

}   // namespace GE

#endif   // _IRR_COMPILE_WITH_GXM_

#endif
