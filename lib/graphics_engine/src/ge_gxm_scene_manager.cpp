#include "ge_gxm_scene_manager.hpp"

#ifdef _IRR_COMPILE_WITH_GXM_

#include "ge_gxm_camera_scene_node.hpp"
#include "ge_gxm_draw_call.hpp"
#include "ge_gxm_driver.hpp"
#include "ge_gxm_mesh_cache.hpp"
#include "ge_main.hpp"
#include "ge_vulkan_animated_mesh_scene_node.hpp"
#include "ge_vulkan_mesh_scene_node.hpp"

#include "../source/Irrlicht/os.h"

#include "IBillboardSceneNode.h"
#include "ILightSceneNode.h"

#include <sstream>

namespace GE
{
// ----------------------------------------------------------------------------
GEGXMSceneManager::GEGXMSceneManager(irr::video::IVideoDriver* driver,
                                     irr::io::IFileSystem* fs,
                                     irr::gui::ICursorControl* cursor_control,
                                     irr::gui::IGUIEnvironment* gui_environment)
                 : CSceneManager(driver, fs, cursor_control,
                                 new GEGXMMeshCache(), gui_environment)
{
    // CSceneManager grabbed the cache.
    getMeshCache()->drop();
}   // GEGXMSceneManager

// ----------------------------------------------------------------------------
GEGXMSceneManager::~GEGXMSceneManager()
{
}   // ~GEGXMSceneManager

// ----------------------------------------------------------------------------
void GEGXMSceneManager::clear()
{
    irr::scene::CSceneManager::clear();
}   // clear

// ----------------------------------------------------------------------------
irr::scene::ICameraSceneNode* GEGXMSceneManager::addCameraSceneNode(
                                              irr::scene::ISceneNode* parent,
                                        const irr::core::vector3df& position,
                                          const irr::core::vector3df& lookat,
                                              irr::s32 id, bool make_active)
{
    if (!parent)
        parent = this;

    irr::scene::ICameraSceneNode* node = new GEGXMCameraSceneNode(parent, this,
        id, position, lookat);
    if (make_active)
        setActiveCamera(node);
    node->drop();
    return node;
}   // addCameraSceneNode

// ----------------------------------------------------------------------------
irr::scene::IAnimatedMeshSceneNode*
    GEGXMSceneManager::addAnimatedMeshSceneNode(
                                        irr::scene::IAnimatedMesh* mesh,
                                        irr::scene::ISceneNode* parent,
                                        irr::s32 id,
                                        const irr::core::vector3df& position,
                                        const irr::core::vector3df& rotation,
                                        const irr::core::vector3df& scale,
                                        bool also_add_if_mesh_pointer_zero)
{
    // Only SPM meshes can be drawn: anything else never reaches the packed
    // vertex buffer the draw call renders out of.
    if (!also_add_if_mesh_pointer_zero && (!mesh ||
        mesh->getMeshType() != irr::scene::EAMT_SPM))
        return NULL;

    if (!parent)
        parent = this;

    // Shared with the Vulkan backend: the class holds the skinning matrices and
    // the joint hierarchy, neither of which is backend specific.
    irr::scene::IAnimatedMeshSceneNode* node =
        new GEVulkanAnimatedMeshSceneNode(mesh, parent, this, id, position,
        rotation, scale);
    node->drop();
    node->setMesh(mesh);
    return node;
}   // addAnimatedMeshSceneNode

// ----------------------------------------------------------------------------
irr::scene::IMeshSceneNode* GEGXMSceneManager::addMeshSceneNode(
                                        irr::scene::IMesh* mesh,
                                        irr::scene::ISceneNode* parent,
                                        irr::s32 id,
                                        const irr::core::vector3df& position,
                                        const irr::core::vector3df& rotation,
                                        const irr::core::vector3df& scale,
                                        bool also_add_if_mesh_pointer_zero)
{
    if (!also_add_if_mesh_pointer_zero && !mesh)
        return NULL;

    bool convert_irrlicht_mesh = false;
    if (mesh)
    {
        for (unsigned i = 0; i < mesh->getMeshBufferCount(); i++)
        {
            irr::scene::IMeshBuffer* b = mesh->getMeshBuffer(i);
            if (b->getVertexType() != irr::video::EVT_SKINNED_MESH)
            {
                if (!getGEConfig()->m_convert_irrlicht_mesh)
                {
                    // Falls back to irrlicht's own node, which this renderer
                    // will then skip in addNode(). Matches the Vulkan
                    // backend's behaviour rather than crashing on a mesh the
                    // pipeline cannot take.
                    return irr::scene::CSceneManager::addMeshSceneNode(mesh,
                        parent, id, position, rotation, scale,
                        also_add_if_mesh_pointer_zero);
                }
                convert_irrlicht_mesh = true;
                break;
            }
        }
    }

    if (!parent)
        parent = this;

    if (convert_irrlicht_mesh)
    {
        irr::scene::IAnimatedMesh* spm = convertIrrlichtMeshToSPM(mesh);
        std::stringstream oss;
        oss << (uint64_t)spm;
        getMeshCache()->addMesh(oss.str().c_str(), spm);
        mesh = spm;
    }

    GEVulkanMeshSceneNode* gxm_node = new GEVulkanMeshSceneNode(mesh, parent,
        this, id, position, rotation, scale);
    irr::scene::IMeshSceneNode* node = gxm_node;
    node->drop();

    if (convert_irrlicht_mesh)
    {
        gxm_node->setRemoveFromMeshCache(true);
        mesh->drop();
    }
    return node;
}   // addMeshSceneNode

// ----------------------------------------------------------------------------
void GEGXMSceneManager::drawAllInternal()
{
    // Any mesh loaded since the last frame has to be packed into the merged
    // vertex buffer before anything can be drawn from it.
    static_cast<GEGXMMeshCache*>(getMeshCache())->updateCache();

    GEGXMCameraSceneNode* cam =
        static_cast<GEGXMCameraSceneNode*>(getActiveCamera());
    OnAnimate(irr::os::Timer::getTime());
    if (cam == NULL)
        return;

    auto it = m_draw_calls.find(cam);
    if (it == m_draw_calls.end())
        return;

    cam->render();
    it->second->prepare(cam);
    // Walks the graph; every visible node comes back through
    // registerNodeForRendering() below.
    OnRegisterSceneNode();
    GEGXMDriver* driver = static_cast<GEGXMDriver*>(getVideoDriver());
    it->second->generate(driver);
    it->second->render(driver, cam);
    it->second->reset();
}   // drawAllInternal

// ----------------------------------------------------------------------------
void GEGXMSceneManager::drawAll(irr::u32 flags)
{
    drawAllInternal();
}   // drawAll

// ----------------------------------------------------------------------------
irr::u32 GEGXMSceneManager::registerNodeForRendering(
                              irr::scene::ISceneNode* node,
                              irr::scene::E_SCENE_NODE_RENDER_PASS pass)
{
    GEGXMCameraSceneNode* cam =
        static_cast<GEGXMCameraSceneNode*>(getActiveCamera());
    if (cam == NULL)
        return 0;
    auto it = m_draw_calls.find(cam);
    if (it == m_draw_calls.end())
        return 0;
    GEGXMDrawCall* dc = it->second.get();

    if (node->getType() == irr::scene::ESNT_SKY_BOX)
    {
        dc->addSkyBox(node);
        return 1;
    }
    if (node->getType() == irr::scene::ESNT_LIGHT)
    {
        dc->addLightNode(static_cast<irr::scene::ILightSceneNode*>(node));
        return 1;
    }
    if (node->getType() == irr::scene::ESNT_BILLBOARD ||
        node->getType() == irr::scene::ESNT_PARTICLE_SYSTEM)
    {
        dc->addBillboardNode(node, node->getType());
        return 1;
    }
    // Meshes register themselves once per pass; the draw call sorts materials
    // itself, so only take them from the solid pass to avoid adding each node
    // several times.
    if ((node->getType() == irr::scene::ESNT_ANIMATED_MESH ||
        node->getType() == irr::scene::ESNT_MESH) &&
        pass != irr::scene::ESNRP_SOLID)
        return 0;

    dc->addNode(node);
    return 1;
}   // registerNodeForRendering

// ----------------------------------------------------------------------------
void GEGXMSceneManager::addDrawCall(GEGXMCameraSceneNode* cam)
{
    GEGXMDriver* driver = static_cast<GEGXMDriver*>(getVideoDriver());
    // Reusing a cached draw call keeps its compiled pipeline map, which is
    // worth a lot on this backend: patching a program costs USSE memory and a
    // trip through the shader patcher.
    m_draw_calls[cam] = driver->getDrawCallFromCache();
}   // addDrawCall

// ----------------------------------------------------------------------------
void GEGXMSceneManager::removeDrawCall(GEGXMCameraSceneNode* cam)
{
    auto it = m_draw_calls.find(cam);
    if (it == m_draw_calls.end())
        return;
    GEGXMDriver* driver = static_cast<GEGXMDriver*>(getVideoDriver());
    it->second->reset();
    // The draw call is about to be handed to a different camera, possibly in a
    // scene with no sun at all, so the sun it carries between frames has to go.
    it->second->clearRetainedSun();
    driver->addDrawCallToCache(it->second);
    m_draw_calls.erase(it);
}   // removeDrawCall

}   // namespace GE

#endif   // _IRR_COMPILE_WITH_GXM_
