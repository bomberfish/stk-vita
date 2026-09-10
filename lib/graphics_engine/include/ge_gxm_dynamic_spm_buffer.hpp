#ifndef HEADER_GE_GXM_DYNAMIC_SPM_BUFFER_HPP
#define HEADER_GE_GXM_DYNAMIC_SPM_BUFFER_HPP

#include "IrrCompileConfig.h"

#ifdef _IRR_COMPILE_WITH_GXM_

#include "ge_spm_buffer.hpp"

namespace GE
{

/** A mesh buffer whose contents change every frame.
 *
 *  Skid marks, kart shadows and the rubber band all rebuild their geometry as
 *  the race runs, so they cannot live in the merged mesh cache, which is only
 *  repacked when a mesh is added or removed.
 *
 *  The Vulkan backend's equivalent, GEVulkanDynamicSPMBuffer, owns a
 *  double-buffered device allocation and uploads dirty ranges into it. Nothing
 *  like that is needed here: GEGXMDrawCall converts and copies the vertices
 *  straight into the driver's per-frame scratch arena when it gathers the node,
 *  and that memory is guaranteed to outlive the frame's GPU work. So all this
 *  class has to do is declare itself streamed, which is what tells the draw
 *  call to take that path instead of looking the buffer up in the cache. */
class GEGXMDynamicSPMBuffer : public GESPMBuffer
{
public:
    // ------------------------------------------------------------------------
    virtual irr::scene::E_HARDWARE_MAPPING getHardwareMappingHint_Vertex() const
                                             { return irr::scene::EHM_STREAM; }
    // ------------------------------------------------------------------------
    virtual irr::scene::E_HARDWARE_MAPPING getHardwareMappingHint_Index() const
                                             { return irr::scene::EHM_STREAM; }
    // ------------------------------------------------------------------------
    virtual void bindVertexIndexBuffer(VkCommandBuffer cmd)                  {}
    // ------------------------------------------------------------------------
    virtual void createVertexIndexBuffer()                                   {}
    // ------------------------------------------------------------------------
    virtual void destroyVertexIndexBuffer()                                  {}
};   // GEGXMDynamicSPMBuffer

}   // namespace GE

#endif   // _IRR_COMPILE_WITH_GXM_

#endif
