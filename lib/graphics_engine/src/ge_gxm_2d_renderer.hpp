#ifndef HEADER_GE_GXM_2D_RENDERER_HPP
#define HEADER_GE_GXM_2D_RENDERER_HPP

#include "IrrCompileConfig.h"

#ifdef _IRR_COMPILE_WITH_GXM_

#include <cstdint>

#include "ITexture.h"
#include "S3DVertex.h"
#include "rect.h"

namespace GE
{
class GEGXMDriver;

/** Batched 2D drawing for the GUI.
 *
 *  Nothing is drawn when STK asks for it. Every quad is appended to a vertex
 *  buffer and the whole lot is replayed at the end of the frame, for two
 *  reasons: the GUI has to land on top of the post processing composite, which
 *  has not run yet at the time the widgets are submitted; and STK's GUI submits
 *  thousands of individual quads per frame, which as thousands of draws would
 *  be entirely CPU bound.
 *
 *  Batches break only when the texture changes or when the scissor rectangle
 *  changes, so a screen of text in one font atlas is one draw. Clipping is
 *  applied by trimming the quad's geometry and texture coordinates on the CPU
 *  rather than by a scissor rectangle, which keeps consecutive clipped widgets
 *  in the same batch. */
namespace GEGXM2dRenderer
{
// ----------------------------------------------------------------------------
void init(GEGXMDriver* driver);
// ----------------------------------------------------------------------------
void destroy();
// ----------------------------------------------------------------------------
/** Drops everything accumulated so far. Called at the top of each frame. */
void clear();
// ----------------------------------------------------------------------------
/** Emits every batch into the scene that is currently open. */
void render();
// ----------------------------------------------------------------------------
/** \param colors four corner colours in the order upper left, lower left,
 *         lower right, upper right, matching irrlicht's draw2DImage(); NULL
 *         means opaque white. */
void addQuad(const irr::video::ITexture* texture,
             const irr::core::rect<irr::s32>& dest_rect,
             const irr::core::rect<irr::s32>& source_rect,
             const irr::core::rect<irr::s32>* clip_rect,
             const irr::video::SColor* colors);
// ----------------------------------------------------------------------------
/** Raw triangle list in screen coordinates, for the GUI paths that build their
 *  own geometry (text billboards, the profiler graph). */
void addVerticesIndices(irr::video::S3DVertex* vertices,
                        unsigned vertices_count, uint16_t* indices,
                        unsigned indices_count,
                        const irr::video::ITexture* texture);
// ----------------------------------------------------------------------------
/** Forgets a texture that is about to be destroyed, so a batch cannot outlive
 *  the texture it references. */
void handleDeletedTexture(const irr::video::ITexture* texture);
};   // GEGXM2dRenderer

}   // namespace GE

#endif   // _IRR_COMPILE_WITH_GXM_

#endif
