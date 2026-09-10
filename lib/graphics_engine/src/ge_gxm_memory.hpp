#ifndef HEADER_GE_GXM_MEMORY_HPP
#define HEADER_GE_GXM_MEMORY_HPP

#include "IrrCompileConfig.h"

#ifdef _IRR_COMPILE_WITH_GXM_

#include <psp2/gxm.h>
#include <psp2/kernel/sysmem.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace GE
{

/** Which physical pool a GPU allocation comes out of.
 *
 *  The Vita has two kinds of memory the GPU can read. LPDDR is the same main
 *  memory the CPU runs from; CDRAM is a separate 128 MB bank with roughly
 *  twice the GPU read bandwidth but painfully slow uncached CPU access (it is
 *  write-combined at best, and reads go straight to the bus).
 *
 *  The rule of thumb the driver follows is: anything the CPU rewrites every
 *  frame goes in LPDDR, anything written once and then sampled or rasterised
 *  into many times goes in CDRAM.
 */
enum GEGXMMemoryPool : unsigned
{
    /** Uncached LPDDR. CPU writes land in memory without an explicit cache
     *  flush, so this is what per-frame uniform, vertex and index data uses. */
    GGMP_LPDDR_UNCACHED = 0,
    /** Cached LPDDR. Faster for the CPU to read back, but every CPU write has
     *  to be flushed out of the data cache before the GPU can see it. Used for
     *  data the CPU reads more than it writes (screenshot readback). */
    GGMP_LPDDR_CACHED,
    /** Video memory. Textures, render targets and the merged mesh buffer. */
    GGMP_CDRAM,
    GGMP_COUNT
};

/** A suballocating heap over a small number of large SceKernel memory blocks.
 *
 *  Going to sceKernelAllocMemBlock() for every texture is not workable: LPDDR
 *  blocks are rounded up to 4 KB and CDRAM blocks to 256 KB, the number of
 *  blocks a process may hold is limited, and each one has to be handed to
 *  sceGxmMapMemory() separately. So the driver reserves a handful of large
 *  blocks and cuts allocations out of them here.
 *
 *  The free list is a pair of maps: m_blocks orders every block (free or not)
 *  by offset so neighbours can be coalesced in O(log n) on free, and
 *  m_free_blocks indexes just the free ones by size so a best fit is a single
 *  lower_bound(). Allocations larger than half a chunk get a dedicated
 *  memory block instead of fragmenting a shared one.
 */
class GEGXMHeap
{
private:
    struct Block
    {
        size_t m_size;
        bool m_free;
    };

    struct Chunk
    {
        SceUID m_uid;
        uint8_t* m_base;
        size_t m_size;
        bool m_dedicated;
        /** offset -> block. Contiguous and gapless by construction. */
        std::map<size_t, Block> m_blocks;
        /** size -> offset, free blocks only. */
        std::multimap<size_t, size_t> m_free_blocks;
    };

    struct Location
    {
        unsigned m_chunk;
        size_t m_offset;
    };

    std::vector<Chunk> m_chunks;

    /** Base address of a live allocation -> where it came from. */
    std::unordered_map<uintptr_t, Location> m_allocations;

    std::recursive_mutex m_mutex;

    SceKernelMemBlockType m_block_type;

    SceGxmMemoryAttribFlags m_attrib;

    size_t m_granularity;

    size_t m_default_chunk_size;

    size_t m_used, m_reserved;

    std::string m_name;

    // ------------------------------------------------------------------------
    int addChunk(size_t size, bool dedicated);
    // ------------------------------------------------------------------------
    void removeFreeBlock(Chunk& c, size_t offset, size_t size);
    // ------------------------------------------------------------------------
    void insertFreeBlock(Chunk& c, size_t offset, size_t size);
public:
    // ------------------------------------------------------------------------
    GEGXMHeap(const std::string& name, SceKernelMemBlockType block_type,
              SceGxmMemoryAttribFlags attrib, size_t granularity,
              size_t default_chunk_size);
    // ------------------------------------------------------------------------
    ~GEGXMHeap();
    // ------------------------------------------------------------------------
    void* allocate(size_t size, size_t alignment);
    // ------------------------------------------------------------------------
    bool free(void* ptr);
    // ------------------------------------------------------------------------
    bool owns(void* ptr) const;
    // ------------------------------------------------------------------------
    void destroy();
    // ------------------------------------------------------------------------
    size_t getUsed() const                                   { return m_used; }
    // ------------------------------------------------------------------------
    size_t getReserved() const                           { return m_reserved; }
    // ------------------------------------------------------------------------
    const std::string& getName() const                       { return m_name; }
};   // GEGXMHeap

/** The process-wide GPU allocator.
 *
 *  Everything the GPU reads or writes - uniform buffers, vertex data, texture
 *  pixels, render target surfaces, the shader patcher's own scratch memory -
 *  is handed out from here so that it is guaranteed to be mapped into the
 *  GPU's address space by sceGxmMapMemory().
 */
namespace GEGXMMemory
{
// ----------------------------------------------------------------------------
/** Creates the heaps. Must run before sceGxm is asked to touch any memory. */
void init();
// ----------------------------------------------------------------------------
void destroy();
// ----------------------------------------------------------------------------
bool isInitialised();
// ----------------------------------------------------------------------------
/** \param pool which physical bank to take the memory from.
 *  \param alignment must be a power of two. GXM has hard alignment rules per
 *         use (16 bytes for textures, 4 for colour surfaces, 4 KB for USSE
 *         code) which the callers pass in here.
 *  \return NULL when out of memory; callers are expected to cope rather than
 *          abort, because running out of CDRAM mid-race should cost a texture,
 *          not the session. */
void* allocate(GEGXMMemoryPool pool, size_t size, size_t alignment);
// ----------------------------------------------------------------------------
/** Frees a pointer returned by allocate(), from any pool. Ignores NULL. */
void free(void* ptr);
// ----------------------------------------------------------------------------
/** Allocates GPU memory that also has USSE (shader code) space mapped for the
 *  vertex pipeline, filling in the USSE offset the shader patcher wants. */
void* allocateVertexUsse(size_t size, unsigned* usse_offset);
// ----------------------------------------------------------------------------
void freeVertexUsse(void* ptr);
// ----------------------------------------------------------------------------
void* allocateFragmentUsse(size_t size, unsigned* usse_offset);
// ----------------------------------------------------------------------------
void freeFragmentUsse(void* ptr);
// ----------------------------------------------------------------------------
/** Bytes handed out and bytes reserved from the kernel, per pool. Used by the
 *  driver's memory report in the log and by the debug overlay. */
void getStats(GEGXMMemoryPool pool, size_t* used, size_t* reserved);
// ----------------------------------------------------------------------------
void logStats();
};   // GEGXMMemory

}   // namespace GE

#endif   // _IRR_COMPILE_WITH_GXM_

#endif
