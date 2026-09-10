#include "ge_gxm_memory.hpp"

#ifdef _IRR_COMPILE_WITH_GXM_

#include "../source/Irrlicht/os.h"

#include <algorithm>
#include <cstdio>
#include <memory>

namespace GE
{
namespace
{
// ----------------------------------------------------------------------------
inline size_t alignUp(size_t value, size_t alignment)
{
    if (alignment <= 1)
        return value;
    return (value + alignment - 1) & ~(alignment - 1);
}   // alignUp
}   // namespace

// ============================================================================
GEGXMHeap::GEGXMHeap(const std::string& name,
                     SceKernelMemBlockType block_type,
                     SceGxmMemoryAttribFlags attrib, size_t granularity,
                     size_t default_chunk_size)
{
    m_name = name;
    m_block_type = block_type;
    m_attrib = attrib;
    m_granularity = granularity;
    m_default_chunk_size = default_chunk_size;
    m_used = 0;
    m_reserved = 0;
}   // GEGXMHeap

// ----------------------------------------------------------------------------
GEGXMHeap::~GEGXMHeap()
{
    destroy();
}   // ~GEGXMHeap

// ----------------------------------------------------------------------------
void GEGXMHeap::destroy()
{
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    for (Chunk& c : m_chunks)
    {
        if (c.m_base != NULL)
            sceGxmUnmapMemory(c.m_base);
        if (c.m_uid >= 0)
            sceKernelFreeMemBlock(c.m_uid);
    }
    m_chunks.clear();
    m_allocations.clear();
    m_used = 0;
    m_reserved = 0;
}   // destroy

// ----------------------------------------------------------------------------
/** Reserves another kernel memory block and maps it for the GPU.
 *  \return the index of the new chunk, or -1 if the kernel refused. */
int GEGXMHeap::addChunk(size_t size, bool dedicated)
{
    size = alignUp(size, m_granularity);
    if (size == 0)
        return -1;

    // The block name shows up in Sony's memory tooling and in crash dumps, so
    // it is worth making it say which heap ran away.
    char block_name[32];
    snprintf(block_name, sizeof(block_name), "ge_gxm_%s_%u", m_name.c_str(),
        (unsigned)m_chunks.size());

    SceUID uid = sceKernelAllocMemBlock(block_name, m_block_type, size, NULL);
    if (uid < 0)
        return -1;

    void* base = NULL;
    if (sceKernelGetMemBlockBase(uid, &base) < 0 || base == NULL)
    {
        sceKernelFreeMemBlock(uid);
        return -1;
    }

    if (sceGxmMapMemory(base, size, m_attrib) < 0)
    {
        sceKernelFreeMemBlock(uid);
        return -1;
    }

    Chunk c;
    c.m_uid = uid;
    c.m_base = (uint8_t*)base;
    c.m_size = size;
    c.m_dedicated = dedicated;
    c.m_blocks[0] = { size, true };
    c.m_free_blocks.insert({ size, 0 });
    m_chunks.push_back(std::move(c));
    m_reserved += size;

    return (int)m_chunks.size() - 1;
}   // addChunk

// ----------------------------------------------------------------------------
void GEGXMHeap::removeFreeBlock(Chunk& c, size_t offset, size_t size)
{
    auto range = c.m_free_blocks.equal_range(size);
    for (auto it = range.first; it != range.second; it++)
    {
        if (it->second == offset)
        {
            c.m_free_blocks.erase(it);
            return;
        }
    }
}   // removeFreeBlock

// ----------------------------------------------------------------------------
void GEGXMHeap::insertFreeBlock(Chunk& c, size_t offset, size_t size)
{
    c.m_free_blocks.insert({ size, offset });
}   // insertFreeBlock

// ----------------------------------------------------------------------------
void* GEGXMHeap::allocate(size_t size, size_t alignment)
{
    if (size == 0)
        return NULL;
    if (alignment == 0)
        alignment = 1;

    std::lock_guard<std::recursive_mutex> lock(m_mutex);

    // A block big enough to matter gets its own kernel allocation. Cutting a
    // 32 MB atlas out of a shared chunk would strand the remainder.
    if (size + alignment > m_default_chunk_size / 2)
    {
        int idx = addChunk(size + alignment, true);
        if (idx < 0)
            return NULL;
        Chunk& c = m_chunks[idx];
        size_t base = (uintptr_t)c.m_base;
        size_t offset = alignUp(base, alignment) - base;
        removeFreeBlock(c, 0, c.m_size);
        c.m_blocks.clear();
        if (offset > 0)
            c.m_blocks[0] = { offset, true };
        c.m_blocks[offset] = { c.m_size - offset, false };
        if (offset > 0)
            insertFreeBlock(c, 0, offset);
        m_used += c.m_size - offset;
        void* ptr = c.m_base + offset;
        m_allocations[(uintptr_t)ptr] = { (unsigned)idx, offset };
        return ptr;
    }

    for (int attempt = 0; attempt < 2; attempt++)
    {
        for (unsigned i = 0; i < m_chunks.size(); i++)
        {
            Chunk& c = m_chunks[i];
            if (c.m_dedicated)
                continue;
            size_t chunk_base = (uintptr_t)c.m_base;
            // Best fit: the smallest free block that can still hold the
            // request once the alignment padding inside it is accounted for.
            for (auto it = c.m_free_blocks.lower_bound(size);
                it != c.m_free_blocks.end(); it++)
            {
                size_t offset = it->second;
                size_t block_size = it->first;
                size_t aligned = alignUp(chunk_base + offset, alignment) -
                    chunk_base;
                size_t pad = aligned - offset;
                if (pad + size > block_size)
                    continue;

                c.m_free_blocks.erase(it);
                auto block_it = c.m_blocks.find(offset);
                block_it->second.m_free = false;

                if (pad > 0)
                {
                    // Split the padding off as a free block in front.
                    block_it->second.m_size = pad;
                    block_it->second.m_free = true;
                    insertFreeBlock(c, offset, pad);
                    c.m_blocks[aligned] = { block_size - pad, false };
                    block_it = c.m_blocks.find(aligned);
                }

                size_t remaining = block_it->second.m_size - size;
                if (remaining > 0)
                {
                    block_it->second.m_size = size;
                    size_t tail = aligned + size;
                    c.m_blocks[tail] = { remaining, true };
                    insertFreeBlock(c, tail, remaining);
                }

                m_used += size;
                void* ptr = c.m_base + aligned;
                m_allocations[(uintptr_t)ptr] = { i, aligned };
                return ptr;
            }
        }
        if (attempt == 0 && addChunk(m_default_chunk_size, false) < 0)
            return NULL;
    }
    return NULL;
}   // allocate

// ----------------------------------------------------------------------------
bool GEGXMHeap::owns(void* ptr) const
{
    if (ptr == NULL)
        return false;
    GEGXMHeap* self = const_cast<GEGXMHeap*>(this);
    std::lock_guard<std::recursive_mutex> lock(self->m_mutex);
    return m_allocations.find((uintptr_t)ptr) != m_allocations.end();
}   // owns

// ----------------------------------------------------------------------------
bool GEGXMHeap::free(void* ptr)
{
    if (ptr == NULL)
        return true;

    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    auto alloc_it = m_allocations.find((uintptr_t)ptr);
    if (alloc_it == m_allocations.end())
        return false;

    unsigned chunk_idx = alloc_it->second.m_chunk;
    size_t offset = alloc_it->second.m_offset;
    m_allocations.erase(alloc_it);

    Chunk& c = m_chunks[chunk_idx];
    auto it = c.m_blocks.find(offset);
    if (it == c.m_blocks.end())
        return false;

    m_used -= it->second.m_size;
    it->second.m_free = true;
    insertFreeBlock(c, offset, it->second.m_size);

    // Coalesce forwards then backwards so the heap does not slowly turn into
    // a list of unusable slivers.
    auto next = std::next(it);
    if (next != c.m_blocks.end() && next->second.m_free)
    {
        removeFreeBlock(c, offset, it->second.m_size);
        removeFreeBlock(c, next->first, next->second.m_size);
        it->second.m_size += next->second.m_size;
        c.m_blocks.erase(next);
        insertFreeBlock(c, offset, it->second.m_size);
    }
    if (it != c.m_blocks.begin())
    {
        auto prev = std::prev(it);
        if (prev->second.m_free)
        {
            removeFreeBlock(c, prev->first, prev->second.m_size);
            removeFreeBlock(c, offset, it->second.m_size);
            prev->second.m_size += it->second.m_size;
            c.m_blocks.erase(it);
            insertFreeBlock(c, prev->first, prev->second.m_size);
            it = prev;
        }
    }

    // Hand a dedicated chunk, or a shared chunk that has gone completely
    // empty, straight back to the kernel. Keeping the first shared chunk
    // around avoids thrashing on the common allocate/free cycle.
    bool empty = c.m_blocks.size() == 1 && c.m_blocks.begin()->second.m_free;
    if (empty && (c.m_dedicated || chunk_idx > 0))
    {
        sceGxmUnmapMemory(c.m_base);
        sceKernelFreeMemBlock(c.m_uid);
        m_reserved -= c.m_size;
        c.m_uid = -1;
        c.m_base = NULL;
        c.m_size = 0;
        c.m_blocks.clear();
        c.m_free_blocks.clear();
        // The chunk index is baked into m_allocations, so the slot has to
        // stay; mark it dedicated so allocate() skips it and reuse it on the
        // next addChunk() only if it happens to be the tail.
        c.m_dedicated = true;
        while (!m_chunks.empty() && m_chunks.back().m_base == NULL)
            m_chunks.pop_back();
    }
    return true;
}   // free

// ============================================================================
namespace GEGXMMemory
{
namespace
{
std::unique_ptr<GEGXMHeap> g_heaps[GGMP_COUNT];

struct UsseAllocation
{
    SceUID m_uid;
    unsigned m_offset;
};
std::unordered_map<uintptr_t, UsseAllocation> g_vertex_usse;
std::unordered_map<uintptr_t, UsseAllocation> g_fragment_usse;
std::mutex g_usse_mutex;

// ----------------------------------------------------------------------------
void* allocateUsse(size_t size, unsigned* usse_offset, bool fragment)
{
    // USSE memory has to be a whole number of 4 KB pages of uncached LPDDR,
    // and gets its own mapping call rather than sceGxmMapMemory().
    size = alignUp(size, 4096);
    SceUID uid = sceKernelAllocMemBlock(fragment ? "ge_gxm_frag_usse" :
        "ge_gxm_vert_usse", SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE, size,
        NULL);
    if (uid < 0)
        return NULL;

    void* base = NULL;
    if (sceKernelGetMemBlockBase(uid, &base) < 0 || base == NULL)
    {
        sceKernelFreeMemBlock(uid);
        return NULL;
    }

    int err = fragment ?
        sceGxmMapFragmentUsseMemory(base, size, usse_offset) :
        sceGxmMapVertexUsseMemory(base, size, usse_offset);
    if (err < 0)
    {
        sceKernelFreeMemBlock(uid);
        return NULL;
    }

    std::lock_guard<std::mutex> lock(g_usse_mutex);
    auto& map = fragment ? g_fragment_usse : g_vertex_usse;
    map[(uintptr_t)base] = { uid, *usse_offset };
    return base;
}   // allocateUsse

// ----------------------------------------------------------------------------
void freeUsse(void* ptr, bool fragment)
{
    if (ptr == NULL)
        return;
    std::lock_guard<std::mutex> lock(g_usse_mutex);
    auto& map = fragment ? g_fragment_usse : g_vertex_usse;
    auto it = map.find((uintptr_t)ptr);
    if (it == map.end())
        return;
    if (fragment)
        sceGxmUnmapFragmentUsseMemory(ptr);
    else
        sceGxmUnmapVertexUsseMemory(ptr);
    sceKernelFreeMemBlock(it->second.m_uid);
    map.erase(it);
}   // freeUsse
}   // namespace

// ----------------------------------------------------------------------------
void init()
{
    if (g_heaps[GGMP_LPDDR_UNCACHED])
        return;
    // 4 MB LPDDR chunks: big enough that the per-frame churn never adds a
    // chunk after warm up, small enough that an idle menu does not hold on to
    // memory the rest of the game wants.
    g_heaps[GGMP_LPDDR_UNCACHED].reset(new GEGXMHeap("lpddr",
        SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE, SCE_GXM_MEMORY_ATTRIB_RW,
        4 * 1024, 4 * 1024 * 1024));
    g_heaps[GGMP_LPDDR_CACHED].reset(new GEGXMHeap("lpddr_cached",
        SCE_KERNEL_MEMBLOCK_TYPE_USER_RW, SCE_GXM_MEMORY_ATTRIB_RW,
        4 * 1024, 1 * 1024 * 1024));
    // CDRAM blocks are 256 KB granular, so the chunk is correspondingly
    // larger: textures are the bulk of what lands here.
    g_heaps[GGMP_CDRAM].reset(new GEGXMHeap("cdram",
        SCE_KERNEL_MEMBLOCK_TYPE_USER_CDRAM_RW, SCE_GXM_MEMORY_ATTRIB_RW,
        256 * 1024, 8 * 1024 * 1024));
}   // init

// ----------------------------------------------------------------------------
void destroy()
{
    {
        std::lock_guard<std::mutex> lock(g_usse_mutex);
        for (auto& p : g_vertex_usse)
        {
            sceGxmUnmapVertexUsseMemory((void*)p.first);
            sceKernelFreeMemBlock(p.second.m_uid);
        }
        g_vertex_usse.clear();
        for (auto& p : g_fragment_usse)
        {
            sceGxmUnmapFragmentUsseMemory((void*)p.first);
            sceKernelFreeMemBlock(p.second.m_uid);
        }
        g_fragment_usse.clear();
    }
    for (unsigned i = 0; i < GGMP_COUNT; i++)
        g_heaps[i].reset();
}   // destroy

// ----------------------------------------------------------------------------
bool isInitialised()
{
    return (bool)g_heaps[GGMP_LPDDR_UNCACHED];
}   // isInitialised

// ----------------------------------------------------------------------------
void* allocate(GEGXMMemoryPool pool, size_t size, size_t alignment)
{
    if (pool >= GGMP_COUNT || !g_heaps[pool])
        return NULL;
    void* ptr = g_heaps[pool]->allocate(size, alignment);
    // CDRAM is a hard 128 MB and textures will exhaust it long before LPDDR
    // runs out, so falling back keeps the game running with a slower texture
    // rather than a missing one.
    if (ptr == NULL && pool == GGMP_CDRAM)
    {
        ptr = g_heaps[GGMP_LPDDR_UNCACHED]->allocate(size, alignment);
        if (ptr != NULL)
        {
            static bool warned = false;
            if (!warned)
            {
                warned = true;
                irr::os::Printer::log("GXM: CDRAM exhausted, spilling GPU "
                    "allocations to LPDDR", irr::ELL_WARNING);
            }
        }
    }
    return ptr;
}   // allocate

// ----------------------------------------------------------------------------
void free(void* ptr)
{
    if (ptr == NULL)
        return;
    for (unsigned i = 0; i < GGMP_COUNT; i++)
    {
        if (g_heaps[i] && g_heaps[i]->free(ptr))
            return;
    }
}   // free

// ----------------------------------------------------------------------------
void* allocateVertexUsse(size_t size, unsigned* usse_offset)
{
    return allocateUsse(size, usse_offset, false);
}   // allocateVertexUsse

// ----------------------------------------------------------------------------
void freeVertexUsse(void* ptr)
{
    freeUsse(ptr, false);
}   // freeVertexUsse

// ----------------------------------------------------------------------------
void* allocateFragmentUsse(size_t size, unsigned* usse_offset)
{
    return allocateUsse(size, usse_offset, true);
}   // allocateFragmentUsse

// ----------------------------------------------------------------------------
void freeFragmentUsse(void* ptr)
{
    freeUsse(ptr, true);
}   // freeFragmentUsse

// ----------------------------------------------------------------------------
void getStats(GEGXMMemoryPool pool, size_t* used, size_t* reserved)
{
    if (pool >= GGMP_COUNT || !g_heaps[pool])
    {
        if (used)
            *used = 0;
        if (reserved)
            *reserved = 0;
        return;
    }
    if (used)
        *used = g_heaps[pool]->getUsed();
    if (reserved)
        *reserved = g_heaps[pool]->getReserved();
}   // getStats

// ----------------------------------------------------------------------------
void logStats()
{
    for (unsigned i = 0; i < GGMP_COUNT; i++)
    {
        if (!g_heaps[i])
            continue;
        char buf[128];
        snprintf(buf, sizeof(buf),
            "GXM memory %s: %.2f MB used of %.2f MB reserved",
            g_heaps[i]->getName().c_str(),
            (double)g_heaps[i]->getUsed() / (1024.0 * 1024.0),
            (double)g_heaps[i]->getReserved() / (1024.0 * 1024.0));
        irr::os::Printer::log(buf, irr::ELL_INFORMATION);
    }
}   // logStats

}   // namespace GEGXMMemory

}   // namespace GE

#endif   // _IRR_COMPILE_WITH_GXM_
