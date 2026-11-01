#pragma once

#include "composite_config.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace exasim::composite {
struct BlockPlan {
    bool changed = false;
    int blocks = 0;
    int maximum = 0;
    std::vector<int> entries;
};

// Element blocks use one-based inclusive intervals and an unchanged region tag.
// Split within each existing interval; never combine across a partition/tag
// boundary or change the element numbering used by connectivity and MPI halos.
inline BlockPlan limitBlockIntervals(const int* entries, int entryCount,
                                    int elements, int blocks, int maximum,
                                    const Config& config) {
    BlockPlan plan;
    if (!config.enabled) return plan;
    if (config.batch <= 0 || elements < 0 || blocks < 0 || maximum < 0 ||
        entryCount < 0 || uint64_t(blocks)*3 != uint64_t(entryCount) ||
        (entryCount && !entries))
        throw std::invalid_argument("compositehdg: malformed element block metadata");
    if (!elements) {
        if (blocks || entryCount || maximum)
            throw std::invalid_argument("compositehdg: nonempty blocks for an empty mesh");
        return plan;
    }
    if (!blocks)
        throw std::invalid_argument("compositehdg: element blocks do not cover the mesh");
    uint64_t next = 1, newBlocks = 0;
    int actualMaximum = 0;
    for (int block = 0; block < blocks; ++block) {
        const int first = entries[3*block], last = entries[3*block+1];
        if (first < 1 || uint64_t(first) != next || last < first || last > elements)
            throw std::invalid_argument("compositehdg: element blocks must cover the mesh in order without gaps or overlaps");
        const uint64_t length = uint64_t(last)-uint64_t(first)+1;
        actualMaximum = std::max(actualMaximum, int(length));
        newBlocks += (length+uint64_t(config.batch)-1)/uint64_t(config.batch);
        next = uint64_t(last)+1;
    }
    if (next != uint64_t(elements)+1 || maximum < actualMaximum)
        throw std::invalid_argument("compositehdg: element block extent/maximum disagrees with mesh metadata");
    // A too-large stored maximum also causes oversized workspaces even if all
    // actual intervals are short. Normalize it only when it exceeds the cap.
    if (actualMaximum <= config.batch && maximum <= config.batch) return plan;
    const int newEntries = count(newBlocks*3, "composite element block table");
    plan.changed = true;
    plan.blocks = int(newBlocks);
    plan.entries.reserve(size_t(newEntries));
    for (int block = 0; block < blocks; ++block) {
        const int tag = entries[3*block+2];
        const uint64_t last = uint64_t(entries[3*block+1]);
        for (uint64_t first = uint64_t(entries[3*block]); first <= last;) {
            const uint64_t end = std::min(last, first+uint64_t(config.batch)-1);
            plan.entries.push_back(int(first));
            plan.entries.push_back(int(end));
            plan.entries.push_back(tag);
            plan.maximum = std::max(plan.maximum, int(end-first+1));
            first = end+1;
        }
    }
    return plan;
}

// Allocation/release callbacks keep ownership and optional memory accounting
// with the backend. An invalid plan leaves the original mesh entirely intact.
template<class Mesh, class Allocate, class Release>
void limitMeshBlocks(Mesh& mesh, const Config& config, Allocate allocate, Release release) {
    if (!config.enabled) return;
    using Index = std::remove_pointer_t<decltype(mesh.eblks)>;
    if constexpr (!std::is_same_v<Index, int>) {
        throw std::invalid_argument("compositehdg: block limiter requires 32-bit mesh indices");
    } else {
        if (!mesh.lsize || mesh.lsize[0] < 3 || !mesh.nsize || !mesh.ndims ||
            mesh.nsize[0] < 7 || mesh.nsize[2] != mesh.szeblks)
            throw std::invalid_argument("compositehdg: incomplete element block metadata");
        const auto plan = limitBlockIntervals(mesh.eblks, mesh.nsize[2], mesh.ndims[1],
                                             mesh.ndims[5], mesh.ndims[6], config);
        if (!plan.changed) return;
        int* replacement = allocate(int(plan.entries.size()));
        if (!replacement) throw std::bad_alloc();
        std::copy(plan.entries.begin(), plan.entries.end(), replacement);
        release(mesh.eblks);
        mesh.eblks = replacement;
        mesh.ndims[5] = plan.blocks;
        mesh.ndims[6] = plan.maximum;
        mesh.nsize[2] = int(plan.entries.size());
        mesh.szeblks = int(plan.entries.size());
    }
}
} // namespace exasim::composite
