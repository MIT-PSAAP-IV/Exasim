#ifndef EXASIM_HDGLDG_PREPROCESSING_HPP
#define EXASIM_HDGLDG_PREPROCESSING_HPP

// One-time CPU topology planning. Native connectivity supplies the face graph,
// including periodic neighbours; this helper does not match geometry again.
#include <algorithm>
#include <array>
#include <charconv>
#include <functional>
#include <istream>
#include <limits>
#include <ostream>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>
#if defined(HAVE_PARMETIS) || defined(HAVE_METIS)
#include <metis.h>
#endif

namespace exasim { namespace hdgldg {

struct Topology {
    int nfe = 0;
    std::vector<int> originalIDs;
    // All face arrays use face + nfe*element. Negative neighbour means boundary.
    std::vector<int> neighbours;
    std::vector<int> oppositeFaces;
    // Physical boundary IDs only; periodic faces must not be physical seeds.
    std::vector<int> physicalBoundaryIDs;
};

struct Graph {
    std::vector<int> offsets, neighbours;
};

struct Region {
    std::vector<int> elements, originalIDs, globalToRegional;
    std::vector<int> worldRanks, parts;
    Graph graph;
};

struct InterfaceFace {
    int hOriginalID, hFace, lOriginalID, lFace;
    int hElement, lElement; // Input element indices, not persistent IDs.
    int hWorldRank, lWorldRank;
};

struct Plan {
    std::vector<int> labels, destinationWorld;
    std::array<Region, 2> regions; // 0: native LDG, 1: native HDG.
    std::vector<InterfaceFace> interfaces;
};

using Partitioner = std::function<std::vector<int>(const Graph&, int)>;

// Optional sidecar only; native PDE scheme and binary formats remain intact.
struct Spec {
    int version = 1, worldSize = 0;
    std::vector<int> hWorldRanks, lWorldRanks;
    std::string selection;
    std::vector<int> labels, seedBoundaries;
    int layers = 0;
};

inline std::string trimManifestText(const std::string& text)
{
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    return text.substr(first, text.find_last_not_of(" \t\r\n")-first+1);
}

inline int readManifestInteger(const std::string& token)
{
    const auto text = trimManifestText(token);
    int value = 0;
    const auto result = std::from_chars(text.data(), text.data()+text.size(), value);
    if (text.empty() || result.ec != std::errc() || result.ptr != text.data()+text.size())
        throw std::invalid_argument("rank-separated manifest requires integer values");
    return value;
}

inline std::vector<int> readManifestIntegers(std::string text)
{
    text = trimManifestText(text);
    if (!text.empty() && (text.front() == '[' || text.back() == ']')) {
        if (text.front() != '[' || text.back() != ']')
            throw std::invalid_argument("rank-separated manifest list has unmatched brackets");
        text = trimManifestText(text.substr(1, text.size()-2));
    }
    if (text.empty()) throw std::invalid_argument("rank-separated manifest list is empty");
    std::vector<int> values;
    std::size_t start = 0;
    while (true) {
        const auto end = text.find(',', start);
        values.push_back(readManifestInteger(text.substr(start, end == std::string::npos ? end : end-start)));
        if (end == std::string::npos) break;
        start = end+1;
    }
    return values;
}

inline void checkSpec(const Spec& spec)
{
    if (spec.version != 1 || spec.worldSize < 2 || spec.hWorldRanks.empty() || spec.lWorldRanks.empty())
        throw std::invalid_argument("rank-separated manifest version or rank groups are invalid");
    std::unordered_set<int> ranks;
    for (const auto* group : {&spec.hWorldRanks, &spec.lWorldRanks})
        for (int rank : *group)
            if (rank < 0 || rank >= spec.worldSize || !ranks.insert(rank).second)
                throw std::invalid_argument("rank-separated manifest rank groups overlap or contain invalid ranks");
    if (ranks.size() != static_cast<std::size_t>(spec.worldSize))
        throw std::invalid_argument("rank-separated manifest rank groups do not cover world_size");
    if (spec.selection == "labels") {
        if (spec.labels.empty() || !spec.seedBoundaries.empty() || spec.layers != 0)
            throw std::invalid_argument("rank-separated labels selection has incompatible fields");
        for (int label : spec.labels) if (label != 0 && label != 1)
            throw std::invalid_argument("rank-separated manifest labels must be zero or one");
    } else if (spec.selection == "boundary_layers") {
        if (!spec.labels.empty() || spec.seedBoundaries.empty() || spec.layers < 1)
            throw std::invalid_argument("rank-separated boundary selection has incompatible fields");
        for (int seed : spec.seedBoundaries) if (seed < 0)
            throw std::invalid_argument("rank-separated manifest seed boundary is negative");
    } else throw std::invalid_argument("rank-separated manifest selection is unknown");
}

inline Spec readSpec(std::istream& input)
{
    Spec spec;
    std::unordered_set<std::string> keys;
    std::string line;
    while (std::getline(input, line)) {
        const auto comment = line.find('#');
        if (comment != std::string::npos) line.erase(comment);
        line = trimManifestText(line);
        if (line.empty()) continue;
        const auto equals = line.find('=');
        if (equals == std::string::npos || line.find('=', equals+1) != std::string::npos)
            throw std::invalid_argument("rank-separated manifest line must have one equals sign");
        const auto key = trimManifestText(line.substr(0, equals));
        const auto value = trimManifestText(line.substr(equals+1));
        if (!keys.insert(key).second)
            throw std::invalid_argument("rank-separated manifest duplicate key: "+key);
        if (key == "version") spec.version = readManifestInteger(value);
        else if (key == "world_size") spec.worldSize = readManifestInteger(value);
        else if (key == "h_ranks") spec.hWorldRanks = readManifestIntegers(value);
        else if (key == "l_ranks") spec.lWorldRanks = readManifestIntegers(value);
        else if (key == "selection") spec.selection = value;
        else if (key == "labels") spec.labels = readManifestIntegers(value);
        else if (key == "seed_boundaries") spec.seedBoundaries = readManifestIntegers(value);
        else if (key == "layers") spec.layers = readManifestInteger(value);
        else throw std::invalid_argument("rank-separated manifest unknown key: "+key);
    }
    if (input.bad()) throw std::runtime_error("rank-separated manifest read failed");
    for (const char* required : {"version", "world_size", "h_ranks", "l_ranks", "selection"})
        if (!keys.count(required)) throw std::invalid_argument(std::string("rank-separated manifest missing key: ")+required);
    if (spec.selection == "labels") {
        if (!keys.count("labels") || keys.count("seed_boundaries") || keys.count("layers"))
            throw std::invalid_argument("rank-separated manifest requires labels only for labels selection");
    } else if (spec.selection == "boundary_layers") {
        if (!keys.count("seed_boundaries") || !keys.count("layers") || keys.count("labels"))
            throw std::invalid_argument("rank-separated manifest requires seed_boundaries and layers only");
    }
    checkSpec(spec);
    return spec;
}

inline void writeSpec(std::ostream& output, const Spec& spec)
{
    checkSpec(spec);
    auto list = [&](const char* key, const std::vector<int>& values) {
        output << key << " = [";
        for (std::size_t i = 0; i < values.size(); ++i) output << (i ? ", " : "") << values[i];
        output << "]\n";
    };
    output << "version = 1\nworld_size = " << spec.worldSize << '\n';
    list("h_ranks", spec.hWorldRanks); list("l_ranks", spec.lWorldRanks);
    output << "selection = " << spec.selection << '\n';
    if (spec.selection == "labels") list("labels", spec.labels);
    else { list("seed_boundaries", spec.seedBoundaries); output << "layers = " << spec.layers << '\n'; }
    if (!output) throw std::runtime_error("rank-separated manifest write failed");
}

inline void checkTopology(const Topology& topology)
{
    const auto ne = topology.originalIDs.size();
    if (topology.nfe <= 0 || ne == 0 ||
        ne > static_cast<std::size_t>(std::numeric_limits<int>::max()/topology.nfe))
        throw std::invalid_argument("rank-separated topology has invalid dimensions");
    const auto nf = ne*topology.nfe;
    if (topology.neighbours.size() != nf || topology.oppositeFaces.size() != nf ||
        topology.physicalBoundaryIDs.size() != nf)
        throw std::invalid_argument("rank-separated face array length mismatch");
    std::unordered_set<int> ids;
    for (int id : topology.originalIDs)
        if (id < 0 || !ids.insert(id).second)
            throw std::invalid_argument("rank-separated original IDs must be unique and nonnegative");
    for (int e = 0; e < static_cast<int>(ne); ++e)
        for (int f = 0; f < topology.nfe; ++f) {
            const int i = f+topology.nfe*e, j = topology.neighbours[i];
            if (j < 0) continue;
            const int g = topology.oppositeFaces[i];
            if (j >= static_cast<int>(ne) || g < 0 || g >= topology.nfe ||
                topology.neighbours[g+topology.nfe*j] != e ||
                topology.oppositeFaces[g+topology.nfe*j] != f)
                throw std::invalid_argument("rank-separated face graph is not reciprocal");
            if (topology.physicalBoundaryIDs[i] >= 0)
                throw std::invalid_argument("rank-separated neighbour face is marked as physical boundary");
        }
}

inline std::vector<int> selectBoundaryLayers(const Topology& topology,
                                            const std::vector<int>& seedBoundaries,
                                            int layers)
{
    checkTopology(topology);
    if (layers < 1 || seedBoundaries.empty())
        throw std::invalid_argument("rank-separated boundary selection requires seeds and positive layers");
    const int ne = static_cast<int>(topology.originalIDs.size());
    std::unordered_set<int> seeds(seedBoundaries.begin(), seedBoundaries.end());
    if (*std::min_element(seedBoundaries.begin(), seedBoundaries.end()) < 0)
        throw std::invalid_argument("rank-separated physical seed boundary ID is negative");
    std::vector<int> distance(ne, -1), labels(ne, 0);
    std::queue<int> queue;
    for (int e = 0; e < ne; ++e)
        for (int f = 0; f < topology.nfe; ++f)
            if (seeds.count(topology.physicalBoundaryIDs[f+topology.nfe*e])) {
                if (distance[e] < 0) { distance[e] = 0; queue.push(e); }
            }
    while (!queue.empty()) {
        const int e = queue.front(); queue.pop(); labels[e] = 1;
        if (distance[e]+1 >= layers) continue;
        for (int f = 0; f < topology.nfe; ++f) {
            const int j = topology.neighbours[f+topology.nfe*e];
            if (j >= 0 && distance[j] < 0) {
                distance[j] = distance[e]+1; queue.push(j);
            }
        }
    }
    return labels;
}

#if defined(HAVE_PARMETIS) || defined(HAVE_METIS)
inline std::vector<int> metisPartitionGraph(const Graph& graph, int parts)
{
    if (graph.offsets.size() < 2 || graph.offsets.front() != 0 ||
        graph.offsets.size()-1 > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        graph.neighbours.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        graph.offsets.back() != static_cast<int>(graph.neighbours.size()) ||
        !std::is_sorted(graph.offsets.begin(), graph.offsets.end()))
        throw std::invalid_argument("rank-separated METIS graph offsets are invalid");
    idx_t ne = static_cast<idx_t>(graph.offsets.size()-1), nparts = parts;
    if (parts < 1 || ne < nparts)
        throw std::invalid_argument("rank-separated METIS partition has too few elements");
    if (parts == 1) return std::vector<int>(static_cast<std::size_t>(ne), 0);
    std::vector<idx_t> xadj(graph.offsets.begin(), graph.offsets.end());
    std::vector<idx_t> adjncy(graph.neighbours.begin(), graph.neighbours.end()), part(ne);
    for (int neighbour : graph.neighbours)
        if (neighbour < 0 || neighbour >= ne)
            throw std::invalid_argument("rank-separated METIS graph neighbour is invalid");
    idx_t options[METIS_NOPTIONS], ncon = 1, edgecut = 0;
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_NUMBERING] = 0;
    options[METIS_OPTION_SEED] = 0;
    const int status = METIS_PartGraphKway(&ne, &ncon, xadj.data(), adjncy.data(),
        nullptr, nullptr, nullptr, &nparts, nullptr, nullptr, options, &edgecut, part.data());
    if (status != METIS_OK)
        throw std::runtime_error("rank-separated METIS graph partition failed");
    return std::vector<int>(part.begin(), part.end());
}
#endif

inline Plan buildPlan(const Topology& topology, const std::vector<int>& labels,
                      const std::vector<int>& hRanks, const std::vector<int>& lRanks,
                      int worldSize, const Partitioner& partitioner = Partitioner())
{
    checkTopology(topology);
    const int ne = static_cast<int>(topology.originalIDs.size());
    if (static_cast<int>(labels.size()) != ne || worldSize < 2 || hRanks.empty() || lRanks.empty())
        throw std::invalid_argument("rank-separated labels or rank groups are invalid");
    std::vector<int> rankSeen(worldSize, 0);
    for (const auto* ranks : {&hRanks, &lRanks})
        for (int rank : *ranks)
            if (rank < 0 || rank >= worldSize || rankSeen[rank]++)
                throw std::invalid_argument("rank-separated rank groups overlap or contain invalid ranks");
    if (std::find(rankSeen.begin(), rankSeen.end(), 0) != rankSeen.end())
        throw std::invalid_argument("rank-separated rank groups must cover coupled communicator");
    Plan plan; plan.labels = labels; plan.destinationWorld.resize(ne);
    plan.regions[0].worldRanks = lRanks; plan.regions[1].worldRanks = hRanks;
    for (auto& region : plan.regions) {
        // Match MPI_Comm_split(..., key=worldRank): part indices are native
        // local ranks, even when the manifest lists its rank sets unsorted.
        std::sort(region.worldRanks.begin(), region.worldRanks.end());
        region.globalToRegional.assign(ne, -1);
    }
    for (int e = 0; e < ne; ++e) {
        if (labels[e] != 0 && labels[e] != 1)
            throw std::invalid_argument("rank-separated label must be native scheme zero or one");
        auto& region = plan.regions[labels[e]];
        region.globalToRegional[e] = static_cast<int>(region.elements.size());
        region.elements.push_back(e); region.originalIDs.push_back(topology.originalIDs[e]);
    }
    for (auto& region : plan.regions) {
        const int count = static_cast<int>(region.elements.size());
        if (count < static_cast<int>(region.worldRanks.size()))
            throw std::invalid_argument("rank-separated region has fewer elements than ranks");
        region.graph.offsets.push_back(0);
        for (int e : region.elements) {
            std::vector<int> adjacent;
            for (int f = 0; f < topology.nfe; ++f) {
                const int j = topology.neighbours[f+topology.nfe*e];
                if (j >= 0 && j != e && labels[j] == labels[e])
                    adjacent.push_back(region.globalToRegional[j]);
            }
            std::sort(adjacent.begin(), adjacent.end());
            adjacent.erase(std::unique(adjacent.begin(), adjacent.end()), adjacent.end());
            region.graph.neighbours.insert(region.graph.neighbours.end(), adjacent.begin(), adjacent.end());
            region.graph.offsets.push_back(static_cast<int>(region.graph.neighbours.size()));
        }
        if (region.worldRanks.size() == 1) region.parts.assign(count, 0);
        else {
            if (!partitioner) throw std::invalid_argument("rank-separated multi-rank region requires graph partitioner");
            region.parts = partitioner(region.graph, static_cast<int>(region.worldRanks.size()));
        }
        if (static_cast<int>(region.parts.size()) != count)
            throw std::invalid_argument("rank-separated partitioner returned wrong number of parts");
        std::vector<int> owned(region.worldRanks.size(), 0);
        for (int i = 0; i < count; ++i) {
            const int part = region.parts[i];
            if (part < 0 || part >= static_cast<int>(owned.size()))
                throw std::invalid_argument("rank-separated partitioner returned invalid part");
            ++owned[part]; plan.destinationWorld[region.elements[i]] = region.worldRanks[part];
        }
        if (std::find(owned.begin(), owned.end(), 0) != owned.end())
            throw std::runtime_error("rank-separated partitioner produced empty rank");
    }
    for (int h = 0; h < ne; ++h) if (labels[h] == 1)
        for (int f = 0; f < topology.nfe; ++f) {
            const int l = topology.neighbours[f+topology.nfe*h];
            if (l >= 0 && labels[l] == 0)
                plan.interfaces.push_back({topology.originalIDs[h], f, topology.originalIDs[l],
                    topology.oppositeFaces[f+topology.nfe*h], h, l,
                    plan.destinationWorld[h], plan.destinationWorld[l]});
        }
    std::sort(plan.interfaces.begin(), plan.interfaces.end(), [](const InterfaceFace& a, const InterfaceFace& b) {
        return a.hOriginalID < b.hOriginalID || (a.hOriginalID == b.hOriginalID && a.hFace < b.hFace);
    });
    return plan;
}

inline Plan selectAndBuildPlan(const Topology& topology, const Spec& spec,
                              Partitioner partitioner = Partitioner())
{
    checkSpec(spec);
    const auto labels = spec.selection == "labels" ? spec.labels :
        selectBoundaryLayers(topology, spec.seedBoundaries, spec.layers);
#if defined(HAVE_PARMETIS) || defined(HAVE_METIS)
    if (!partitioner) partitioner = metisPartitionGraph;
#endif
    return buildPlan(topology, labels, spec.hWorldRanks, spec.lWorldRanks,
                     spec.worldSize, partitioner);
}

// Element arrays are [rows,ne], column major. Face-indexed trace data requires
// a separate face/orientation map and must not be passed to this helper.
template <class T>
inline std::vector<T> extractElementColumns(const std::vector<T>& input, int rows,
                                           int ne, const std::vector<int>& elements)
{
    if (rows < 0 || ne < 0 || static_cast<std::size_t>(rows)*ne != input.size())
        throw std::invalid_argument("rank-separated aligned field shape mismatch");
    std::vector<T> output(static_cast<std::size_t>(rows)*elements.size());
    for (std::size_t i = 0; i < elements.size(); ++i) {
        const int e = elements[i];
        if (e < 0 || e >= ne) throw std::invalid_argument("rank-separated extraction element out of range");
        std::copy_n(input.begin()+static_cast<std::size_t>(rows)*e, rows,
                    output.begin()+static_cast<std::size_t>(rows)*i);
    }
    return output;
}

}} // namespace exasim::hdgldg
#endif
