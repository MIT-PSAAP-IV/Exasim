#ifndef EXASIM_HDGLDG_REGIONS_HPP
#define EXASIM_HDGLDG_REGIONS_HPP

// Include after the native preprocessing implementation. This is CPU-only
// setup orchestration, not a second mesh builder or a runtime discretization.
#include "hdgldgpreprocessing.hpp"
#include <fstream>
#include <map>
#include <numeric>

namespace exasim { namespace hdgldg {

struct PreparedRegion {
    Mesh mesh;
    PDE pde;
    std::vector<DMD> partitions;
    std::vector<int> worldRanks;
    // Per global interface record: ordered H vertices followed by L vertices.
    std::vector<std::vector<int>> interfaceVertexIDs;
    int scheme = 0, hlSelector = -1, hlBoundaryCode = -1;
};

// buildMesh has already identified periodic vertices. Matching sorted vertex
// keys therefore reuses exactly the native connectivity's periodic topology.
inline Topology topologyFromPreparedMesh(Mesh& mesh)
{
    if (mesh.elemGlobalID.empty()) {
        mesh.elemGlobalID.resize(mesh.ne);
        std::iota(mesh.elemGlobalID.begin(), mesh.elemGlobalID.end(), 0);
    }
    if (mesh.elemGlobalID.size() != static_cast<std::size_t>(mesh.ne) ||
        mesh.f.size() != static_cast<std::size_t>(mesh.nfe*mesh.ne) ||
        mesh.localfaces.size() != static_cast<std::size_t>(mesh.nvf*mesh.nfe))
        throw std::invalid_argument("rank-separated mesh must be built before topology selection");
    mesh.t2t.resize(mesh.nfe*mesh.ne);
    mesh.nf = mke2e_hash(mesh.t2t.data(), mesh.t.data(), mesh.localfaces.data(),
                        mesh.ne, mesh.nve, mesh.nvf, mesh.nfe);
    Topology topology;
    topology.nfe = mesh.nfe; topology.originalIDs = mesh.elemGlobalID;
    topology.neighbours = mesh.t2t;
    topology.oppositeFaces.assign(mesh.nfe*mesh.ne, -1);
    topology.physicalBoundaryIDs.assign(mesh.nfe*mesh.ne, -1);
    std::map<std::vector<int>, std::vector<int>> faces;
    for (int e = 0; e < mesh.ne; ++e) for (int f = 0; f < mesh.nfe; ++f) {
        const int index = f+mesh.nfe*e;
        std::vector<int> key(mesh.nvf);
        for (int v = 0; v < mesh.nvf; ++v)
            key[v] = mesh.t[mesh.localfaces[v+mesh.nvf*f]+mesh.nve*e];
        std::sort(key.begin(), key.end());
        faces[key].push_back(index);
        // Selection seeds are model boundary-condition codes, not zero-based
        // boundary-expression selectors. Repeated codes select their union.
        if (mesh.t2t[index] < 0 && mesh.f[index] >= 0)
            topology.physicalBoundaryIDs[index] = mesh.boundaryConditions.at(mesh.f[index]);
    }
    for (const auto& entry : faces) {
        const auto& pair = entry.second;
        if (pair.size() > 2)
            throw std::invalid_argument("rank-separated face key is nonmanifold or periodic topology is degenerate");
        if (pair.size() == 2) {
            topology.oppositeFaces[pair[0]] = pair[1]%mesh.nfe;
            topology.oppositeFaces[pair[1]] = pair[0]%mesh.nfe;
        }
    }
    checkTopology(topology);
    return topology;
}

inline std::vector<double> extractOptionalField(const std::vector<double>& field,
                                               int ne, const std::vector<int>& elements)
{
    if (field.empty()) return {};
    if (ne < 1 || field.size()%ne != 0)
        throw std::invalid_argument("rank-separated element field has incomplete columns");
    return extractElementColumns(field, static_cast<int>(field.size()/ne), ne, elements);
}

inline PreparedRegion extractPreparedRegion(const Mesh& complete, const Master& master,
                                            const PDE& input, const Plan& plan, int scheme)
{
    if (scheme != 0 && scheme != 1)
        throw std::invalid_argument("rank-separated region scheme must be native LDG or HDG");
    if (input.coupledinterface != 0 || !input.interactionSelectors.empty() || !input.interactions.empty())
        throw std::invalid_argument("rank-separated preparation cannot combine existing legacy or generalized coupling metadata");
    if (!complete.uhat.empty())
        throw std::invalid_argument("rank-separated supplied uhat requires an explicit original-face extraction map");
    const auto checkField = [&](const std::vector<double>& field, int components, const char* name) {
        if (!field.empty() && field.size() != static_cast<std::size_t>(master.npe)*components*complete.ne)
            throw std::invalid_argument(std::string("rank-separated invalid field extent: ")+name);
    };
    checkField(complete.xdg, complete.dim, "xdg");
    if (!complete.udg.empty() &&
        complete.udg.size() != static_cast<std::size_t>(master.npe)*input.ncu*complete.ne)
        checkField(complete.udg, input.nc, "udg");
    checkField(complete.vdg, input.ncv, "vdg");
    checkField(complete.wdg, input.ncw, "wdg");
    const Region& selection = plan.regions[scheme];
    if (selection.originalIDs.size() != selection.elements.size() ||
        selection.parts.size() != selection.elements.size() || selection.worldRanks.empty())
        throw std::invalid_argument("rank-separated regional plan is incomplete");
    PreparedRegion output;
    output.scheme = scheme; output.worldRanks = selection.worldRanks;
    output.mesh = complete; output.pde = input;
    for (const auto& face : plan.interfaces) {
        std::vector<int> vertices;
        for (const auto endpoint : {std::make_pair(face.hElement, face.hFace),
                                    std::make_pair(face.lElement, face.lFace)})
            for (int v = 0; v < complete.nvf; ++v)
                vertices.push_back(complete.t[complete.localfaces[v+complete.nvf*endpoint.second]
                                                +complete.nve*endpoint.first]);
        output.interfaceVertexIDs.push_back(std::move(vertices));
    }
    Mesh& mesh = output.mesh; PDE& pde = output.pde;
    mesh.ne = static_cast<int>(selection.elements.size());
    mesh.ne_global = mesh.ne; mesh.elemGlobalID = selection.originalIDs;
    mesh.elem2cpu = selection.parts;
    mesh.t = extractElementColumns(complete.t, complete.nve, complete.ne, selection.elements);
    mesh.tg = complete.tg.empty() ? std::vector<int>() :
        extractElementColumns(complete.tg, complete.nve, complete.ne, selection.elements);
    mesh.f = extractElementColumns(complete.f, complete.nfe, complete.ne, selection.elements);
    mesh.xdg = extractOptionalField(complete.xdg, complete.ne, selection.elements);
    mesh.udg = extractOptionalField(complete.udg, complete.ne, selection.elements);
    mesh.vdg = extractOptionalField(complete.vdg, complete.ne, selection.elements);
    mesh.wdg = extractOptionalField(complete.wdg, complete.ne, selection.elements);
    for (auto* dims : {&mesh.xdgdims,&mesh.udgdims,&mesh.vdgdims,&mesh.wdgdims})
        if (dims->size() == 3) (*dims)[2] = mesh.ne;
    mesh.uhat.clear(); mesh.uhatdims.clear();
    mesh.inte.clear(); mesh.intl.clear(); mesh.bf.clear(); mesh.f2t.clear();
    mesh.t2f.clear(); mesh.t2lf.clear(); mesh.epart_local.clear(); mesh.elmdist.clear();
    // Coordinates and vertex IDs are shared read-only setup data; do not
    // renumber periodic vertices or regenerate projected high-order nodes.
    output.hlSelector = static_cast<int>(mesh.boundaryConditions.size());
    output.hlBoundaryCode = 1;
    for (int code : mesh.boundaryConditions)
        output.hlBoundaryCode = std::max(output.hlBoundaryCode, code+1);
    mesh.boundaryConditions.push_back(output.hlBoundaryCode);
    mesh.nbcm = static_cast<int>(mesh.boundaryConditions.size());
    mesh.curvedBoundaries.push_back(0);
    for (const auto& face : plan.interfaces) {
        const int originalIndex = scheme ? face.hElement : face.lElement;
        const int localFace = scheme ? face.hFace : face.lFace;
        const int regional = selection.globalToRegional.at(originalIndex);
        if (regional < 0) throw std::invalid_argument("rank-separated interface endpoint absent from region");
        mesh.f.at(localFace+mesh.nfe*regional) = output.hlSelector;
        mesh.inte.push_back(regional); mesh.intl.push_back(localFace);
    }
    mesh.t2t.resize(mesh.nfe*mesh.ne);
    mesh.nf = mke2e_hash(mesh.t2t.data(), mesh.t.data(), mesh.localfaces.data(),
                        mesh.ne, mesh.nve, mesh.nvf, mesh.nfe);
    pde.hybrid = scheme; pde.discretization = scheme ? "hdg" : "ldg";
    pde.mpiprocs = static_cast<int>(selection.worldRanks.size());
    pde.coupledinterface = 0; pde.interactionSelectors.clear();
    // No generalized-coupling partition semantics are borrowed. HL metadata
    // lives in the explicit sidecar and native ghost depths stay unchanged.
    if (pde.mpiprocs == 1) output.partitions = {initializeDMD(pde, mesh)};
    else {
        output.partitions.resize(pde.mpiprocs);
        if (scheme) build_dmdhdg(output.partitions, mesh.t2t.data(), mesh.elem2cpu.data(),
            nullptr, 0, mesh.nfe, mesh.ne, pde);
        else build_dmdldg(output.partitions, mesh.t2t.data(), mesh.elem2cpu.data(),
            mesh.nfe, mesh.ne, pde);
    }
    return output;
}

// New sidecar only; native app/master/mesh/sol payload layouts are unchanged.
// Element records include ghosts. Only the owning endpoint contributes flux/D;
// ghost endpoints are read-only trace consumers for native L q reconstruction.
inline void writeRegionSidecar(const PreparedRegion& region, const Plan& plan,
                               int localRank, const std::string& filename)
{
    std::ofstream stream(filename);
    if (!stream) throw std::runtime_error("cannot write rank-separated region sidecar");
    const auto& selection = plan.regions[region.scheme];
    const auto& dmd = region.partitions.at(localRank);
    const int world = region.worldRanks.at(localRank);
    stream << "EXASIM_HDGLDG 1\n" << region.scheme << ' ' << world << ' '
           << localRank << ' ' << region.hlSelector << ' ' << region.hlBoundaryCode << '\n';
    stream << dmd.elempart.size() << '\n';
    std::vector<int> regionalToLocal(selection.elements.size(), -1);
    for (std::size_t i = 0; i < dmd.elempart.size(); ++i) {
        const int regional = dmd.elempart[i]; regionalToLocal.at(regional) = static_cast<int>(i);
        stream << i << ' ' << selection.originalIDs.at(regional) << ' '
               << region.worldRanks.at(selection.parts.at(regional)) << '\n';
    }
    std::vector<const InterfaceFace*> endpoints;
    for (const auto& face : plan.interfaces) {
        const int input = region.scheme ? face.hElement : face.lElement;
        const int regional = selection.globalToRegional.at(input);
        if (regionalToLocal.at(regional) >= 0) endpoints.push_back(&face);
    }
    stream << endpoints.size() << '\n';
    for (const auto* face : endpoints) {
        const int input = region.scheme ? face->hElement : face->lElement;
        const int regional = selection.globalToRegional.at(input);
        stream << face->hOriginalID << ' ' << face->hFace << ' ' << face->lOriginalID << ' '
               << face->lFace << ' ' << face->hWorldRank << ' ' << face->lWorldRank << ' '
               << regionalToLocal.at(regional) << ' ' << (selection.parts.at(regional) == localRank) << '\n';
    }
    if (!stream) throw std::runtime_error("failed writing rank-separated region sidecar");
}

inline void writePreparedRegion(PreparedRegion& region, const Master& master,
                                const Plan& plan, const std::string& directory)
{
    ensure_dir(directory);
    std::ofstream metadata(make_path(directory,"hdgldg.meta"));
    if (!metadata) throw std::runtime_error("cannot write rank-separated metadata");
    metadata << "EXASIM_HDGLDG_REGION 1\n" << region.scheme << ' '
             << region.worldRanks.size() << ' ' << region.hlSelector << ' '
             << region.hlBoundaryCode << ' ' << region.mesh.nvf << '\n';
    for (int rank : region.worldRanks) metadata << rank << ' ';
    metadata << '\n' << plan.interfaces.size() << '\n';
    for (std::size_t i=0; i<plan.interfaces.size(); ++i) {
        const auto& f=plan.interfaces[i];
        metadata << f.hOriginalID << ' ' << f.hFace << ' ' << f.lOriginalID << ' '
                 << f.lFace << ' ' << f.hWorldRank << ' ' << f.lWorldRank;
        for (int vertex : region.interfaceVertexIDs.at(i)) metadata << ' ' << vertex;
        metadata << '\n';
    }
    if (!metadata) throw std::runtime_error("failed writing rank-separated metadata");
    region.pde.datainpath = directory;
    pdeFinalizeDerived(region.pde);
    writepde(region.pde, make_path(directory,"app.bin"));
    writemaster(master, make_path(directory,"master.bin"));
    for (int rank = 0; rank < region.pde.mpiprocs; ++rank) {
        const DMD& dmd = region.partitions[rank];
        std::vector<int> f(region.mesh.nfe*dmd.elempart.size()), bf(f.size());
        select_columns(f.data(), region.mesh.f.data(), dmd.elempart.data(),
                       region.mesh.nfe, static_cast<int>(dmd.elempart.size()));
        apply_bcm(bf.data(), f.data(), region.mesh.boundaryConditions.data(),
                  static_cast<int>(f.size()), region.mesh.nbcm);
        const std::string suffix = region.pde.mpiprocs > 1 ? std::to_string(rank+1) : "";
        writesol(region.pde, region.mesh, master, dmd.elempart, make_path(directory,"sol"+suffix+".bin"));
        writemesh(region.pde, region.mesh, master, dmd, bf, make_path(directory,"mesh"+suffix+".bin"));
        writeRegionSidecar(region, plan, rank, make_path(directory,"hdgldg"+suffix+".txt"));
    }
}

}} // namespace exasim::hdgldg
#endif
