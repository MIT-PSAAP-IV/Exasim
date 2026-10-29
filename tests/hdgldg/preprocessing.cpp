#include "backend/Preprocessing/hdgldgpreprocessing.hpp"
#include <iostream>
#include <sstream>
#include <string>

using namespace exasim::hdgldg;
static int failures = 0;
static void require(bool ok, const char* text)
{ if (!ok) { ++failures; std::cerr << text << '\n'; } }
template <class F> static void rejects(F action, const char* text)
{ try { action(); require(false, text); } catch (const std::exception&) {} }

static Topology grid(int nx, int ny, int nz = 1, bool threeD = false)
{
    Topology t; t.nfe = threeD ? 6 : 4;
    const int ne = nx*ny*nz;
    t.originalIDs.resize(ne); t.neighbours.assign(ne*t.nfe, -1);
    t.oppositeFaces.assign(ne*t.nfe, -1); t.physicalBoundaryIDs.assign(ne*t.nfe, -1);
    for (int z = 0; z < nz; ++z) for (int y = 0; y < ny; ++y) for (int x = 0; x < nx; ++x) {
        const int e = x+nx*(y+ny*z); t.originalIDs[e] = 100+13*e;
        const int coords[] = {x,y,z}, sizes[] = {nx,ny,nz}, steps[] = {1,nx,nx*ny};
        for (int d = 0; d < t.nfe/2; ++d) for (int side = 0; side < 2; ++side) {
            const int f = 2*d+side, i = f+t.nfe*e;
            if ((side == 0 && coords[d] > 0) || (side == 1 && coords[d]+1 < sizes[d])) {
                t.neighbours[i] = e+(side ? steps[d] : -steps[d]);
                t.oppositeFaces[i] = 2*d+1-side;
            } else t.physicalBoundaryIDs[i] = f;
        }
    }
    return t;
}

int main()
{
    try {
        Topology ring; ring.nfe = 3;
        ring.originalIDs = {80, 13, 400, 92};
        ring.neighbours.assign(12, -1); ring.oppositeFaces.assign(12, -1);
        ring.physicalBoundaryIDs.assign(12, -1);
        for (int e = 0; e < 4; ++e) {
            ring.neighbours[3*e] = (e+3)%4; ring.oppositeFaces[3*e] = 1;
            ring.neighbours[3*e+1] = (e+1)%4; ring.oppositeFaces[3*e+1] = 0;
        }
        ring.physicalBoundaryIDs[2] = 7;
        require(selectBoundaryLayers(ring, {7}, 1) == std::vector<int>({1,0,0,0}), "one-layer labels");
        require(selectBoundaryLayers(ring, {7}, 2) == std::vector<int>({1,1,0,1}), "periodic crossing BFS");
        // A two-cell periodic chain shares two distinct faces. Endpoint
        // pairing must use the supplied face IDs, not only neighbour IDs.
        Topology two; two.nfe = 2; two.originalIDs = {91, 4};
        two.neighbours = {1,1,0,0}; two.oppositeFaces = {1,0,1,0};
        two.physicalBoundaryIDs.assign(4,-1);
        auto doubled = buildPlan(two, {1,0}, {0}, {1}, 2);
        require(doubled.interfaces.size() == 2 && doubled.interfaces[0].lFace == 1 &&
                doubled.interfaces[1].lFace == 0, "two-cell periodic face pairing");
        rejects([&] { selectBoundaryLayers(ring, {7}, 0); }, "zero layers accepted");
        auto invalid = ring; invalid.physicalBoundaryIDs[0] = 7;
        rejects([&] { checkTopology(invalid); }, "periodic neighbour seeded as physical boundary");

        auto mesh = grid(4,4);
        std::vector<int> labels(16);
        for (int e = 0; e < 16; ++e) labels[e] = (e%4 < 2 || e/4 < 2) ? 1 : 0;
        auto plan = buildPlan(mesh, labels, {1}, {0}, 2);
        Spec manifest; manifest.worldSize = 2; manifest.hWorldRanks = {1};
        manifest.lWorldRanks = {0}; manifest.selection = "labels"; manifest.labels = labels;
        std::ostringstream text; writeSpec(text, manifest);
        std::istringstream input(text.str()); const auto restored = readSpec(input);
        require(selectAndBuildPlan(mesh, restored).destinationWorld == plan.destinationWorld,
                "manifest labels roundtrip");
        Spec layerManifest = manifest; layerManifest.selection = "boundary_layers";
        layerManifest.labels.clear(); layerManifest.seedBoundaries = {7}; layerManifest.layers = 2;
        std::ostringstream layerText; writeSpec(layerText, layerManifest);
        std::istringstream layerInput(layerText.str());
        require(selectAndBuildPlan(ring, readSpec(layerInput)).labels == std::vector<int>({1,1,0,1}),
                "manifest boundary-layer roundtrip");
        auto badManifest = [&](const std::string& contents) {
            rejects([&] { std::istringstream data(contents); readSpec(data); }, "invalid manifest accepted");
        };
        badManifest(text.str()+"version=1\n");
        badManifest(text.str()+"unknown=1\n");
        badManifest(text.str()+"layers=0\n");
        badManifest("world_size=2\nh_ranks=0\nl_ranks=1\nselection=labels\nlabels=1,0\n");
        const std::string prefix = "version=1\nworld_size=2\nh_ranks=[0]\nl_ranks=1\nselection=labels\n";
        badManifest(prefix+"labels=1.0,0\n");
        badManifest(prefix+"labels=[1,0\n");
        badManifest(prefix+"labels=1,0,\n");
        badManifest(prefix+"labels=1 0\n");
        badManifest(prefix+"labels=2,0\n");
        badManifest(prefix+"labels=[]\n");
        badManifest(prefix+"labels=99999999999999999999,0\n");
        badManifest(prefix+"labels==1,0\n");
        badManifest("Version=1\nworld_size=2\nh_ranks=0\nl_ranks=1\nselection=labels\nlabels=1,0\n");
        badManifest("version=2\nworld_size=2\nh_ranks=0\nl_ranks=1\nselection=labels\nlabels=1,0\n");
        badManifest(prefix+"labels=1,0\nseed_boundaries=7\n");
        badManifest("version=1\nworld_size=2\nh_ranks=0\nl_ranks=0\nselection=labels\nlabels=1,0\n");
        badManifest("version=1\nworld_size=2\nh_ranks=0\nl_ranks=1\nselection=boundary_layers\nseed_boundaries=7\n");
        std::istringstream optionalBrackets("# explicit contract\nversion=1\nworld_size=2\nh_ranks=0\nl_ranks=[1]\nselection=labels\nlabels=[1,0] # input order\n");
        require(readSpec(optionalBrackets).labels == std::vector<int>({1,0}), "manifest brackets/comments");
        require(plan.interfaces.size() == 4, "HL face count");
        bool twoFaces = false;
        for (int l = 0; l < 16; ++l) {
            int faces = 0;
            for (const auto& face : plan.interfaces) if (face.lElement == l) ++faces;
            twoFaces = twoFaces || faces == 2;
        }
        require(twoFaces, "multiple HL faces not exercised");
        for (const auto& face : plan.interfaces) {
            require(face.hOriginalID == mesh.originalIDs[face.hElement], "H persistent ID");
            require(face.lOriginalID == mesh.originalIDs[face.lElement], "L persistent ID");
            require(face.hWorldRank == 1 && face.lWorldRank == 0, "noncontiguous ownership");
            require(mesh.neighbours[face.lFace+4*face.lElement] == face.hElement, "endpoint reversal");
        }
        std::vector<double> fields(16*3);
        for (int e = 0; e < 16; ++e) for (int c = 0; c < 3; ++c) fields[c+3*e] = 1000*mesh.originalIDs[e]+c;
        for (const auto& region : plan.regions) {
            auto extracted = extractElementColumns(fields, 3, 16, region.elements);
            for (std::size_t e = 0; e < region.elements.size(); ++e) for (int c = 0; c < 3; ++c)
                require(extracted[c+3*e] == 1000*region.originalIDs[e]+c, "aligned-field extraction");
            for (std::size_t e = 0; e < region.elements.size(); ++e)
                for (int j = region.graph.offsets[e]; j < region.graph.offsets[e+1]; ++j)
                    require(labels[region.elements[e]] == labels[region.elements[region.graph.neighbours[j]]], "cross-scheme graph edge");
        }
        rejects([&] { extractElementColumns(fields, 4, 16, plan.regions[0].elements); }, "bad field dimensions accepted");
        rejects([&] { buildPlan(mesh, labels, {0}, {0}, 2); }, "overlapping ranks accepted");
        rejects([&] { buildPlan(mesh, labels, {1}, {0}, 3); }, "unassigned rank accepted");
        rejects([&] { buildPlan(mesh, std::vector<int>(16,1), {1}, {0}, 2); }, "empty L region accepted");
        invalid = mesh; invalid.originalIDs[0] = invalid.originalIDs[1];
        rejects([&] { buildPlan(invalid, labels, {1}, {0}, 2); }, "duplicate original ID accepted");
        invalid = mesh; invalid.oppositeFaces[1] = 1;
        rejects([&] { checkTopology(invalid); }, "nonreciprocal graph accepted");

        auto cube = grid(2,2,2,true);
        auto three = buildPlan(cube, {1,0,1,0,1,0,1,0}, {0}, {1}, 2);
        require(three.interfaces.size() == 4, "3D planar endpoints");
        for (const auto& face : three.interfaces) require(face.hFace == 1 && face.lFace == 0, "3D face direction");

#if defined(HAVE_PARMETIS) || defined(HAVE_METIS)
        auto large = grid(6,6);
        std::vector<int> split(36);
        for (int e = 0; e < 36; ++e) split[e] = e%6 < 3;
        auto distributed = buildPlan(large, split, {1,3}, {0,2}, 4, metisPartitionGraph);
        auto unequal = buildPlan(large, split, {1}, {0,2}, 3, metisPartitionGraph);
        require(unequal.regions[0].originalIDs == distributed.regions[0].originalIDs &&
                unequal.regions[1].originalIDs == distributed.regions[1].originalIDs,
                "selection changed with rank split");
        auto repeat = buildPlan(large, split, {1,3}, {0,2}, 4, metisPartitionGraph);
        require(repeat.destinationWorld == distributed.destinationWorld, "METIS repeat changed destinations");
        auto unsorted = buildPlan(large, split, {3,1}, {2,0}, 4, metisPartitionGraph);
        require(unsorted.destinationWorld == distributed.destinationWorld, "rank-set order changed local rank mapping");
        std::vector<int> owned(4,0);
        for (int e = 0; e < 36; ++e) {
            const int rank = distributed.destinationWorld[e]; ++owned[rank];
            require((rank == 1 || rank == 3) == (split[e] == 1), "METIS homogeneous destinations");
        }
        for (int n : owned) require(n > 0, "METIS empty rank");
        require(distributed.interfaces.size() == 6, "METIS endpoint preservation");
        rejects([&] { metisPartitionGraph(Graph{}, 2); }, "empty CSR accepted");
        std::cout << "METIS parts: " << owned[0] << ' ' << owned[1] << ' ' << owned[2] << ' ' << owned[3] << '\n';
#endif
    } catch (const std::exception& e) {
        ++failures; std::cerr << "unexpected: " << e.what() << '\n';
    }
    std::cout << "rank-separated preprocessing failures=" << failures << '\n';
    return failures ? 1 : 0;
}
