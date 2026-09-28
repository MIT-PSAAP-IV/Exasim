#include "exasim_paths.h"  // exasim_data_dir()
#include <cmath>
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>

// ---------------------------------------------------------------------------
// Surface corner handling. The face nodes arrive in reference-lattice order
// (see the surfLattice* helpers below), so the k corners are known a priori
// from surfLatticeBoundary -- no geometric corner search is needed. Angle
// sorting (orderCorners) then gives a consistent cyclic winding in physical
// space, and surfCalibrateOrientation picks the lattice traversal (possibly
// mirrored) that matches it.
// ---------------------------------------------------------------------------

// Cyclically order a set of k face corner nodes around the face.
static inline void orderCorners(const dstype* plane, int ncx, int k, int* corners)
{
    if (k < 3) return; // endpoints: order irrelevant for a line
    const dstype* p0 = &plane[corners[0]*ncx];
    const dstype* p1 = &plane[corners[1]*ncx];
    const dstype* p2 = &plane[corners[2]*ncx];
    double u[3] = {p1[0]-p0[0], (ncx>1)?(p1[1]-p0[1]):0.0, (ncx>2)?(p1[2]-p0[2]):0.0};
    double v[3] = {p2[0]-p0[0], (ncx>1)?(p2[1]-p0[1]):0.0, (ncx>2)?(p2[2]-p0[2]):0.0};
    double n[3] = {u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]};
    double nn = std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
    if (nn < 1e-30) return;                    // degenerate: keep input order
    double c[3] = {0,0,0};
    for (int i = 0; i < k; ++i) for (int d = 0; d < ncx; ++d) c[d] += plane[corners[i]*ncx+d];
    for (int d = 0; d < 3; ++d) c[d] /= k;
    double e[3] = {plane[corners[0]*ncx+0]-c[0],
                   (ncx>1)?(plane[corners[0]*ncx+1]-c[1]):0.0,
                   (ncx>2)?(plane[corners[0]*ncx+2]-c[2]):0.0};
    // u = e projected onto plane perpendicular to n
    double dot = e[0]*n[0]+e[1]*n[1]+e[2]*n[2];
    double uu[3] = {e[0]-dot*n[0], e[1]-dot*n[1], e[2]-dot*n[2]};
    double un = std::sqrt(uu[0]*uu[0]+uu[1]*uu[1]+uu[2]*uu[2]);
    if (un < 1e-30) return;
    double ax[3] = {uu[0]/un, uu[1]/un, uu[2]/un};
    double ay[3] = {n[1]*ax[2]-n[2]*ax[1], n[2]*ax[0]-n[0]*ax[2], n[0]*ax[1]-n[1]*ax[0]};
    double ang[4];
    for (int i = 0; i < k; ++i)
        ang[i] = std::atan2((plane[corners[i]*ncx+0]-c[0])*ay[0]
                            + ((ncx>1)?(plane[corners[i]*ncx+1]-c[1])*ay[1]:0)
                            + ((ncx>2)?(plane[corners[i]*ncx+2]-c[2])*ay[2]:0),
                            (plane[corners[i]*ncx+0]-c[0])*ax[0]
                            + ((ncx>1)?(plane[corners[i]*ncx+1]-c[1])*ax[1]:0)
                            + ((ncx>2)?(plane[corners[i]*ncx+2]-c[2])*ax[2]:0));
    int order[4] = {0,1,2,3};
    for (int i = 1; i < k; ++i)
        for (int j = i; j > 0 && ang[j-1] > ang[j]; --j) std::swap(ang[j-1], ang[j]), std::swap(order[j-1], order[j]);
    int tmp[4];
    for (int i = 0; i < k; ++i) tmp[i] = corners[order[i]];
    for (int i = 0; i < k; ++i) corners[i] = tmp[i];
}

// Face-lattice helpers for subdivided surface output. Face nodes arrive in
// lattice order (verified from master data and runs): lines ascending with
// endpoints at 0/np f-1; quads column-major tensor (idx=i+(p+1)*j);
// tris row-major uniform lattice (idx=j*(p+1)-j*(j-1)/2+i).
// Subdivision emits linear sub-cells over ALL face nodes so every computed
// kernel value is plotted (corner-only output would discard the interior
// face data). Winding is calibrated per face against orderCorners output
// so the subdivided patch matches the corner-cell orientation.
static inline int surfLatticeDegree(int npf, int isTri, int isQuad)
{
    if (!isTri && !isQuad) return (npf >= 2) ? npf - 1 : -1; // lines
    for (int p = 1; p <= 16; ++p) {
        if (isQuad && (p + 1) * (p + 1) == npf) return p;
        if (isTri && (p + 1) * (p + 2) / 2 == npf) return p;
    }
    return -1;
}

// Lattice index of grid position (i,j): quads tensor, tris uniform lattice.
static inline int surfLatticeIdx(int i, int j, int p, int isTri, int isQuad, int mirror)
{
    if (mirror) { int t = i; i = j; j = t; }
    if (isQuad) return i + (p + 1) * j;
    if (isTri) return j * (p + 1) - j * (j - 1) / 2 + i;
    return i; // lines: j == 0
}

// Canonical lattice boundary traversal (corner local indices, CCW in
// lattice coords).
static inline void surfLatticeBoundary(int* out, int p, int isTri, int isQuad)
{
    if (!isTri && !isQuad) { out[0] = 0; out[1] = p; return; } // lines
    if (isQuad) {
        out[0] = 0; out[1] = p; out[2] = p * p + 2 * p; out[3] = p * (p + 1);
        return;
    }
    out[0] = 0; out[1] = p; out[2] = p * (p + 3) / 2; // tris
}

// Compare corners[] against the lattice boundary traversal, up to rotation,
// in either direction. Returns +1 (agree), -1 (mirror lattice), 0 (no match
// -> caller falls back to a single corner cell). Only meaningful for k >= 3;
// for lines (k == 2) forward and reverse coincide, so the caller orients line
// segments directly from the corner order.
static inline int surfCalibrateOrientation(const int* corners, int k, int p,
                                           int isTri, int isQuad)
{
    int lat[4];
    surfLatticeBoundary(lat, p, isTri, isQuad);
    for (int r = 0; r < k; ++r) {
        int okFwd = 1, okRev = 1;
        for (int i = 0; i < k; ++i) {
            if (corners[i] != lat[(r + i) % k]) okFwd = 0;
            if (corners[i] != lat[(r + k - i) % k]) okRev = 0;
        }
        if (okFwd) return +1;
        if (okRev) return -1;
    }
    return 0;
}

class CVisualization {
public:
    float* scafields=nullptr;
    float* vecfields=nullptr;
    float* tenfields=nullptr;
    float* srffields=nullptr;

    // Geometry (owned; always 3D padded: [3 x npoints], column-major)
    std::vector<float>  cgnodes;
    
    // Topology (owned)
    std::vector<int32_t> cgcells;      // [nve x ncells], column-major (0-based)
    std::vector<int32_t> celloffsets;  // [ncells] cumulative ends (nve, 2*nve, ...)
    std::vector<uint8_t> celltypes;    // [ncells] VTK cell type per cell (uniform)

    // Geometry (owned; always 3D padded: [3 x nnodes], column-major)
    std::vector<float>  facenodes;
    
    // Topology (owned)
    std::vector<int32_t> faceconn;     // [nvf x nfaces], column-major (0-based)
    std::vector<int32_t> faceoffsets;  // [ncells] cumulative ends (nve, 2*nve, ...)
    std::vector<uint8_t> facetypes;    // [ncells] VTK cell type per cell (uniform)
    
    // Field schemas (names only; data are provided at write time)
    std::vector<std::string> scalar_names;   // nscalars
    std::vector<std::string> vector_names;   // nvectors (3 comps each, z padded if nd==2)
    std::vector<std::string> tensor_names;   // ntensors (ntc comps each, ntc=nd*nd)
    std::vector<std::string> surface_names;   // nsurfq (surface scalar vis fields)

    // ------------------------------------------------------------------
    // Surface visualization: the ibs-list boundary faces, resolved on the
    // DG face nodes. Every face node is a unique surface point, and the
    // SurfaceQuantities evaluator supplies the field values directly.
    int   nsurfq       = 0;    // number of surface scalar fields (>=0)
    int   surf_nnodes    = 0;    // surface nodes (all face nodes, DG-unique per face)
    int   surf_ncells    = 0;    // linear sub-cells (lines / tris / quads)
    int   surf_k         = 0;    // corners per cell (2, 3 or 4)
    bool  surfvis_enabled = false;
    std::vector<float>   surf_nodes;      // [3 x surf_nnodes]
    std::vector<int32_t> surf_cellconn;   // [surf_k x surf_ncells]
    std::vector<int32_t> surf_face2cell;  // [nf] surf-face ordinal owning the face, or -1
    std::vector<int32_t> surf_celloffsets;// [surf_ncells]
    std::vector<uint8_t> surf_celltypes;  // [surf_ncells]
    // Precomputed appended-data offsets for the surface VTU metadata
    // (volume analogue: scalar_offsets / points_offset / ... below).
    std::vector<std::uint64_t> surf_scalar_offsets; // [nsurfq]
    std::uint64_t surf_points_offset = 0;
    std::uint64_t surf_conn_offset   = 0;
    std::uint64_t surf_offs_offset   = 0;
    std::uint64_t surf_types_offset  = 0;

    // how fields were allocated: 0=CPU malloc/free, 2=CUDA host (cudaHostAlloc),
    // 3=HIP  host (hipHostMalloc), anything else => unknown/none
    int host_alloc_backend = 0;

    // Sizes / meta
    int nd      = 0;   // spatial dimension (2 or 3)
    int ntc     = 0;   // tensor components = nd*nd (4 or 9)
    int npoints = 0;
    int ncells  = 0;
    int nve     = 0;
    int nnodes  = 0;
    int nfaces  = 0;
    int nvf     = 0;    
    int savemode = 0;
    int rank = 0;
    
    // Precomputed appended-data offsets for VTU metadata
    std::vector<std::uint64_t> scalar_offsets; // size = nscalars
    std::vector<std::uint64_t> vector_offsets; // size = nvectors
    std::vector<std::uint64_t> tensor_offsets; // size = ntensors
    std::uint64_t points_offset = 0;
    std::uint64_t conn_offset   = 0;
    std::uint64_t offs_offset   = 0;
    std::uint64_t types_offset  = 0;

public:
    void UpdateCoordinates(const dstype* xdg, const int* cgent2dgent,
                           const int* colent2elem, const int* rowent2elem,
                           int ne, int ndg, int backend)
    {
        if (npoints <= 0) return;

        float* coordinates = cgnodes.data();
        float* deviceCoordinates = nullptr;
        if (backend > 1) {
            TemplateMalloc(&deviceCoordinates, 3*npoints, backend);
            coordinates = deviceCoordinates;
        }

        ArraySetValue<float>(coordinates, 0.0f, 3*npoints);
        VisDG2CG(coordinates, xdg, cgent2dgent, colent2elem, rowent2elem,
                 ne, npoints, ndg, 3, nd, 1);
        Kokkos::fence();

        if (deviceCoordinates != nullptr) {
            TemplateCopytoHost(cgnodes.data(), deviceCoordinates, 3*npoints, backend);
            TemplateFree(deviceCoordinates, backend);
        }
    }

    // Ordinal of face f among the selected surface faces, or -1 when the
    // face is not selected. Out-of-range faces (a face map that predates a
    // topology change) also yield -1 here; the SaveSurfaces precondition
    // reports that case loudly instead of indexing out of bounds.
    int vis_face_ordinal(Int f) const
    {
        if (f < 0 || (size_t)f >= surf_face2cell.size()) return -1;
        return surf_face2cell[(size_t)f];
    }

    // Refresh the surface point coordinates from the current xdg. Mesh
    // adaptation moves xdg in place without changing face topology, so the
    // corner/sub-cell layout (surf_cellconn, surf_ncells) and the face
    // ordinals (surf_face2cell) stay valid and only surf_nodes goes stale.
    // This is the surface counterpart of UpdateCoordinates above (which the
    // volume writer calls before every write); SaveSurfaces calls this
    // before every save for the same reason. No DG->CG averaging here:
    // DG surface nodes are written at their own face coordinates.
    void UpdateSurfaceCoordinates(CDiscretization& disc, int backend)
    {
        if (!surfvis_enabled || surf_nnodes <= 0) return;

        const auto& common = disc.common;
        const auto& mesh   = disc.mesh;
        const auto& sol    = disc.sol;

        const Int nbf = common.meshsizes.nbf;
        const Int npf = common.grid.npf;
        const Int ncx = common.components.ncx;

        // One staging buffer per call, reused across blocks (as in
        // UpdateCoordinates above), sized for the largest selected block.
        // f1/f2 use the same 1-based-to-0-based conversion as the work loop
        // below, so maxnf is the true largest nfblk, not one face short.
        Int maxnf = 0;
        for (Int j = 0; j < nbf; ++j) {
            if (!common.qoiparams.isSaveBoundary(common.fblks[3*j+2])) continue;
            Int f1 = common.fblks[3*j] - 1;
            Int f2 = common.fblks[3*j+1];
            maxnf = std::max(maxnf, f2 - f1);
        }
        if (maxnf == 0) return;
        std::vector<dstype> xg((size_t)npf*maxnf*ncx);
        dstype* dbuf = nullptr;
        if (backend >= 2) TemplateMalloc(&dbuf, npf*maxnf*ncx, backend);

        for (Int j = 0; j < nbf; ++j) {
            Int ib = common.fblks[3*j+2];
            if (!common.qoiparams.isSaveBoundary(ib)) continue;
            Int f1 = common.fblks[3*j] - 1;
            Int f2 = common.fblks[3*j+1];
            Int nfblk = f2 - f1;
            if (nfblk == 0) continue;
            Int nn = npf*nfblk;
            gatherBoundaryNodes(xg.data(), sol.xdg, mesh.findxdg1, f1, nn, npf, ncx, backend, dbuf);
            for (Int ff = 0; ff < nfblk; ++ff) {
                Int f = f1 + ff;
                Int o = vis_face_ordinal(f);
                if (o < 0) continue;
                for (Int ln = 0; ln < npf; ++ln) {
                    // xg is component-major [dim][block-point], same
                    // transpose as in InitSurfaces.
                    const size_t s = (size_t)o*npf + ln;
                    surf_nodes[3*s+0] = (float)xg[(size_t)0*nn + (size_t)ff*npf + ln];
                    surf_nodes[3*s+1] = (ncx>1)?(float)xg[(size_t)1*nn + (size_t)ff*npf + ln]:0.0f;
                    surf_nodes[3*s+2] = (ncx>2)?(float)xg[(size_t)2*nn + (size_t)ff*npf + ln]:0.0f;
                }
            }
        }
        if (dbuf) TemplateFree(dbuf, backend);
    }

    // CVisualization(const dstype* xcg, int nd_in, int np,
    //                const int* cgelcon, int npe, int ne,
    //                const int* telem,   int nce, int nverts_per_cell,
    //                int elemtype,
    //                const std::vector<std::string>& scalars,
    //                const std::vector<std::string>& vectors,
    //                const std::vector<std::string>& tensors,
    //                const std::vector<std::string>& surfaces)
    // {
    //     if (np > 0) {
    //         Init(xcg, nd_in, np, cgelcon, npe, ne,
    //              telem, nce, nverts_per_cell, elemtype,
    //              scalars, vectors, tensors, surfaces);            
    //     }
    // }

    CVisualization(CDiscretization& disc, int backend) {      
        rank = disc.common.mpiRank;
        int nd_in   = disc.common.grid.nd;
        int npoints_in = disc.sol.szxcg / nd_in;
        
        if (npoints_in > 0 && nd_in > 1) {            
            int porder  = disc.common.grid.porder;        
            int nsca    = disc.common.qoiparams.nsca;
            int nvec    = disc.common.qoiparams.nvec;            
            int nten    = disc.common.qoiparams.nten;            
            int nsurfq= disc.common.qoiparams.nsurfq;
            int npe     = disc.common.grid.npe;
            int ne      = disc.common.meshsizes.ne1;
            int elemtype= disc.common.grid.elemtype;
            int nve_in  = (elemtype==0) ? (nd_in + 1) : std::pow(2, nd_in);
                
            std::string fn1 = make_path(exasim_data_dir(), "masternodes.bin");
            std::vector<dstype> xpe, xpf;
            std::vector<int> telem, tface, perm;
            masternodes(xpe, telem, xpf, tface, perm, porder, nd_in, elemtype, fn1);
    
            int nce = (int)telem.size() / nve_in;
    
            std::vector<std::string> scalars(nsca);
            for (int i = 0; i < nsca; i++) scalars[i] = "Scalar Field " + std::to_string(i);
    
            std::vector<std::string> vectors(nvec);
            for (int i = 0; i < nvec; i++) vectors[i] = "Vector Field " + std::to_string(i);
    
            std::vector<std::string> tensors(nten);
            for (int i = 0; i < nten; i++) tensors[i] = "Tensor Field " + std::to_string(i);

            std::vector<std::string> surfaces(nsurfq);
            for (int i = 0; i < nsurfq; i++) surfaces[i] = "Surface Field " + std::to_string(i);
            
            int* cgelcon;
            if (backend==0) cgelcon = &disc.mesh.cgelcon[0];
            else {
                TemplateMalloc(&cgelcon, npe*ne, 0);
                TemplateCopytoHost(cgelcon, disc.mesh.cgelcon, npe*ne, backend);
            }            
            
            Init(disc.sol.xcg, nd_in, npoints_in, cgelcon, npe, ne,
                 telem.data(), nce, nve_in, elemtype,
                 scalars, vectors, tensors, surfaces);

            if (backend != 0) CPUFREE(cgelcon);    

            surfvis_enabled = (disc.common.qoiparams.saveParaview != 0) && (nsurfq > 0) &&
                              !disc.common.qoiparams.ibslist.empty();
            this->nsurfq   = nsurfq;
            surf_scalar_offsets.assign(nsurfq, 0);
            if (surfvis_enabled) InitSurfaces(disc, backend);

            savemode = (disc.common.qoiparams.saveParaview != 0) && (nsca + nvec + nten > 0 || surfvis_enabled); 
        
            if (backend==2) { // GPU
            #ifdef HAVE_CUDA        
                cudaTemplateHostAlloc(&scafields, npoints*nsca, cudaHostAllocMapped); // zero copy
                cudaTemplateHostAlloc(&vecfields, 3*npoints*nvec, cudaHostAllocMapped); // zero copy
                cudaTemplateHostAlloc(&tenfields, ntc*npoints*nten, cudaHostAllocMapped); // zero copy
                host_alloc_backend = 2;
            #endif                  
            }
            else if (backend==3) { // GPU
            #ifdef HAVE_HIP        
                hipTemplateHostMalloc(&scafields, npoints*nsca, hipHostMallocMapped); // zero copy
                hipTemplateHostMalloc(&vecfields, 3*npoints*nvec, hipHostMallocMapped); // zero copy
                hipTemplateHostMalloc(&tenfields, ntc*npoints*nten, hipHostMallocMapped); // zero copy
                host_alloc_backend = 3;
            #endif                  
            }    
            else { // CPU
                scafields = (float *) malloc(npoints*nsca*sizeof(float));
                vecfields = (float *) malloc(3*npoints*nvec*sizeof(float));
                tenfields = (float *) malloc(ntc*npoints*nten*sizeof(float));
                host_alloc_backend = 0;
            }

            // Surface fields are host-only data: SaveSurfaces scatters into
            // srffields on the host and surfvtuwrite serializes from it, so unlike
            // the volume fields (consumed on-device by VisDG2CG) mapped memory
            // buys nothing. Ordinary host storage, allocated whenever surface
            // output is requested -- even when this rank owns no tagged faces:
            // an untagged MPI rank must still emit a valid (empty) piece so the
            // PVTU references resolve, and surfvtuwrite treats null data as an
            // error whenever surface names exist. Mirrors the volume path,
            // which always allocates its field buffers.
            if (nsurfq > 0) {
                const size_t nval = (size_t)std::max(surf_nnodes, 1) * (size_t)nsurfq;
                srffields = (float *) malloc(nval * sizeof(float));
                for (int i = 0; i < surf_nnodes*nsurfq; i++) srffields[i] = 0.0;
            }
            
            for (int i = 0; i < npoints*nsca; i++) scafields[i] = 0.0;
            for (int i = 0; i < 3*npoints*nvec; i++) vecfields[i] = 0.0;
            for (int i = 0; i < ntc*npoints*nten; i++) tenfields[i] = 0.0;

            //cout<<ne<<"  "<<npoints<<endl;
            if (disc.common.mpiRank == 0) printf("finish CVisualization constructor... \n");    
        }        
    }

    ~CVisualization() {
        // Free scafields / vecfields / tenfields / srffields according to how they were allocated.
        auto free_field = [this](float*& p) {
            if (!p) return;
            switch (host_alloc_backend) {
                case 2:  // CUDA pinned host
                #ifdef HAVE_CUDA
                    cudaFreeHost(p);
                    p = nullptr;
                    break;
                #endif
    
                case 3:  // HIP pinned host
                #ifdef HAVE_HIP
                    hipHostFree(p);
                    p = nullptr;
                    break;
                #endif
    
                case 0:  // CPU malloc
                     std::free(p);
                     p = nullptr;
                     break;

                default:
                    std::free(p);
                    p = nullptr;
                    break;
            }
        };

        free_field(scafields);
        free_field(vecfields);
        free_field(tenfields);
        // srffields is always plain host storage (see constructor), so it must
        // not go through free_field: on a CUDA/HIP build that would call
        // cudaFreeHost/hipHostFree on a malloc'd pointer.
        std::free(srffields);
        srffields = nullptr;
        if (rank==0) printf("CVisualization is freed successfully.\n");
    }

    // ============================ Writers ============================

    // scalarfields: [npoints * nscalars]
    // vectorfields: [3*npoints * nvectors] (z padded if nd==2)
    // tensorfields: [ntc*npoints * ntensors], ntc = nd*nd (4 for 2D, 9 for 3D)
    void vtuwrite(const std::string& filename_no_ext,
                  const float* scalarfields,   // nullptr allowed iff scalar_names empty
                  const float* vectorfields = nullptr,   // nullptr allowed iff vector_names empty
                  const float* tensorfields = nullptr) const // nullptr allowed iff tensor_names empty
    {
        if (!scalar_names.empty() && !scalarfields)
            throw std::invalid_argument("vtuwrite: scalarfields is null but scalar_names is non-empty.");
        if (!vector_names.empty() && !vectorfields)
            throw std::invalid_argument("vtuwrite: vectorfields is null but vector_names is non-empty.");
        if (!tensor_names.empty() && !tensorfields)
            throw std::invalid_argument("vtuwrite: tensorfields is null but tensor_names is non-empty.");

        const std::string filename = filename_no_ext + ".vtu";
        std::ofstream os(filename, std::ios::binary);
        if (!os) throw std::runtime_error("Cannot open output file: " + filename);

        os << "<?xml version=\"1.0\"?>\n";
        os << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\""
           << vtk_byte_order() << "\" header_type=\"UInt64\">\n";
        os << "  <UnstructuredGrid>\n";
        os << "    <Piece NumberOfPoints=\"" << npoints
           << "\" NumberOfCells=\"" << ncells << "\">\n";

        // PointData schema
        if (!scalar_names.empty() || !vector_names.empty() || !tensor_names.empty()) {
            os << "      <PointData Scalars=\"scalars\">\n";
            for (int i = 0; i < (int)scalar_names.size(); ++i)
                os << "        <DataArray type=\"Float32\" Name=\"" << scalar_names[i]
                   << "\" Format=\"appended\" offset=\"" << scalar_offsets[i] << "\"/>\n";
            for (int i = 0; i < (int)vector_names.size(); ++i)
                os << "        <DataArray type=\"Float32\" Name=\"" << vector_names[i]
                   << "\" NumberOfComponents=\"3\" Format=\"appended\" offset=\"" << vector_offsets[i] << "\"/>\n";
            for (int i = 0; i < (int)tensor_names.size(); ++i)
                os << "        <DataArray type=\"Float32\" Name=\"" << tensor_names[i]
                   << "\" NumberOfComponents=\"" << ntc << "\" Format=\"appended\" offset=\"" << tensor_offsets[i] << "\"/>\n";
            os << "      </PointData>\n";
        }

        // Points schema
        os << "      <Points>\n";
        os << "        <DataArray type=\"Float32\" Name=\"points\" NumberOfComponents=\"3\" Format=\"appended\" offset=\""
           << points_offset << "\"/>\n";
        os << "      </Points>\n";

        // Cells schema
        os << "      <Cells>\n";
        os << "        <DataArray type=\"Int32\" Name=\"connectivity\" Format=\"appended\" offset=\"" << conn_offset << "\"/>\n";
        os << "        <DataArray type=\"Int32\" Name=\"offsets\"     Format=\"appended\" offset=\"" << offs_offset << "\"/>\n";
        os << "        <DataArray type=\"UInt8\" Name=\"types\"       Format=\"appended\" offset=\"" << types_offset << "\"/>\n";
        os << "      </Cells>\n";

        os << "    </Piece>\n";
        os << "  </UnstructuredGrid>\n";
        os << "  <AppendedData encoding=\"raw\">\n";
        os << "   _";

        // Appended blocks
        for (int si = 0; si < (int)scalar_names.size(); ++si) {
            //const dstype* src = &scalarfields[npoints * si];
            //write_as_float32(os, src, npoints);
            write_block(os, filename, "scalar:" + scalar_names[si],
                        &scalarfields[npoints * si],
                        byte_count(npoints, sizeof(float)));
        }       

        for (int vi = 0; vi < (int)vector_names.size(); ++vi) {
            //const dstype* src = &vectorfields[3 * npoints * vi];
            //write_as_float32(os, src, 3 * npoints);
            write_block(os, filename, "vector:" + vector_names[vi],
                        &vectorfields[3 * npoints * vi],
                        byte_count(3, npoints, sizeof(float)));
        }

        for (int ti = 0; ti < (int)tensor_names.size(); ++ti) {
            //const dstype* src = &tensorfields[ntc * npoints * ti];
            //write_as_float32(os, src, ntc * npoints);
            write_block(os, filename, "tensor:" + tensor_names[ti],
                        &tensorfields[ntc * npoints * ti],
                        byte_count(ntc, npoints, sizeof(float)));
        }

        write_block(os, filename, "points", cgnodes.data(),
                    byte_count(3, npoints, sizeof(float)));
        write_block(os, filename, "connectivity", cgcells.data(),
                    byte_count(nve, ncells, sizeof(int32_t)));
        write_block(os, filename, "offsets", celloffsets.data(),
                    byte_count(ncells, sizeof(int32_t)));
        write_block(os, filename, "types", celltypes.data(),
                    byte_count(ncells, sizeof(uint8_t)));

        os << "\n  </AppendedData>\n";
        os << "</VTKFile>\n";
        os.close();
    }

    // Parallel writer (rank pieces + PVTU on rank 0)
    void vtuwrite_parallel(const std::string& base_name,
                           int rank, int nranks,
                           const float* scalarfields,
                           const float* vectorfields = nullptr,
                           const float* tensorfields = nullptr) const
    {
        const std::string my_base = base_name + rank_tag(rank);
        vtuwrite(my_base, scalarfields, vectorfields, tensorfields);

        if (rank == 0) {
            std::vector<std::string> pieces;
            pieces.reserve(nranks);
            for (int r = 0; r < nranks; ++r) {
                const std::filesystem::path piece = base_name + rank_tag(r) + ".vtu";
                pieces.push_back(piece.filename().generic_string());
            }
            write_pvtu(base_name, pieces, scalar_names, vector_names, tensor_names, ntc);
        }
    }

    // Surface writer (serial): scalar surface fields on a boundary surface mesh.
    void surfvtuwrite(const std::string& filename_no_ext,
                      const float* srffields_data) const
    {
        if (!surface_names.empty() && !srffields_data)
            throw std::invalid_argument("surfvtuwrite: srffields_data is null but surface_names is non-empty.");

        const std::string filename = filename_no_ext + ".vtu";
        std::ofstream os(filename, std::ios::binary);
        if (!os) throw std::runtime_error("Cannot open output file: " + filename);

        os << "<?xml version=\"1.0\"?>\n";
        os << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\""
           << vtk_byte_order() << "\" header_type=\"UInt64\">\n";
        os << "  <UnstructuredGrid>\n";
        os << "    <Piece NumberOfPoints=\"" << surf_nnodes
           << "\" NumberOfCells=\"" << surf_ncells << "\">\n";
        if (!surface_names.empty()) {
            os << "      <PointData Scalars=\"scalars\">\n";
            for (int s = 0; s < nsurfq; ++s)
                os << "        <DataArray type=\"Float32\" Name=\"" << surface_names[s]
                   << "\" Format=\"appended\" offset=\"" << surf_scalar_offsets[s] << "\"/>\n";
            os << "      </PointData>\n";
        }
        os << "      <Points>\n";
        os << "        <DataArray type=\"Float32\" Name=\"points\" NumberOfComponents=\"3\""
           << " Format=\"appended\" offset=\"" << surf_points_offset << "\"/>\n";
        os << "      </Points>\n";
        os << "      <Cells>\n";
        os << "        <DataArray type=\"Int32\" Name=\"connectivity\" Format=\"appended\" offset=\"" << surf_conn_offset << "\"/>\n";
        os << "        <DataArray type=\"Int32\" Name=\"offsets\"     Format=\"appended\" offset=\"" << surf_offs_offset << "\"/>\n";
        os << "        <DataArray type=\"UInt8\" Name=\"types\"       Format=\"appended\" offset=\"" << surf_types_offset << "\"/>\n";
        os << "      </Cells>\n";
        os << "    </Piece>\n";
        os << "  </UnstructuredGrid>\n";
        os << "  <AppendedData encoding=\"raw\">\n";
        os << "   _";
        for (int s = 0; s < nsurfq; ++s)
            write_block(os, filename, "surfscalar:" + surface_names[s],
                        &srffields_data[surf_nnodes * s],
                        byte_count(surf_nnodes, sizeof(float)));
        write_block(os, filename, "points", surf_nodes.data(),
                    byte_count(3, surf_nnodes, sizeof(float)));
        write_block(os, filename, "connectivity", surf_cellconn.data(),
                    byte_count(surf_k, surf_ncells, sizeof(int32_t)));
        write_block(os, filename, "offsets", surf_celloffsets.data(),
                    byte_count(surf_ncells, sizeof(int32_t)));
        write_block(os, filename, "types", surf_celltypes.data(),
                    byte_count(surf_ncells, 1));
        os << "\n  </AppendedData>\n";
        os << "</VTKFile>\n";
        os.close();
    }

    // Parallel surface writer: rank pieces + PVTU on rank 0.
    void surfvtuwrite_parallel(const std::string& base_name,
                               int rank, int nranks,
                               const float* srffields_data) const
    {
        surfvtuwrite(base_name + rank_tag(rank), srffields_data);
        if (rank == 0) {
            std::vector<std::string> pieces;
            pieces.reserve(nranks);
            for (int r = 0; r < nranks; ++r) {
                const std::filesystem::path piece = base_name + rank_tag(r) + ".vtu";
                pieces.push_back(piece.filename().generic_string());
            }
            write_pvtu(base_name, pieces, surface_names, {}, {}, ntc);
        }
    }

    // PVD writers (unchanged)
    static void pvdwrite(const std::string& pvd_name_no_ext,
                         const std::vector<std::string>& files,
                         const std::vector<float>& times) {
        if (files.size() != times.size())
            throw std::invalid_argument("pvdwrite: files and times must have same length.");
        const std::string pvdfile = pvd_name_no_ext + ".pvd";
        std::ofstream os(pvdfile, std::ios::binary);
        if (!os) throw std::runtime_error("Cannot open PVD file: " + pvdfile);
        os << "<?xml version=\"1.0\"?>\n";
        os << "<VTKFile type=\"Collection\" version=\"0.1\"\n";
        os << "         byte_order=\"" << vtk_byte_order() << "\"\n";
        os << "         compressor=\"vtkZLibDataCompressor\">\n";
        os << "  <Collection>\n";
        for (int i = 0; i < (int)files.size(); ++i) {
            os << "    <DataSet timestep=\"" << times[i] << "\" group=\"\" part=\"0\"\n";
            os << "             file=\"" << files[i] << "\"/>\n";
        }
        os << "  </Collection>\n";
        os << "</VTKFile>\n";
        if (!os) throw std::runtime_error("Error writing PVD file.");
    }

    static void pvdwrite_series(const std::string& base,
                                const dstype* dt, int nt, int nm,
                                const std::string& ext /* "vtu" or "pvtu" */) {
        if (!dt && nt > 0) throw std::invalid_argument("pvdwrite_series: dt pointer is null.");
        if (nt < 0)        throw std::invalid_argument("pvdwrite_series: nt must be non-negative.");
        if (!(ext == "vtu" || ext == "pvtu"))
            throw std::invalid_argument("pvdwrite_series: ext must be \"vtu\" or \"pvtu\".");
        // Strip directory prefix so .pvd uses relative paths
        std::string base_name = base;
        auto pos = base.find_last_of("/\\");
        if (pos != std::string::npos)
            base_name = base.substr(pos + 1);
        std::vector<std::string> files; files.reserve(nt);
        std::vector<float>       times; times.reserve(nt);
        float t = 0.0f;
        for (int k = 0; k < nt; ++k) {
            t += dt[k];
            if ((k+1) % nm == 0) {
                files.push_back(base_name + "_" + step_tag(k+1) + "." + ext);
                times.push_back(t);
            }
        }
        //cout<<nt<<",  "<<nm<<",  "<<base<<endl;        
        pvdwrite(base, files, times);
    }

private:
    // Gather one boundary face block's DG node coordinates out of xdg.
    // Output xg is component-major [ncx][nn] (see faceindex1 in
    // backend/Common/cpuimpl.h). Shared by InitSurfaces (construction-time
    // snapshot) and UpdateSurfaceCoordinates (per-save refresh). On device
    // backends the caller supplies dbuf, a device buffer of at least nn*ncx
    // entries, so the staging allocation happens once per call (as in
    // UpdateCoordinates) rather than once per block.
    static void gatherBoundaryNodes(dstype* xg, const dstype* xdg, const int* findxdg1,
                                    Int f1, Int nn, Int npf, Int ncx, int backend,
                                    dstype* dbuf)
    {
        if (backend >= 2) {
            GetArrayAtIndex(dbuf, xdg, &findxdg1[npf*ncx*f1], nn*ncx);
            TemplateCopytoHost(xg, dbuf, nn*ncx, backend);
        } else {
            GetArrayAtIndex(xg, xdg, &findxdg1[npf*ncx*f1], nn*ncx);
        }
        // The writer consumes xg on the host immediately; fence so an
        // asynchronous device backend cannot race the transpose below.
        Kokkos::fence();
    }

    // Build the boundary surface mesh (points/topology) for the requested tag
    // and the per-node mappings used by SaveSurfaces to scatter the surface
    // scalar fields onto the surface nodes.
    //
    // NOTE (DG surface mesh): each tagged face contributes ALL its npf face
    // nodes as surface nodes (placed at the face's own coordinates), split
    // into linear sub-cells (lines: npf-1 segments; quads: p^2 quads;
    // tris: p^2 tris). There is deliberately NO matching against volume CG
    // nodes and NO deduplication: DG fields are discontinuous across faces,
    // so merging shared corners would mix values from different faces
    // (last-wins) and teleport points when the nearest CG node is far
    // (curved faces, ragged partitions). Sub-cell winding is calibrated per
    // face against the orderCorners output so the patch matches the
    // corner-cell orientation; faces failing the lattice checks fall back
    // to a single corner cell (previous behavior).
    void InitSurfaces(CDiscretization& disc, int backend)
    {
        const auto& common = disc.common;
        const auto& mesh   = disc.mesh;
        const auto& sol    = disc.sol;

        const Int nbf = common.meshsizes.nbf;
        const Int nf  = common.meshsizes.nf;
        const Int npf = common.grid.npf;
        const Int ncx = common.components.ncx;

        const int elemtype = common.grid.elemtype;
        if (nd == 2)            surf_k = 2;
        else if (elemtype == 0) surf_k = 3;
        else                    surf_k = 4;
        const int k = surf_k;
        const uint8_t ctype = (nd == 2) ? 3 : ((elemtype == 0) ? 5 : 9);
        const int isTri = (nd == 3 && elemtype == 0);
        const int isQuad = (nd == 3 && elemtype != 0);
        // Lattice degree from the face node count; -1 = unknown layout.
        const int p = surfLatticeDegree((int)npf, isTri, isQuad);

        surf_face2cell.assign((size_t)nf, -1);

        int nsel_faces = 0;
        Int maxnf = 0;
        for (Int j = 0; j < nbf; ++j) {
            Int ib = common.fblks[3*j+2];
            if (!common.qoiparams.isSaveBoundary(ib)) continue;
            Int f1 = common.fblks[3*j] - 1;
            Int f2 = common.fblks[3*j+1];
            nsel_faces += (int)(f2 - f1);
            maxnf = std::max(maxnf, f2 - f1);
        }
        surf_ncells = 0;
        surf_nnodes = 0;
        if (nsel_faces == 0) return;

        // Staging buffers sized once and reused (as in Init above, which
        // pre-sizes its geometry/topology vectors and fills them in loops):
        // xg for the largest selected block, plane for one face, plus one
        // device buffer on device backends (see gatherBoundaryNodes).
        std::vector<dstype> xg((size_t)npf*maxnf*ncx);
        std::vector<dstype> plane((size_t)npf*ncx);
        dstype* dbuf = nullptr;
        if (backend >= 2) TemplateMalloc(&dbuf, npf*maxnf*ncx, backend);

        surf_nodes.clear();
        surf_nodes.reserve((size_t)3 * nsel_faces * npf);
        int nfallback = 0;
        surf_cellconn.clear();
        surf_celloffsets.clear();
        surf_celltypes.clear();
        // Emit one linear sub-cell; lns holds master-face local node ids.
        auto emitCell = [&](int base, const int* lns, int n) {
            for (int c = 0; c < n; ++c)
                surf_cellconn.push_back(base + lns[c]);
            surf_celloffsets.push_back((int)surf_cellconn.size());
            surf_celltypes.push_back(ctype);
        };
        int ordinal = 0;
        for (Int j = 0; j < nbf; ++j) {
            Int ib = common.fblks[3*j+2];
            if (!common.qoiparams.isSaveBoundary(ib)) continue;
            Int f1 = common.fblks[3*j] - 1;
            Int f2 = common.fblks[3*j+1];
            Int nfblk = f2 - f1;
            if (nfblk == 0) continue;
            Int nn = npf*nfblk;
            gatherBoundaryNodes(xg.data(), sol.xdg, mesh.findxdg1, f1, nn, npf, ncx, backend, dbuf);
            for (Int ff = 0; ff < nfblk; ++ff) {
                // xg is component-major [dim][block-point] (see faceindex1);
                // gather this face's nodes into an interleaved plane buffer
                // for corner processing. Reading xg as [point][dim] directly
                // pairs x of one node with y of another (off-wall garbage).
                for (Int ln = 0; ln < npf; ++ln)
                    for (int d = 0; d < ncx; ++d)
                        plane[(size_t)ln*ncx + d] =
                            xg[(size_t)d*nn + (size_t)ff*npf + ln];
                int cs[4];
                if (p >= 1) {
                    // Corners are the reference-lattice boundary corners;
                    // orderCorners sorts them into physical CCW order.
                    surfLatticeBoundary(cs, p, isTri, isQuad);
                    orderCorners(plane.data(), (int)ncx, k, cs);
                } else {
                    // Unknown lattice degree: single cell over the first k nodes.
                    for (int ci = 0; ci < k; ++ci) cs[ci] = ci;
                }
                Int f = f1 + ff;
                // Emit all face nodes (DG-unique per face); values scatter 1:1.
                const int base = ordinal * (int)npf;
                for (Int ln = 0; ln < npf; ++ln) {
                    const dstype* pc = &plane.data()[(size_t)ln*ncx];
                    surf_nodes.push_back((float)pc[0]);
                    surf_nodes.push_back((ncx>1)?(float)pc[1]:0.0f);
                    surf_nodes.push_back((ncx>2)?(float)pc[2]:0.0f);
                }
                surf_face2cell[f] = ordinal;
                // Subdivide into linear sub-cells over the lattice, oriented
                // to match the corner-cell boundary orientation; fall back to
                // a single corner cell when the lattice checks fail.
                int mirror = 0, ok = 0;
                if (p >= 1 && !isTri && !isQuad) {
                    // Lines: corners are the lattice endpoints {0, p} by
                    // construction. Segment direction is irrelevant for VTK_LINE.
                    ok = 1;
                    for (int l = 0; l < p; ++l) {
                        const int seg[2] = {l, l + 1};
                        emitCell(base, seg, 2);
                    }
                } else if (p >= 1) {
                    const int ori = surfCalibrateOrientation(cs, k, p, isTri, isQuad);
                    if (ori != 0) {
                        ok = 1;
                        mirror = (ori < 0);
                        auto lat = [&](int i, int j) {
                            return surfLatticeIdx(i, j, p, isTri, isQuad, mirror);
                        };
                        if (isQuad) {
                            for (int jj = 0; jj < p; ++jj)
                                for (int ii = 0; ii < p; ++ii) {
                                    const int quad[4] = {lat(ii,jj), lat(ii+1,jj),
                                                         lat(ii+1,jj+1), lat(ii,jj+1)};
                                    emitCell(base, quad, 4);
                                }
                        } else {
                            for (int jj = 0; jj < p; ++jj)
                                for (int ii = 0; ii + jj < p; ++ii) {
                                    const int up[3] = {lat(ii,jj), lat(ii+1,jj),
                                                       lat(ii,jj+1)};
                                    emitCell(base, up, 3);
                                    if (ii + jj + 2 <= p) {
                                        const int dn[3] = {lat(ii+1,jj), lat(ii+1,jj+1),
                                                           lat(ii,jj+1)};
                                        emitCell(base, dn, 3);
                                    }
                                }
                        }
                    }
                }
                if (!ok) {
                    // Fallback: single cell with cs (lattice corners in
                    // physical CCW order, or the first-k nodes when the
                    // lattice degree is unknown).
                    emitCell(base, cs, k);
                    ++nfallback;
                }
                ++ordinal;
            }
        }
        if (dbuf) TemplateFree(dbuf, backend);
        surf_nnodes = (int)surf_nodes.size() / 3;
        surf_ncells = (int)surf_celloffsets.size();
        // Precompute the appended-data offsets once (volume Init pattern);
        // surfvtuwrite reuses them on every save.
        {
            const std::uint64_t obytesize = 8; // UInt64 header per block
            std::uint64_t soff = 0;
            auto s_add_off = [&](std::uint64_t payload_bytes) {
                std::uint64_t here = soff;
                soff += payload_bytes + obytesize;
                return here;
            };
            for (int s = 0; s < nsurfq; ++s)
                surf_scalar_offsets[s] = s_add_off(byte_count(surf_nnodes, sizeof(float)));
            surf_points_offset = s_add_off(byte_count(3, surf_nnodes, sizeof(float)));
            surf_conn_offset   = s_add_off(byte_count(surf_k, surf_ncells, sizeof(int32_t)));
            surf_offs_offset   = s_add_off(byte_count(surf_ncells, sizeof(int32_t)));
            surf_types_offset  = s_add_off(byte_count(surf_ncells, 1));
        }
        if (nfallback > 0 && disc.common.mpiRank == 0)
            printf("Surface visualization: %d of %d faces use single corner cells (lattice check failed).\n",
                   nfallback, nsel_faces);
    }

    void Init(const dstype* xcg, int nd_in, int np,
              const int* cgelcon, int npe, int ne,
              const int* telem,   int nce, int nverts_per_cell,
              int elemtype,
              const std::vector<std::string>& s_names,
              const std::vector<std::string>& v_names,
              const std::vector<std::string>& t_names,
              const std::vector<std::string>& surf_names)
    {
        nd = nd_in;
        ntc = nd * nd;           // <-- your requested change
        npoints = np;
        nve = nverts_per_cell;
        scalar_names = s_names;
        vector_names = v_names;
        tensor_names = t_names;
        surface_names = surf_names;

        if (!xcg)     error("Visualization: xcg pointer is null.");
        if (!cgelcon) error("Visualization: cgelcon pointer is null.");
        if (!telem)   error("Visualization: telem pointer is null.");
        if (nd != 2 && nd != 3) throw std::invalid_argument("nd must be 2 or 3.");
        if (npoints < 0 || ne <= 0 || npe <= 0 || nce <= 0 || nve <= 0)
            error("Visualization: invalid sizes (np, ne, npe, nce, nve).");

        // cgnodes padded to 3D
        cgnodes.assign(3 * npoints, 0.0f);
        for (int p = 0; p < npoints; ++p) {
            const int in_col  = p * nd;
            const int out_col = p * 3;
            cgnodes[out_col + 0] = (float) xcg[in_col + 0];
            cgnodes[out_col + 1] = (nd >= 2) ? (float) xcg[in_col + 1] : 0.0f;
            cgnodes[out_col + 2] = (nd == 3) ? (float) xcg[in_col + 2] : 0.0f;
        }

        // connectivity
        ncells = nce * ne;
        cgcells.assign(nve * ncells, 0);
        const int plane = nve * nce;
        for (int el = 0; el < ne; ++el)
            for (int j2 = 0; j2 < nce; ++j2)
                for (int i2 = 0; i2 < nve; ++i2) {
                    const int r   = telem[j2 + i2 * nce];
                    const int val = cgelcon[r + el * npe];
                    cgcells[i2 + j2 * nve + el * plane] = static_cast<int32_t>(val);
                }

        // offsets & types
        celloffsets.resize(ncells);
        for (int c = 0; c < ncells; ++c) celloffsets[c] = (c + 1) * nve;

        const int celltype = (nd == 2) ? ((elemtype == 0) ? 5 : 9)  // tri/quad
                                       : ((elemtype == 0) ? 10 : 12); // tet/hex
        celltypes.assign(ncells, static_cast<uint8_t>(celltype));

        // appended offsets
        const int fbytesize = 4; // Float32
        const int ibytesize = 4; // Int32
        const int obytesize = 8; // UInt64

        std::uint64_t offset = 0;
        auto add_off = [&](std::uint64_t payload_bytes) {
            std::uint64_t here = offset;
            offset += (std::uint64_t)payload_bytes + (std::uint64_t)obytesize;
            return here;
        };

        scalar_offsets.clear();
        vector_offsets.clear();
        tensor_offsets.clear();
        scalar_offsets.reserve((int)scalar_names.size());
        vector_offsets.reserve((int)vector_names.size());
        tensor_offsets.reserve((int)tensor_names.size());

        for (int i = 0; i < (int)scalar_names.size(); ++i)
            scalar_offsets.push_back(add_off(byte_count(npoints, fbytesize)));
        for (int i = 0; i < (int)vector_names.size(); ++i)
            vector_offsets.push_back(add_off(byte_count(3, npoints, fbytesize)));
        for (int i = 0; i < (int)tensor_names.size(); ++i)
            tensor_offsets.push_back(add_off(byte_count(ntc, npoints, fbytesize)));

        points_offset = add_off(byte_count(3, npoints, fbytesize));
        conn_offset   = add_off(byte_count(ncells, nve, ibytesize));
        offs_offset   = add_off(byte_count(ncells, ibytesize));
        types_offset  = add_off(byte_count(ncells, 1));
    }

    // endianness
    static const char* vtk_byte_order() {
        const uint16_t x = 0x0102;
        return (*reinterpret_cast<const uint8_t*>(&x) == 0x02)
            ? "LittleEndian" : "BigEndian";
    }

    // VTK appended block (UInt64 length + payload)
    static std::uint64_t byte_count(std::uint64_t n, std::uint64_t bytes_per_entry) {
        if (bytes_per_entry != 0 &&
            n > std::numeric_limits<std::uint64_t>::max() / bytes_per_entry)
            throw std::overflow_error("Visualization byte count overflow.");
        return n * bytes_per_entry;
    }

    static std::uint64_t byte_count(std::uint64_t n1, std::uint64_t n2,
                                    std::uint64_t bytes_per_entry) {
        return byte_count(byte_count(n1, n2), bytes_per_entry);
    }

    static void write_block(std::ofstream& s, const std::string& filename,
                            const std::string& block_name, const void* data,
                            std::uint64_t nbytes) {
        if (!data && nbytes > 0)
            throw std::runtime_error("Cannot write VTU block '" + block_name +
                                     "' to " + filename + ": null data pointer.");

        std::uint64_t nb = nbytes;
        errno = 0;
        s.write(reinterpret_cast<const char*>(&nb), sizeof(std::uint64_t));
        const char* ptr = reinterpret_cast<const char*>(data);
        while (s && nbytes > 0) {
            const std::uint64_t chunk64 =
                std::min<std::uint64_t>(nbytes,
                    static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max()));
            s.write(ptr, static_cast<std::streamsize>(chunk64));
            ptr += chunk64;
            nbytes -= chunk64;
        }
        if (!s) {
            std::string msg = "Error writing VTU appended block '" + block_name +
                              "' to " + filename + " (" + std::to_string(nb) +
                              " bytes)";
            if (errno != 0) msg += ": " + std::string(std::strerror(errno));
            throw std::runtime_error(msg);
        }
    }

    // Write N values as Float32 appended block from either float* or double*
    template <class T>
    static void write_as_float32(std::ofstream& os, const T* src, int N) {
        using U = std::remove_cv_t<T>;
        if constexpr (std::is_same_v<U, float>) {
            // Fast path: already Float32, write directly
            write_block(os, "<stream>", "float32", src, byte_count(N, sizeof(float)));
        } else {
            // U is double: convert once into a reusable buffer and write
            static thread_local std::vector<float> buf; // reuse to avoid re-allocs
            buf.resize(N);
            std::transform(src, src + N, buf.begin(),
                           [](double v){ return static_cast<float>(v); });
            write_block(os, "<stream>", "float32", buf.data(), byte_count(N, sizeof(float)));
        }
    }
    
    // utilities
    static std::string rank_tag(int rank, int width = 5) {
        std::ostringstream ss; ss << "_" << std::setw(width) << std::setfill('0') << rank; return ss.str();
    }
    static std::string step_tag(int k, int width = 6) {
        std::ostringstream ss; ss << std::setw(width) << std::setfill('0') << k; return ss.str();
    }

    // PVTU: advertise ntc components for tensors
    static void write_pvtu(const std::string& pvtu_basename_no_ext,
                           const std::vector<std::string>& piece_files,
                           const std::vector<std::string>& scalar_names,
                           const std::vector<std::string>& vector_names,
                           const std::vector<std::string>& tensor_names,
                           int ntc /* nd*nd */)
    {
        const std::string fname = pvtu_basename_no_ext + ".pvtu";
        std::ofstream os(fname, std::ios::binary);
        if (!os) throw std::runtime_error("Cannot open PVTU file: " + fname);

        os << "<?xml version=\"1.0\"?>\n";
        os << "<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" byte_order=\""
           << vtk_byte_order() << "\" header_type=\"UInt64\">\n";
        os << "  <PUnstructuredGrid GhostLevel=\"0\">\n";

        if (!scalar_names.empty() || !vector_names.empty() || !tensor_names.empty()) {
            os << "    <PPointData Scalars=\"scalars\">\n";
            for (const auto& s : scalar_names)
                os << "      <PDataArray type=\"Float32\" Name=\"" << s
                   << "\" NumberOfComponents=\"1\"/>\n";
            for (const auto& v : vector_names)
                os << "      <PDataArray type=\"Float32\" Name=\"" << v
                   << "\" NumberOfComponents=\"3\"/>\n";
            for (const auto& t : tensor_names)
                os << "      <PDataArray type=\"Float32\" Name=\"" << t
                   << "\" NumberOfComponents=\"" << ntc << "\"/>\n";
            os << "    </PPointData>\n";
        }

        os << "    <PPoints>\n";
        os << "      <PDataArray type=\"Float32\" NumberOfComponents=\"3\"/>\n";
        os << "    </PPoints>\n";

        os << "    <PCells>\n";
        os << "      <PDataArray type=\"Int32\"  Name=\"connectivity\"/>\n";
        os << "      <PDataArray type=\"Int32\"  Name=\"offsets\"/>\n";
        os << "      <PDataArray type=\"UInt8\"  Name=\"types\"/>\n";
        os << "    </PCells>\n";

        for (const auto& pf : piece_files)
            os << "    <Piece Source=\"" << pf << "\"/>\n";

        os << "  </PUnstructuredGrid>\n";
        os << "</VTKFile>\n";
        if (!os) throw std::runtime_error("Error writing PVTU file.");
    }
};
