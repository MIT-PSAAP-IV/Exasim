#include "exasim_paths.h"  // exasim_data_dir()
#include <cmath>
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>
#include <unordered_map>

// ---------------------------------------------------------------------------
// Surface corner machinery. The boundary cells of a tag are resolved on the
// trace nodes: each master face contributes npf nodes and exactly k of them
// are the geometric corners (2 endpoints in 2D, 3/4 vertices on a 3D face).
// Greedy farthest-point sampling returns exactly those k vertices for a face
// whose true vertices are its mutually far-nodes (true for simplex and box
// cells, including p>1 curved faces), and angle sorting gives a consistent
// cyclic winding. No assumption is made about the reference face numbering.
// ---------------------------------------------------------------------------

static inline dstype dist2to(const dstype* p, const dstype* q, int ncx)
{
    dstype s = 0.0;
    for (int d = 0; d < ncx; ++d) { dstype dt = p[d] - q[d]; s += dt*dt; }
    return s;
}

// plane: [npf x ncx] positions of one face's nodes. Returns the k corner
// local node indices (not yet cyclically ordered).
static inline void cornersOfFace(const dstype* plane, int npf, int ncx, int k, int* corners)
{
    dstype cx = 0, cy = 0, cz = 0;
    for (int n = 0; n < npf; ++n) {
        cx += plane[n*ncx+0];
        if (ncx > 1) cy += plane[n*ncx+1];
        if (ncx > 2) cz += plane[n*ncx+2];
    }
    cx /= npf; cy /= npf; cz /= npf;
    const dstype cg[3] = {cx, cy, cz};

    int* sel = corners;
    for (int ci = 0; ci < k; ++ci) {
        int best = -1; dstype bestd = -1;
        for (int n = 0; n < npf; ++n) {
            dstype dmin = dist2to(&plane[n*ncx], cg, ncx);
            for (int p = 0; p < ci && ci != 0; ++p)
                dmin = std::min(dmin, dist2to(&plane[n*ncx], &plane[sel[p]*ncx], ncx));
            if (dmin > bestd) { bestd = dmin; best = n; }
        }
        sel[ci] = best;
    }
}

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
    std::vector<std::string> surface_names;   // nsurfsca (surface scalar vis fields)

    // ------------------------------------------------------------------
    // Surface visualization: the boundary cells of the requested tag,
    // resolved on the trace nodes, with the value/eval plumbing so the
    // surface fields can be reconstructed at the enclosing CG corners.
    int   nsurfsca       = 0;    // number of surface scalar fields (>=0)
    int   surf_nnodes    = 0;    // unique surface corner nodes
    int   surf_ncells    = 0;    // boundary cells (2D edges / 3D tri-quads)
    int   surf_k         = 0;    // corners per cell (2, 3 or 4)
    int   surf_ibvis     = 0;    // requested boundary tag (0 => feature off)
    bool  surfvis_enabled = false;
    std::vector<float>   surf_nodes;      // [3 x surf_nnodes]
    std::vector<int32_t> surf_cellconn;   // [surf_k x surf_ncells]
    std::vector<uint8_t> surf_celllocal;  // [surf_k x surf_ncells] master-face node of each corner
    std::vector<int32_t> surf_cellface;   // [surf_ncells] local face index of each cell
    std::vector<int32_t> surf_face2cell;  // [nf] cell id owning the face, or -1
    std::vector<int32_t> surf_celloffsets;// [surf_ncells]
    std::vector<uint8_t> surf_celltypes;  // [surf_ncells]

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
            int nsurfsca= disc.common.qoiparams.nsurfsca;            
            int ibvis   = disc.common.qoiparams.ibvis;            
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

            std::vector<std::string> surfaces(nsurfsca);
            for (int i = 0; i < nsurfsca; i++) surfaces[i] = "Surface Field " + std::to_string(i);
            
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

            surfvis_enabled = (disc.common.qoiparams.saveParaview != 0) && (nsurfsca > 0) && (ibvis > 0);
            this->nsurfsca   = nsurfsca;
            surf_ibvis       = ibvis;
            if (surfvis_enabled) InitSurfaces(disc, backend);

            savemode = (disc.common.qoiparams.saveParaview != 0) && (nsca + nvec + nten > 0 || surfvis_enabled); 
        
            if (backend==2) { // GPU
            #ifdef HAVE_CUDA        
                cudaTemplateHostAlloc(&scafields, npoints*nsca, cudaHostAllocMapped); // zero copy
                cudaTemplateHostAlloc(&vecfields, 3*npoints*nvec, cudaHostAllocMapped); // zero copy
                cudaTemplateHostAlloc(&tenfields, ntc*npoints*nten, cudaHostAllocMapped); // zero copy
                cudaTemplateHostAlloc(&srffields, surf_nnodes*nsurfsca, cudaHostAllocMapped); // zero copy
                host_alloc_backend = 2;
            #endif                  
            }
            else if (backend==3) { // GPU
            #ifdef HAVE_HIP        
                hipTemplateHostMalloc(&scafields, npoints*nsca, hipHostMallocMapped); // zero copy
                hipTemplateHostMalloc(&vecfields, 3*npoints*nvec, hipHostMallocMapped); // zero copy
                hipTemplateHostMalloc(&tenfields, ntc*npoints*nten, hipHostMallocMapped); // zero copy
                hipTemplateHostMalloc(&srffields, surf_nnodes*nsurfsca, hipHostMallocMapped); // zero copy                
                host_alloc_backend = 3;
            #endif                  
            }    
            else { // CPU
                scafields = (float *) malloc(npoints*nsca*sizeof(float));
                vecfields = (float *) malloc(3*npoints*nvec*sizeof(float));
                tenfields = (float *) malloc(ntc*npoints*nten*sizeof(float));
                srffields = (float *) malloc(surf_nnodes*nsurfsca*sizeof(float));
                host_alloc_backend = 0;
            }
            
            for (int i = 0; i < npoints*nsca; i++) scafields[i] = 0.0;
            for (int i = 0; i < 3*npoints*nvec; i++) vecfields[i] = 0.0;
            for (int i = 0; i < ntc*npoints*nten; i++) tenfields[i] = 0.0;
            for (int i = 0; i < surf_nnodes*nsurfsca; i++) srffields[i] = 0.0;

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
        free_field(srffields); // in case you allocate this later
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

    // Surface writer (serial): scalar surface fields (+ surface normals) on a
    // boundary surface mesh.
    void surfwrite(const std::string& filename_no_ext,
                   const float* srffields_data,
                   const float* normals) const
    {
        const std::string filename = filename_no_ext + ".vtu";
        std::ofstream os(filename, std::ios::binary);
        if (!os) throw std::runtime_error("Cannot open output file: " + filename);

        std::uint64_t off = 0;
        auto add_off = [&](std::uint64_t nb) {
            std::uint64_t here = off;
            off += nb + (std::uint64_t)8;
            return here;
        };
        std::vector<std::uint64_t> foffs(nsurfsca);
        for (int s = 0; s < nsurfsca; ++s)
            foffs[s] = add_off(byte_count(surf_nnodes, sizeof(float)));
        std::uint64_t noff = add_off(byte_count(3, surf_nnodes, sizeof(float)));
        std::uint64_t poff = add_off(byte_count(3, surf_nnodes, sizeof(float)));
        std::uint64_t coff = add_off(byte_count(surf_k, surf_ncells, sizeof(int32_t)));
        std::uint64_t ooff = add_off(byte_count(surf_ncells, sizeof(int32_t)));
        std::uint64_t toff = add_off(byte_count(surf_ncells, 1));

        os << "<?xml version=\"1.0\"?>\n";
        os << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\""
           << vtk_byte_order() << "\" header_type=\"UInt64\">\n";
        os << "  <UnstructuredGrid>\n";
        os << "    <Piece NumberOfPoints=\"" << surf_nnodes
           << "\" NumberOfCells=\"" << surf_ncells << "\">\n";
        if (nsurfsca > 0 || normals != nullptr) {
            os << "      <PointData Scalars=\"surfscalars\">\n";
            for (int s = 0; s < nsurfsca; ++s)
                os << "        <DataArray type=\"Float32\" Name=\"" << surface_names[s]
                   << "\" Format=\"appended\" offset=\"" << foffs[s] << "\"/>\n";
            if (normals != nullptr)
                os << "        <DataArray type=\"Float32\" Name=\"Surface Normals\""
                   << " NumberOfComponents=\"3\" Format=\"appended\" offset=\"" << noff << "\"/>\n";
            os << "      </PointData>\n";
        }
        os << "      <Points>\n";
        os << "        <DataArray type=\"Float32\" Name=\"points\" NumberOfComponents=\"3\""
           << " Format=\"appended\" offset=\"" << poff << "\"/>\n";
        os << "      </Points>\n";
        os << "      <Cells>\n";
        os << "        <DataArray type=\"Int32\" Name=\"connectivity\" Format=\"appended\" offset=\"" << coff << "\"/>\n";
        os << "        <DataArray type=\"Int32\" Name=\"offsets\"     Format=\"appended\" offset=\"" << ooff << "\"/>\n";
        os << "        <DataArray type=\"UInt8\" Name=\"types\"       Format=\"appended\" offset=\"" << toff << "\"/>\n";
        os << "      </Cells>\n";
        os << "    </Piece>\n";
        os << "  </UnstructuredGrid>\n";
        os << "  <AppendedData encoding=\"raw\">\n";
        os << "   _";
        for (int s = 0; s < nsurfsca; ++s)
            write_block(os, filename, "surfscalar:" + surface_names[s],
                        &srffields_data[surf_nnodes * s],
                        byte_count(surf_nnodes, sizeof(float)));
        if (normals != nullptr)
            write_block(os, filename, "normals", normals,
                        byte_count(3, surf_nnodes, sizeof(float)));
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
    void surfwrite_parallel(const std::string& base_name,
                            int rank, int nranks,
                            const float* srffields_data,
                            const float* normals) const
    {
        surfwrite(base_name + rank_tag(rank), srffields_data, normals);
        if (rank == 0) {
            std::vector<std::string> pieces;
            pieces.reserve(nranks);
            for (int r = 0; r < nranks; ++r) {
                const std::filesystem::path piece = base_name + rank_tag(r) + ".vtu";
                pieces.push_back(piece.filename().generic_string());
            }
            std::vector<std::string> norms;
            if (normals != nullptr) norms.push_back("Surface Normals");
            write_pvtu(base_name, pieces, surface_names, norms, {}, 3);
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
    // Build the boundary surface mesh (points/topology) for the requested tag
    // and the per-corner mappings used by SaveSurfaces to scatter the surface
    // scalar fields from the trace nodes onto the surface corner nodes.
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

        // Quantized CG-node lookup
        const double q = 1e6;
        auto quant = [&](const dstype* p) -> std::uint64_t {
            unsigned long long a = (unsigned long long) llround((double)p[0]*q);
            unsigned long long b = (unsigned long long) llround(((ncx>1)?(double)p[1]:0.0)*q);
            unsigned long long c = (unsigned long long) llround(((ncx>2)?(double)p[2]:0.0)*q);
            return (a*73856093ULL) ^ (b*19349663ULL) ^ (c*83492791ULL);
        };
        std::unordered_map<std::uint64_t,int> cgmap;
        cgmap.reserve(2 * npoints);
        for (int p = 0; p < npoints; ++p) cgmap.emplace(quant(&sol.xcg[p*ncx]), p);
        auto cgfind = [&](const dstype* p) -> int {
            dstype bestd = std::numeric_limits<dstype>::infinity();
            int best = -1;
            for (int cg = 0; cg < npoints; ++cg) {
                dstype d = dist2to(p, &sol.xcg[cg*ncx], (int)ncx);
                if (d < bestd) { bestd = d; best = cg; }
            }
            if (best < 0) best = 0;
            return best;
        };
        (void)cgmap; // table reserved for future fast-path; nearest scan is exact

        surf_face2cell.assign((size_t)nf, -1);

        int ncell = 0;
        for (Int j = 0; j < nbf; ++j) {
            Int ib = common.fblks[3*j+2];
            if (surf_ibvis > 0 && ib != surf_ibvis) continue;
            Int f1 = common.fblks[3*j] - 1;
            Int f2 = common.fblks[3*j+1];
            ncell += (int)(f2 - f1);
        }
        surf_ncells = ncell;
        if (ncell == 0) return;

        surf_cellconn.assign((size_t)k*ncell, 0);
        surf_celllocal.assign((size_t)k*ncell, 0);
        surf_cellface.assign(ncell, 0);
        surf_celloffsets.resize(ncell);
        surf_celltypes.assign(ncell, ctype);
        for (int c = 0; c < ncell; ++c) surf_celloffsets[c] = (c + 1) * k;

        int cell = 0;
        for (Int j = 0; j < nbf; ++j) {
            Int ib = common.fblks[3*j+2];
            if (surf_ibvis > 0 && ib != surf_ibvis) continue;
            Int f1 = common.fblks[3*j] - 1;
            Int f2 = common.fblks[3*j+1];
            Int nfblk = f2 - f1;
            if (nfblk == 0) continue;
            Int nn = npf*nfblk;
            std::vector<dstype> xg((size_t)nn*ncx, 0.0);
            if (backend >= 2) {
                dstype* d = nullptr;
                TemplateMalloc(&d, nn*ncx, backend);
                GetArrayAtIndex(d, sol.xdg, &mesh.findxdg1[npf*ncx*f1], nn*ncx);
                TemplateCopytoHost(xg.data(), d, nn*ncx, backend);
                CPUFREE(d);
            } else {
                GetArrayAtIndex(xg.data(), sol.xdg, &mesh.findxdg1[npf*ncx*f1], nn*ncx);
            }
            for (Int ff = 0; ff < nfblk; ++ff) {
                const dstype* plane = &xg[(size_t)ff*npf*ncx];
                int cs[4];
                cornersOfFace(plane, (int)npf, (int)ncx, k, cs);
                orderCorners(plane, (int)ncx, k, cs);
                Int f = f1 + ff;
                surf_cellface[cell] = (int32_t)f;
                surf_face2cell[f] = cell;
                for (int ci = 0; ci < k; ++ci) {
                    int ln = cs[ci];
                    int cg = cgfind(&plane[(size_t)ln*ncx]);
                    surf_cellconn[(size_t)cell*k + ci] = cg;
                    surf_celllocal[(size_t)cell*k + ci] = (uint8_t)ln;
                }
                ++cell;
            }
        }

        // Deduplicate the per-cell CG ids into a compact surface node set.
        std::unordered_map<int,int> cgtosurf;
        cgtosurf.reserve((size_t)k*ncell);
        surf_nodes.clear();
        for (int c = 0; c < ncell; ++c) {
            for (int ci = 0; ci < k; ++ci) {
                int cg = surf_cellconn[(size_t)c*k + ci];
                auto it = cgtosurf.find(cg);
                int s;
                if (it == cgtosurf.end()) {
                    s = (int)surf_nodes.size() / 3;
                    cgtosurf.emplace(cg, s);
                    const dstype* p0 = &sol.xcg[(size_t)cg*ncx];
                    surf_nodes.push_back((float)p0[0]);
                    surf_nodes.push_back((ncx>1)?(float)p0[1]:0.0f);
                    surf_nodes.push_back((ncx>2)?(float)p0[2]:0.0f);
                } else {
                    s = it->second;
                }
                surf_cellconn[(size_t)c*k + ci] = s;
            }
        }
        surf_nnodes = (int)surf_nodes.size() / 3;
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
