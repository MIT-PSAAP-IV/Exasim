/*
    nonmatchinginterface.hpp

    Data structure of the non-conformal, non-matching interface solver
    (docs/nonmatching/nonmatching_multiphysics.tex, Section 4.5).

    FROZEN after phase 0: changing this header needs the review of both agents
    and the maintainer's approval.

    This file holds data only. The setup (donor search, distribution of the
    setup data, communication lists) is added by later phases in separate
    functions; the run-time kernels live in nonmatchingcoupling.hpp.

    Requirements: include after Common/common.h (uses ::dstype and ::Int as the
    default template arguments, like the other *structT types).

    Terminology (Section 3):
      direction 0: coupling on Gamma^1, eq. (1). Owners are the ranks of the
                   model with fileoffset == 0; donors are the ranks of the
                   model with fileoffset > 0.
      direction 1: coupling on Gamma^2, eq. (2). Roles exchanged.
      owner:  rank that owns an interface face F_l and assembles the coupling
              terms of that face into its trace residual / product (eq. (64)).
      donor:  rank that owns the donor faces and donor elements of some
              quadrature points of F_l, evaluates fint there and sends partial
              results to the owner (eqs. (63), (67), (68)).
      point:  a quadrature point x_gl of an owner face (one of ngf per face).
      slot:   a pair (owner rank, owner face) for which this donor holds points.
      pair:   a pair (slot, donor element); the unit of the coupling matrix.

    Conventions: zero-based indices; column-major flattened arrays; the leading
    (fastest) dimension is listed first in each layout comment. Per-point
    coordinate-like arrays are POINT-FIRST, e.g. px[p + npts*d] for
    "npts x nd", so they can be passed to FintDriver (component k of point i
    at k*N + i) and to the batched mkshape (pts[p + npoints*d]) without a
    repack. Basis arrays are NODE-FIRST, e.g. pphi[j + npe*p], as the
    interpolation kernels use them. Ranks are WORLD ranks (EXASIM_COMM_WORLD).

    One master element: in this version both models use the same master
    element. The setup (phase 3) must check that both models have equal nd,
    elemtype, porder, pgauss, npf, nfe and ngf, and stop with an error
    otherwise. Every size below is therefore common to owner and donor.
    Counts default to 0 and pointers to nullptr. Arrays are allocated with the
    Template* helpers on common.backend; host-only arrays say so explicitly.
    There is no destructor (same ownership model as meshstructT): a free
    function added in a later phase releases the arrays.
*/

#ifndef __NONMATCHINGINTERFACE_HPP
#define __NONMATCHINGINTERFACE_HPP

template <class T = ::dstype, class I = ::Int>
struct nonmatchinginterfaceT {
    using dstype = T; using Int = I;

    // ------------------------------------------------------------------
    // Identification and cached sizes
    // ------------------------------------------------------------------
    I direction = -1;   // 0: coupling on Gamma^1 (eq. (1)); 1: on Gamma^2 (eq. (2)); -1: unset
    I isowner = 0;      // 1 if this rank owns faces of this direction's interface
    I isdonor = 0;      // 1 if this rank serves donor points for this direction

    // Master-element sizes, equal on both models (one master element, see above).
    I nd = 0;           // spatial dimension
    I npe = 0;          // nodes per element
    I npf = 0;          // nodes per face (test functions: rows of Ri, Hi; donor face nodes:
                        //   columns of Hi; length of ppsi)
    I nfe = 0;          // faces per element (decodes mesh.intfaces on owners, columns of Hi on donors)
    I ngf = 0;          // quadrature points per face (range of pg)
    // Field sizes of the DONOR model (the model whose fields are transferred).
    I ncu = 0;          // ncu of the donor model: columns of Ki, Gi, Hi (eqs. (15), (20))
    I ncq = 0;          // ncq of the donor model (= ncu*nd, or 0 if q is absent, eq. (42))
    I ncu12 = 0;        // flux components = common.szinterfacefluxmap (rows of Ri, Hi);
                        // equal on owner and donor, see interfacefluxmap (eq. (64))

    // ------------------------------------------------------------------
    // OWNER side: interface faces of this rank (Section 4.5)
    // ------------------------------------------------------------------
    I nownfaces = 0;          // number of own interface faces (= common.couplingparams.nintfaces)
    I* ownface = nullptr;     // [nownfaces] index into mesh.intfaces; mesh.intfaces[ownface[l]]
                              //   = lf + nfe*e gives the element e (I_l) and local face lf (J_l)
    I* ownfaceglobal = nullptr; // [nownfaces] partition-independent face index
                              //   nfe*elempart[e] + lf (per model, Section 3 "Global face numbering")

    I nrecvnb = 0;            // number of donor ranks this owner receives from
    I* recvnb = nullptr;      // [nrecvnb] donor ranks, strictly increasing (assembly order, eq. (64))
    I* recvcount = nullptr;   // [nrecvnb] number of face blocks received from each donor
    I* recvface = nullptr;    // [sum(recvcount)] own-face index (0..nownfaces-1) of each received
                              //   block, grouped by donor in recvnb order, in list order L_pr
                              //   (increasing owner global face index, eq. (62))

    // ------------------------------------------------------------------
    // DONOR side: points of the other model's faces served by this rank
    // ------------------------------------------------------------------
    I npts = 0;               // number of donor points on this rank

    // Donor tuple (eq. (56)) and target, per point. Points are sorted by
    // slot, then by donor element (see slotptr, pairptr). The donor rank is
    // implicit: it is the rank that stores the point.
    I* pe = nullptr;          // [npts] local donor element e^2_gl (0..ne-1) containing the donor face
    I* pa = nullptr;          // [npts] local face number a^2_gl (0..nfe-1) of the donor face in pe
    I* pf = nullptr;          // [npts] global index f^2_gl = k^2_gl of the donor face (tie-break, eq. (3))
    I* pslot = nullptr;       // [npts] slot (0..nslots-1) the point contributes to
    I* pg = nullptr;          // [npts] quadrature index g (0..ngf-1) of the point on its owner face;
                              //   W^{l,r}(i,g) = pwJ[p]*psi_i(g), eq. (61), with the test functions
                              //   psi_i at the reference quadrature points taken from the common
                              //   master (master.shapfgt), since both models share it

    // Geometry received from the owner in setup (Step D6). Geometric
    // quantities are always those of the owner face (eq. (33)).
    T* px = nullptr;          // [npts x nd]  x^1_gl, px[p + npts*d]
    T* pnl = nullptr;         // [npts x nd]  owner outward normal n_1(x^1_gl), pnl[p + npts*d];
                              //   fint is evaluated with -pnl (eq. (33))
    T* pwJ = nullptr;         // [npts]       omega_g * J^1_gl (eqs. (37), (61))

    // Reference data computed in setup (Steps 1-3 of Section 2.4).
    T* pxi = nullptr;         // [npts x (nd-1)] xi^2_gl, closest point in the reference face,
                              //   pxi[p + npts*k], k < nd-1, eq. (21); always inside F_hat
    T* pzeta = nullptr;       // [npts x nd]  zeta^2_gl, reference coordinates of x^1_gl in pe,
                              //   pzeta[p + npts*d], eq. (28); NOT restricted to E_hat
    T* pphi = nullptr;        // [npe x npts] phi_j(zeta^2_gl), pphi[j + npe*p], eq. (30)
    T* pphibar = nullptr;     // [npe x npts] phi_j(zetabar^2_gl), zetabar = chi_a(xi),
                              //   pphibar[j + npe*p], eq. (30)
    T* ppsi = nullptr;        // [npf x npts] psi_j(xi^2_gl), face basis of the donor face,
                              //   ppsi[j + npf*p], eq. (30)
    T* pd = nullptr;          // [npts x nd]  d^2_gl = x^1_gl - xbar^2_gl, pd[p + npts*d], eq. (23)
    T* pdphibar = nullptr;    // [npe x nd x npts] d phi_j / d x_k at xbar^2_gl, index j + npe*(k + nd*p);
                              //   allocated only when ncq == 0 (eq. (42)), else nullptr

    // Slots: (owner rank, owner face) pairs for which this rank holds points.
    // Ordered by owner rank, then by owner global face index (list L_pr, eq. (62)).
    I nslots = 0;
    I* slotrank = nullptr;    // [nslots] owner rank of the slot
    I* slotface = nullptr;    // [nslots] owner global face index (ownfaceglobal on the owner)
    I* slotptr = nullptr;     // [nslots+1] points of slot s are slotptr[s] .. slotptr[s+1]-1

    // Pairs (slot, donor element): one coupling matrix block each (eq. (67)).
    // Ordered by slot, then by local donor element.
    I npairs = 0;
    I* pairslot = nullptr;    // [npairs] slot of the pair
    I* pairelem = nullptr;    // [npairs] local donor element of the pair
    I* pairptr = nullptr;     // [npairs+1] points of pair q are pairptr[q] .. pairptr[q+1]-1
                              //   (a sub-range of the points of slot pairslot[q])

    // Send lists (donor -> owner). The slots of owner sendnb[n] are the
    // contiguous range following the slots of sendnb[0..n-1]; this is L_pr.
    I nsendnb = 0;
    I* sendnb = nullptr;      // [nsendnb] owner ranks, strictly increasing
    I* sendcount = nullptr;   // [nsendnb] number of slots sent to each owner

    // ------------------------------------------------------------------
    // Work arrays (allocated by later phases)
    // ------------------------------------------------------------------
    T* Ri = nullptr;          // [ncu12 x npf x nslots] partial condensed residual per slot,
                              //   eq. (63); also the partial product (68) in the matrix-vector
                              //   product. Index n + ncu12*(i + npf*s), same order as res.Ri.
    T* Hi = nullptr;          // [(ncu12*npf) x (ncu*npf*nfe) x npairs] condensed pair matrix
                              //   Hbar^{le}_12, eq. (67). Index (n + ncu12*i) + ncu12*npf*(c +
                              //   ncu*(j + npf*a)) + ncu12*npf*ncu*npf*nfe*q, same block layout
                              //   as res.Hi (rows: flux component fastest, then test function;
                              //   columns: ncu fastest, then face node, then local face).
    T* sendbuf = nullptr;     // [ncu12 x npf x nslots] send buffer (slots in send-list order)
    T* recvbuf = nullptr;     // [ncu12 x npf x sum(recvcount)] receive buffer (owner side)
    I szRi = 0, szHi = 0, szsendbuf = 0, szrecvbuf = 0;   // allocated sizes of the work arrays
};

template <class T = ::dstype, class I = ::Int>
struct nonmatchingdataT {
    using dstype = T; using Int = I;

    I interfacetype = 0;                   // 0: conformal (existing path, default); 1: non-matching.
                                           //   Read from app.problem[34] (approved by the maintainer;
                                           //   phase 9 makes it a fixed slot with default 0).
    nonmatchinginterfaceT<T, I> dir[2];    // dir[0]: coupling on Gamma^1; dir[1]: on Gamma^2.
                                           //   A rank of model 1 is owner in dir[0] and donor in
                                           //   dir[1]; a rank of model 2 the reverse.
};

using nonmatchinginterface = nonmatchinginterfaceT<::dstype, ::Int>;
using nonmatchingdata = nonmatchingdataT<::dstype, ::Int>;

#endif
