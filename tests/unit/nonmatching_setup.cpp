// Phase 3 of the non-matching interface solver: setup distribution and
// communication lists (design Steps D6-D7, eqs. (61)-(62)).  Reuse the
// synthetic mesh fixture from the Phase 2 test in this separate executable.
#define main nonmatching_donorsearch_embedded_main
#include "nonmatching_donorsearch.cpp"
#undef main
void FintDriver(...); // discarded AbiAdapter branch; this test instantiates only exchange helpers
#include "../../backend/Discretization/nonmatchingcoupling.hpp"

namespace {

using SetupMap = std::map<Key, std::vector<double>>;

template <class T>
void append_cache_bytes(std::vector<unsigned char>& out, const T* data, std::size_t count)
{
    if (count == 0) return;
    check(data != nullptr, "phase6: cache snapshot pointer");
    const unsigned char* first = reinterpret_cast<const unsigned char*>(data);
    out.insert(out.end(), first, first + count * sizeof(T));
}

template <class T>
void append_cache_value(std::vector<unsigned char>& out, const T& value)
{
    append_cache_bytes(out, &value, 1);
}

std::vector<unsigned char> cache_snapshot(const nonmatchingdata& nm)
{
    std::vector<unsigned char> out;
    append_cache_value(out, nm.interfacetype);
    for (int direction = 0; direction < 2; ++direction) {
        const nonmatchinginterface& d = nm.dir[direction];
        const Int scalars[] = {d.direction, d.isowner, d.isdonor, d.nd, d.npe, d.npf,
            d.nfe, d.ngf, d.ncu, d.ncq, d.ncu12, d.nownfaces, d.nrecvnb, d.npts,
            d.nslots, d.npairs, d.nsendnb, d.szRi, d.szHi, d.szsendbuf, d.szrecvbuf};
        append_cache_bytes(out, scalars, sizeof(scalars) / sizeof(scalars[0]));
        Int nrecv = 0;
        for (Int q = 0; q < d.nrecvnb; ++q) nrecv += d.recvcount[q];
        append_cache_bytes(out, d.ownface, d.nownfaces);
        append_cache_bytes(out, d.ownfaceglobal, d.nownfaces);
        append_cache_bytes(out, d.recvnb, d.nrecvnb);
        append_cache_bytes(out, d.recvcount, d.nrecvnb);
        append_cache_bytes(out, d.recvface, nrecv);
        append_cache_bytes(out, d.pe, d.npts); append_cache_bytes(out, d.pa, d.npts);
        append_cache_bytes(out, d.pf, d.npts); append_cache_bytes(out, d.pslot, d.npts);
        append_cache_bytes(out, d.pg, d.npts);
        append_cache_bytes(out, d.px, static_cast<std::size_t>(d.npts) * d.nd);
        append_cache_bytes(out, d.pnl, static_cast<std::size_t>(d.npts) * d.nd);
        append_cache_bytes(out, d.pwJ, d.npts);
        append_cache_bytes(out, d.pxi, static_cast<std::size_t>(d.npts) * (d.nd - 1));
        append_cache_bytes(out, d.pzeta, static_cast<std::size_t>(d.npts) * d.nd);
        append_cache_bytes(out, d.pphi, static_cast<std::size_t>(d.npts) * d.npe);
        append_cache_bytes(out, d.pphibar, static_cast<std::size_t>(d.npts) * d.npe);
        append_cache_bytes(out, d.ppsi, static_cast<std::size_t>(d.npts) * d.npf);
        append_cache_bytes(out, d.pd, static_cast<std::size_t>(d.npts) * d.nd);
        if (d.ncq == 0)
            append_cache_bytes(out, d.pdphibar,
                    static_cast<std::size_t>(d.npts) * d.nd * d.npe);
        append_cache_bytes(out, d.slotrank, d.nslots); append_cache_bytes(out, d.slotface, d.nslots);
        append_cache_bytes(out, d.slotptr, d.nslots > 0 ? d.nslots + 1 : 0);
        append_cache_bytes(out, d.pairslot, d.npairs); append_cache_bytes(out, d.pairelem, d.npairs);
        append_cache_bytes(out, d.pairptr, d.npairs > 0 ? d.npairs + 1 : 0);
        append_cache_bytes(out, d.sendnb, d.nsendnb); append_cache_bytes(out, d.sendcount, d.nsendnb);
        append_cache_bytes(out, d.Ri, d.szRi); append_cache_bytes(out, d.Hi, d.szHi);
        append_cache_bytes(out, d.sendbuf, d.szsendbuf); append_cache_bytes(out, d.recvbuf, d.szrecvbuf);
    }
    return out;
}

Int coupling_request_count(const nonmatchingdata& nm)
{
    Int count = 0;
    for (int direction = 0; direction < 2; ++direction) {
        const nonmatchinginterface& d = nm.dir[direction];
        if (d.isdonor)
            for (Int q = 0; q < d.nsendnb; ++q) count += d.sendcount[q] > 0;
        if (d.isowner)
            for (Int q = 0; q < d.nrecvnb; ++q) count += d.recvcount[q] > 0;
    }
    return count;
}

int setup_record_size(const MasterData& m)
{
    return 16 + 4 * m.npe + m.npf;
}

void pack_setup_records(std::vector<double>& out, const nonmatchinginterface& donor,
        const RankStructs& rs, const MasterData& m)
{
    const int nr = setup_record_size(m);
    for (Int p = 0; p < donor.npts; p++) {
        const Int s = donor.pslot[p];
        std::vector<double> v(static_cast<std::size_t>(nr), 0.0);
        int q = 0;
        v[q++] = donor.direction;
        v[q++] = donor.slotface[s];
        v[q++] = donor.pg[p];
        v[q++] = rs.elempart[donor.pe[p]];
        v[q++] = donor.pf[p];
        v[q++] = donor.pa[p];
        for (Int d = 0; d < donor.nd; d++) v[q++] = donor.px[p + donor.npts * d];
        for (Int d = 0; d < donor.nd; d++) v[q++] = donor.pnl[p + donor.npts * d];
        v[q++] = donor.pwJ[p];
        v[q++] = donor.pxi[p];
        for (Int d = 0; d < donor.nd; d++) v[q++] = donor.pzeta[p + donor.npts * d];
        for (Int j = 0; j < donor.npe; j++) v[q++] = donor.pphi[j + donor.npe * p];
        for (Int j = 0; j < donor.npe; j++) v[q++] = donor.pphibar[j + donor.npe * p];
        for (Int j = 0; j < donor.npf; j++) v[q++] = donor.ppsi[j + donor.npf * p];
        for (Int d = 0; d < donor.nd; d++) v[q++] = donor.pd[p + donor.npts * d];
        for (Int d = 0; d < donor.nd; d++)
            for (Int j = 0; j < donor.npe; j++)
                v[q++] = donor.pdphibar[j + donor.npe * (d + donor.nd * p)];
        check(q == nr, "phase3: packed record size");
        out.insert(out.end(), v.begin(), v.end());
    }
}

SetupMap setup_to_map(const std::vector<double>& packed, int nr)
{
    SetupMap result;
    for (std::size_t i = 0; i + static_cast<std::size_t>(nr) <= packed.size(); i += nr) {
        const Key key((long(packed[i]) << 40) + long(packed[i + 1]), long(packed[i + 2]));
        check(result.emplace(key, std::vector<double>(packed.begin() + i, packed.begin() + i + nr)).second,
              "phase3: every owner point served exactly once");
    }
    check(packed.size() % static_cast<std::size_t>(nr) == 0, "phase3: complete packed records");
    return result;
}

void check_local_setup(const nonmatchingdata& nm, const DonorSearchResult<dstype, Int>& res,
        const RankStructs& rs, const MasterData& m, int model, bool matching, const std::string& tag)
{
    const nonmatchinginterface& owner = nm.dir[model];
    const nonmatchinginterface& donor = nm.dir[1 - model];
    SimplexNodalBasis2D simplexBasis;
    const SimplexNodalBasis2D* simplexBasisPtr = nullptr;
    if (UsesPolynomialSimplexBasis(m.nd, m.elemtype)) {
        simplexBasis.initialize(m.xpe.data(), m.npe, m.porder);
        simplexBasisPtr = &simplexBasis;
    }
    check(owner.nownfaces == res.nfaces && res.npts == res.nfaces * res.ngf,
          tag + ": all owner points searched");
    check(donor.direction == 1 - model, tag + ": donor direction");
    check(donor.npts == 0 || donor.isdonor == 1, tag + ": donor role");
    check(donor.nslots == 0 || donor.slotptr[donor.nslots] == donor.npts, tag + ": slot partition");
    check(donor.npairs == 0 || donor.pairptr[donor.npairs] == donor.npts, tag + ": pair partition");
    check(donor.szRi == donor.ncu12 * donor.npf * donor.nslots, tag + ": Ri size");
    check(donor.szsendbuf == donor.szRi, tag + ": send buffer size");
    check(donor.szHi == donor.ncu12 * donor.npf * donor.ncu * donor.npf * donor.nfe * donor.npairs,
          tag + ": Hi size");
    check((donor.ncq == 0) == (donor.pdphibar != nullptr || donor.npts == 0), tag + ": q-free gradients");

    long recvSlots = 0;
    std::vector<int> appearances(static_cast<std::size_t>(owner.nownfaces), 0);
    for (Int n = 0; n < owner.nrecvnb; n++) {
        if (n > 0) check(owner.recvnb[n] > owner.recvnb[n - 1], tag + ": recv rank order");
        for (Int j = 0; j < owner.recvcount[n]; j++) appearances[owner.recvface[recvSlots + j]]++;
        recvSlots += owner.recvcount[n];
    }
    check(owner.szrecvbuf == owner.ncu12 * owner.npf * recvSlots, tag + ": receive buffer size");
    if (matching)
        for (int count : appearances) check(count == 1, tag + ": matching |R_l| = 1");

    for (Int s = 0; s < donor.nslots; s++) {
        if (s > 0)
            check(std::tie(donor.slotrank[s - 1], donor.slotface[s - 1]) <
                  std::tie(donor.slotrank[s], donor.slotface[s]), tag + ": slot order");
        for (Int p = donor.slotptr[s]; p < donor.slotptr[s + 1]; p++) check(donor.pslot[p] == s, tag + ": pslot");
    }
    for (Int q = 0; q < donor.npairs; q++) {
        if (q > 0)
            check(std::tie(donor.pairslot[q - 1], donor.pairelem[q - 1]) <
                  std::tie(donor.pairslot[q], donor.pairelem[q]), tag + ": pair order");
        for (Int p = donor.pairptr[q]; p < donor.pairptr[q + 1]; p++)
            check(donor.pslot[p] == donor.pairslot[q] && donor.pe[p] == donor.pairelem[q], tag + ": pair contents");
    }

    for (Int p = 0; p < donor.npts; p++) {
        check(donor.pg[p] >= 0 && donor.pg[p] < donor.ngf, tag + ": quadrature index");
        check(donor.pwJ[p] > 0 && std::isfinite(donor.pwJ[p]), tag + ": positive weighted Jacobian");
        check(std::abs(std::hypot(donor.pnl[p], donor.pnl[p + donor.npts]) - 1.0) < 2e-13,
              tag + ": unit owner normal");
        std::vector<dstype> element(static_cast<std::size_t>(m.npe * m.nd));
        for (int d = 0; d < m.nd; d++)
            for (int j = 0; j < m.npe; j++)
                element[j + m.npe * d] = rs.xdg[j + m.npe * d + m.npe * m.nd * donor.pe[p]];
        dstype xi[1] = {donor.pxi[p]}, zbar[2], psi[16];
        FaceToElementReference(zbar, psi, xi, m.xpe.data(), m.xpf.data(), m.perm.data(), donor.pa[p],
                m.nd, m.npe, m.npf, m.elemtype, m.porder);
        dstype mapped[3], jac[9], shape[64], dshape[128];
        const dstype zeta[2] = {donor.pzeta[p], donor.pzeta[p + donor.npts]};
        MapNonmatchingReferenceToPhysical(
                mapped, jac, shape, dshape, zeta, element.data(), m.xpe.data(),
                m.nd, m.npe, m.elemtype, m.porder, simplexBasisPtr);
        const dstype xpoint[2] = {donor.px[p], donor.px[p + donor.npts]};
        const dstype scale = inverse_map_scale(xpoint, element.data(), Int(m.nd), Int(m.npe));
        const dstype mappedResidual = std::hypot(mapped[0] - xpoint[0], mapped[1] - xpoint[1]);
        const dstype exactTolerance = 128 * std::numeric_limits<dstype>::epsilon() * scale;
        const dstype reportedMaximum = res.maxInverseResidual[1 - model];
        check(std::isfinite(mappedResidual) &&
              mappedResidual <= std::max(exactTolerance, reportedMaximum * (1 + 8e-14)),
              tag + ": zeta is exact or the reported projected least-squares point");
        dstype sumPhi = 0, sumPhiBar = 0;
        for (int j = 0; j < m.npe; j++) { sumPhi += donor.pphi[j + m.npe * p]; sumPhiBar += donor.pphibar[j + m.npe * p]; }
        check(std::abs(sumPhi - 1) < 2e-12 && std::abs(sumPhiBar - 1) < 2e-12, tag + ": basis partition of unity");
        if (m.elemtype == 0) {
            dstype interpolated = 0.0;
            for (int j = 0; j < m.npe; ++j) {
                const dstype nodal = 3.0 + 2.0 * m.xpe[j] - 5.0 * m.xpe[j + m.npe];
                interpolated += donor.pphi[j + m.npe * p] * nodal;
            }
            const dstype exact = 3.0 + 2.0 * zeta[0] - 5.0 * zeta[1];
            check(std::abs(interpolated - exact) < 2e-11,
                  tag + ": stored triangle pphi reproduces linear polynomials");
        }
        for (int j = 0; j < m.npf; j++) check(std::abs(donor.ppsi[j + m.npf * p] - psi[j]) < 2e-13, tag + ": psi");
        if (matching) {
            check(std::hypot(donor.pd[p], donor.pd[p + donor.npts]) <= exactTolerance,
                  tag + ": matching d = 0 to scale-aware roundoff");
            check(std::hypot(zeta[0] - zbar[0], zeta[1] - zbar[1]) < 2e-11, tag + ": matching zeta = zetabar");
        }
    }
}

SetupMap run_setup_partition(const std::string& name, const ModelMesh& m1, const ModelMesh& m2,
        MasterData& m, int n, bool matching, bool requireExact)
{
    MPI_Comm sub = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < n ? 0 : MPI_UNDEFINED, worldRank, &sub);
    std::vector<double> gathered;
    if (sub != MPI_COMM_NULL) {
        EXASIM_COMM_WORLD = sub;
        const int k = n / 2, model = worldRank < k ? 0 : 1;
        RankStructs rs;
        build_rank_structs(rs, model == 0 ? m1 : m2, m, worldRank - model * k, k, 1);
        nonmatchingdata nm;
        std::unique_ptr<exasim::nonmatching::NonmatchingRuntime<dstype, Int>> runtime;
        commonstruct requestCommon;
        requestCommon.components.nc = 3;
        requestCommon.components.ncu = 1;
        requestCommon.components.ncq = 2;
        requestCommon.components.ncx = 2;
        requestCommon.components.nco = 0;
        requestCommon.components.ncw = 0;
        requestCommon.meshsizes.ne = rs.mesh.ndims[1];
        requestCommon.meshsizes.ne1 = rs.mesh.ndims[1];
        Int interfaceMap = 0;
        rs.app.interfacefluxmap = &interfaceMap;
        std::vector<Int> intfaces;
        for (Int e = 0; e < rs.mesh.ndims[1]; ++e)
            for (Int lf = 0; lf < m.nfe; ++lf)
                if (rs.mesh.bf[lf + m.nfe * e] == 2)
                    intfaces.push_back(lf + m.nfe * e);
        rs.mesh.intfaces = intfaces.data();
        rs.mesh.szintfaces = static_cast<Int>(intfaces.size());
        DonorSearchResult<dstype, Int> res;
        const bool ran = rebuild_nonmatching_interface(rs.app, rs.master, rs.mesh, rs.sol, requestCommon,
                static_cast<Int>(n), static_cast<Int>(worldRank), static_cast<Int>(model * k),
                nm, runtime, Int(0), &res);
        const std::string tag = name + " (" + std::to_string(k) + " per model)";
        check(ran, tag + ": full setup ran");
        check_local_setup(nm, res, rs, m, model, matching, tag);

        // Phase 6: setup is a cache rebuild, not a one-shot initializer.  Move
        // both meshes rigidly, rebuild without an explicit free, disable type 1
        // to clear the cache, then restore the original mesh and rebuild again.
        // Running this on the smallest matching partition keeps earlier
        // partition/checksum coverage unchanged.
        if (name == "matching" && n == 2) {
            std::vector<double> original;
            pack_setup_records(original, nm.dir[1 - model], rs, m);
            const std::vector<unsigned char> originalCache = cache_snapshot(nm);
            const std::vector<dstype> originalXdg = rs.xdg;
            for (Int e = 0; e < rs.mesh.ndims[1]; ++e)
                for (Int j = 0; j < m.npe; ++j) {
                    rs.sol.xdg[j + m.npe * (0 + m.nd * e)] += 0.125;
                    rs.sol.xdg[j + m.npe * (1 + m.nd * e)] -= 0.25;
                }
            DonorSearchResult<dstype, Int> movedRes;
            check(rebuild_nonmatching_interface(rs.app, rs.master, rs.mesh, rs.sol, requestCommon,
                      static_cast<Int>(n), static_cast<Int>(worldRank),
                      static_cast<Int>(model * k), nm, runtime, Int(0), &movedRes),
                  tag + ": mesh-motion rebuild ran");
            check_local_setup(nm, movedRes, rs, m, model, true, tag + " moved");
            std::vector<double> moved;
            pack_setup_records(moved, nm.dir[1 - model], rs, m);
            check(moved.size() == original.size(), tag + ": rebuild record count unchanged");
            for (std::size_t i = 0; i + static_cast<std::size_t>(setup_record_size(m)) <= moved.size();
                 i += static_cast<std::size_t>(setup_record_size(m))) {
                check(std::abs((moved[i + 6] - original[i + 6]) - 0.125) < 2e-13,
                      tag + ": rebuilt owner x coordinate");
                check(std::abs((moved[i + 7] - original[i + 7]) + 0.25) < 2e-13,
                      tag + ": rebuilt owner y coordinate");
            }

            rs.app.problem[34] = 0;
            check(!rebuild_nonmatching_interface(rs.app, rs.master, rs.mesh, rs.sol, requestCommon,
                       static_cast<Int>(n), static_cast<Int>(worldRank),
                       static_cast<Int>(model * k), nm, runtime, Int(0),
                       static_cast<DonorSearchResult<dstype, Int>*>(nullptr)),
                  tag + ": inactive rebuild returns false");
            check(nm.interfacetype == 0, tag + ": inactive rebuild resets interface type");
            check(!runtime, tag + ": inactive rebuild resets runtime cache");
            for (int direction = 0; direction < 2; ++direction) {
                const nonmatchinginterface& d = nm.dir[direction];
                check(d.direction == -1 && d.npts == 0 && d.nslots == 0 && d.npairs == 0,
                      tag + ": inactive rebuild clears sizes");
                check(d.pe == nullptr && d.pphi == nullptr && d.Ri == nullptr && d.Hi == nullptr,
                      tag + ": inactive rebuild clears allocations");
            }

            rs.xdg = originalXdg;
            rs.sol.xdg = rs.xdg.data();
            rs.app.problem[34] = 1;
            check(rebuild_nonmatching_interface(rs.app, rs.master, rs.mesh, rs.sol, requestCommon,
                      static_cast<Int>(n), static_cast<Int>(worldRank),
                      static_cast<Int>(model * k), nm, runtime, Int(0), &res),
                  tag + ": original-geometry rebuild ran");
            check_local_setup(nm, res, rs, m, model, true, tag + " rebuilt");
            std::vector<double> rebuilt;
            pack_setup_records(rebuilt, nm.dir[1 - model], rs, m);
            check(rebuilt == original, tag + ": original cache reproduced bitwise");
            check(cache_snapshot(nm) == originalCache,
                  tag + ": complete owner/donor cache reproduced bitwise");
        }

        // Phase 6 Revision 1: with two ranks per model, move only model 2 so
        // interface strips overlap different ranks.  The atomic rebuild must
        // resize requests before the real compact exchange and owner scatter.
        if (name == "matching" && n == 4) {
            const std::vector<dstype> originalXdg = rs.xdg;
            const std::vector<unsigned char> originalCache = cache_snapshot(nm);
            const Int oldLocal = coupling_request_count(nm);
            Int oldTotal = 0;
            MPI_Allreduce(&oldLocal, &oldTotal, 1, mpi_type<Int>(), MPI_SUM, sub);
            if (model == 1)
                for (Int e = 0; e < rs.mesh.ndims[1]; ++e)
                    for (Int j = 0; j < m.npe; ++j)
                        rs.sol.xdg[j + m.npe * (1 + m.nd * e)] += 0.2;

            DonorSearchResult<dstype, Int> shiftedRes;
            check(rebuild_nonmatching_interface(rs.app, rs.master, rs.mesh, rs.sol, requestCommon,
                      static_cast<Int>(n), static_cast<Int>(worldRank),
                      static_cast<Int>(model * k), nm, runtime, Int(0), &shiftedRes),
                  tag + ": changed-neighbor rebuild ran");
            const Int newLocal = coupling_request_count(nm);
            Int newTotal = 0;
            MPI_Allreduce(&newLocal, &newTotal, 1, mpi_type<Int>(), MPI_SUM, sub);
            check(newTotal > oldTotal, tag + ": rebuild increased coupling-neighbor requests");

            check(runtime != nullptr, tag + ": changed-neighbor rebuild publishes runtime cache");
            for (int direction = 0; direction < 2; ++direction) {
                const nonmatchinginterface& hostDirection = nm.dir[direction];
                auto& d = runtime->dir[direction];
                const dstype value = static_cast<dstype>(worldRank + 1 + 100 * direction);
                if (d.view.isdonor) {
                    Int slotOffset = 0;
                    for (std::size_t q = 0; q < d.sendnb.size(); ++q)
                        for (Int entry = 0; entry < d.sendcount[q]; ++entry, ++slotOffset)
                            check(hostDirection.slotrank[slotOffset] == d.sendnb[q],
                                  tag + ": direct Ri segments follow send-neighbor order");
                    check(slotOffset == d.view.nslots,
                          tag + ": direct Ri segments cover every donor slot");
                    std::fill(d.Ri.data, d.Ri.data + d.Ri.size, value);
                }
            }
            Int requestCounter = 0;
            begin_coupling_slot_exchange(*runtime, requestCommon, requestCounter);
            check(requestCounter == newLocal, tag + ": rebuild supplied exact coupling request count");
            MPI_Waitall(requestCounter, requestCommon.requests, requestCommon.statuses);
            for (int direction = 0; direction < 2; ++direction) {
                const auto& owner = runtime->dir[direction];
                const auto& view = owner.view;
                if (!view.isowner) continue;
                Int block = 0;
                for (std::size_t q = 0; q < owner.recvnb.size(); ++q)
                    for (Int entry = 0; entry < owner.recvcount[q]; ++entry, ++block)
                        for (Int i = 0; i < view.ncu12 * view.npf; ++i)
                            check(owner.recvbuf.data[i + view.ncu12 * view.npf * block] ==
                                      static_cast<dstype>(owner.recvnb[q] + 1 + 100 * direction),
                                  tag + ": changed-neighbor exchange payload");
            }
            std::vector<dstype> assembled(static_cast<std::size_t>(m.npf) * m.nfe * rs.mesh.ndims[1], 0);
            assemble_coupling_slot_owner(assembled.data(), *runtime, rs.mesh, rs.app, requestCommon);
            dstype localMagnitude = 0, globalMagnitude = 0;
            for (dstype value : assembled) localMagnitude += std::abs(value);
            MPI_Allreduce(&localMagnitude, &globalMagnitude, 1, mpi_type<dstype>(), MPI_SUM, sub);
            check(globalMagnitude > 0, tag + ": changed-neighbor owner assembly is nonzero");

            rs.xdg = originalXdg;
            rs.sol.xdg = rs.xdg.data();
            check(rebuild_nonmatching_interface(rs.app, rs.master, rs.mesh, rs.sol, requestCommon,
                      static_cast<Int>(n), static_cast<Int>(worldRank),
                      static_cast<Int>(model * k), nm, runtime, Int(0), &res),
                  tag + ": changed-neighbor geometry restored");
            check(cache_snapshot(nm) == originalCache,
                  tag + ": changed-neighbor cache restored bitwise");
        }
        Int localProjected = res.projectedExactInversePoints[0] +
                             res.projectedExactInversePoints[1] +
                             res.projectedInexactInversePoints[0] +
                             res.projectedInexactInversePoints[1];
        Int globalProjected = 0;
        MPI_Allreduce(&localProjected, &globalProjected, 1, mpi_type<Int>(), MPI_SUM, sub);
        if (requireExact) check(globalProjected == 0, tag + ": all selected donor inverses are exact");
        if (matching) {
            check(globalProjected == 0, tag + ": matching inverse remains unrestricted");
            const dstype relative = std::max(res.maxRelativeInverseResidual[0],
                                              res.maxRelativeInverseResidual[1]);
            check(relative <= 64 * std::numeric_limits<dstype>::epsilon(),
                  tag + ": matching inverse satisfies scale-invariant tolerance");
        }
        std::vector<double> local;
        pack_setup_records(local, nm.dir[1 - model], rs, m);
        int count = static_cast<int>(local.size());
        std::vector<int> counts(n), displs(n);
        MPI_Gather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, sub);
        int total = 0;
        for (int r = 0; r < n; r++) { displs[r] = total; total += counts[r]; }
        if (worldRank == 0) gathered.resize(static_cast<std::size_t>(total));
        MPI_Gatherv(local.data(), count, MPI_DOUBLE, gathered.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, sub);
        free_nonmatching_data(nm);
        free_nonmatching_data(nm);
        CPUFREE(requestCommon.requests);
        CPUFREE(requestCommon.statuses);
        MPI_Comm_free(&sub);
        EXASIM_COMM_WORLD = MPI_COMM_WORLD;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    return setup_to_map(gathered, setup_record_size(m));
}

void check_matching_faceperm(const SetupMap& records, const ModelMesh& m1, const ModelMesh& m2,
        const MasterData& m, int nparts)
{
    const auto g1 = global_faces(m1, m, 0, nparts), g2 = global_faces(m2, m, 1, nparts);
    for (const auto& kv : records) {
        const auto& v = kv.second;
        const int dir = static_cast<int>(v[0]), ownerGlobal = static_cast<int>(v[1]);
        const int g = static_cast<int>(v[2]), donorGlobal = static_cast<int>(v[4]);
        const auto& own = dir == 0 ? g1 : g2;
        const auto& donor = dir == 0 ? g2 : g1;
        int lo = -1, ld = -1;
        for (int i = 0; i < own.size(); i++) if (own.global[i] == ownerGlobal) lo = i;
        for (int i = 0; i < donor.size(); i++) if (donor.global[i] == donorGlobal) ld = i;
        check(lo >= 0 && ld >= 0, "phase3 matching: global faces found");
        if (lo < 0 || ld < 0) continue;
        const int psiOffset = 14 + 2 * m.npe;
        dstype coordinateMagnitude = 0, faceExtent = 0;
        for (int d = 0; d < m.nd; ++d) {
            dstype xmin = donor.nodes[m.npf * d + m.npf * m.nd * ld];
            dstype xmax = xmin;
            for (int j = 0; j < m.npf; ++j) {
                const dstype value = donor.nodes[j + m.npf * d + m.npf * m.nd * ld];
                coordinateMagnitude = std::max(coordinateMagnitude, std::abs(value));
                xmin = std::min(xmin, value);
                xmax = std::max(xmax, value);
            }
            faceExtent = std::max(faceExtent, xmax - xmin);
        }
        const dstype psiTolerance = std::max<dstype>(2e-13,
                64 * std::numeric_limits<dstype>::epsilon() * coordinateMagnitude /
                std::max(faceExtent, std::numeric_limits<dstype>::min()));
        for (int j = 0; j < m.npf; j++) {
            int ownerNode = -1;
            for (int i = 0; i < m.npf; i++)
                if (std::abs(donor.nodes[j + m.npf * m.nd * ld] - own.nodes[i + m.npf * m.nd * lo]) < 1e-13 &&
                    std::abs(donor.nodes[j + m.npf + m.npf * m.nd * ld] - own.nodes[i + m.npf + m.npf * m.nd * lo]) < 1e-13)
                    ownerNode = i;
            check(ownerNode >= 0, "phase3 matching: faceperm node found");
            if (ownerNode >= 0)
                check(std::abs(v[psiOffset + j] - m.shapfgt[g + m.ngf * ownerNode]) <= psiTolerance,
                      "phase3 matching: psi reproduces faceperm");
        }
    }
}

void check_owner_geometry_indexing(const SetupMap& records, const ModelMesh& m1, const ModelMesh& m2,
        const MasterData& m, int nparts)
{
    const auto g1 = global_faces(m1, m, 0, nparts), g2 = global_faces(m2, m, 1, nparts);
    for (const auto& kv : records) {
        const auto& v = kv.second;
        const int dir = static_cast<int>(v[0]), ownerGlobal = static_cast<int>(v[1]);
        const int g = static_cast<int>(v[2]);
        const auto& own = dir == 0 ? g1 : g2;
        int l = -1;
        for (int i = 0; i < own.size(); i++) if (own.global[i] == ownerGlobal) l = i;
        check(l >= 0, "phase3 geometry: owner face found");
        if (l < 0) continue;
        dstype tx = 0, ty = 0;
        for (int j = 0; j < m.npf; j++) {
            const dstype dpsi = m.shapfgt[m.ngf * m.npf + g + m.ngf * j];
            tx += dpsi * own.nodes[j + m.npf * m.nd * l];
            ty += dpsi * own.nodes[j + m.npf + m.npf * m.nd * l];
        }
        const dstype jac = std::hypot(tx, ty);
        check(std::abs(v[8] - ty / jac) < 2e-13 && std::abs(v[9] + tx / jac) < 2e-13,
              "phase3 geometry: nlg uses g + ngf*(lf + nfe*e)");
        check(std::abs(v[10] - m.gwf[g] * jac) < 2e-13,
              "phase3 geometry: pwJ uses matching jac index and Gauss weight");
    }
}

void run_phase3_case(const std::string& name, const ModelMesh& m1, const ModelMesh& m2,
        MasterData& m, bool matching, bool requireExact = false)
{
    std::vector<SetupMap> partitions;
    for (int n = 2; n <= worldSize; n *= 2)
        partitions.push_back(run_setup_partition(name, m1, m2, m, n, matching, requireExact));
    if (worldRank != 0) return;
    const std::size_t expected = static_cast<std::size_t>((global_faces(m1, m, 0, worldSize / 2).size() +
            global_faces(m2, m, 1, worldSize / 2).size()) * m.ngf);
    for (std::size_t i = 0; i < partitions.size(); i++) {
        check(partitions[i].size() == expected, name + ": sum_r |G_l,r| = ngf");
        if (i > 0) check(partitions[i] == partitions[0], name + ": partition-independent stored point data");
        std::printf("%s phase3: %d per model, %zu stored points, checksum %016llx\n", name.c_str(),
                    1 << i, partitions[i].size(), checksum(partitions[i]));
    }
    check_owner_geometry_indexing(partitions.back(), m1, m2, m, worldSize / 2);
    if (matching) check_matching_faceperm(partitions.back(), m1, m2, m, worldSize / 2);
}

ModelMesh transformed_mesh(ModelMesh mesh, dstype scale, dstype tx, dstype ty)
{
    for (dstype& x : mesh.xs) x = tx + scale * x;
    for (dstype& y : mesh.ys) y = ty + scale * y;
    mesh.xiface = tx + scale * mesh.xiface;
    mesh.xfar = tx + scale * mesh.xfar;
    mesh.amp *= scale;
    return mesh;
}

// [nonmatching] Phase 9 (C5): Gamma^1 extends 12.5% beyond Gamma^2.  Production keeps
// no OUTSIDE enum, so extrapolated owner points are identified from the stored donor
// reference coordinates pzeta (outside the reference square), compared with the
// geometric prediction (owner y > 1), and the stored basis values at those points are
// checked against an independent polynomial-reproduction reference.
void check_extension_case(MasterData& m)
{
    MPI_Comm sub = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < 2 ? 0 : MPI_UNDEFINED, worldRank, &sub);
    long counts[6] = {0, 0, 0, 0, 0, 0};   // predicted, identified, mismatch, reproduction failures, points, dir-1 extrapolated
    double maxReproduction = 0;
    if (sub != MPI_COMM_NULL) {
        EXASIM_COMM_WORLD = sub;
        ModelMesh e1, e2;
        e1.xs = uniform(0.0, 0.5, 4); e1.ys = uniform(0.0, 1.125, 9); e1.xiface = 0.5; e1.xfar = 0.0;
        e2.xs = uniform(0.5, 1.0, 3); e2.ys = uniform(0.0, 1.0, 6); e2.xiface = 0.5; e2.xfar = 1.0;
        const int model = worldRank;
        RankStructs rs;
        build_rank_structs(rs, model == 0 ? e1 : e2, m, 0, 1, 1);
        nonmatchingdata nm;
        std::unique_ptr<exasim::nonmatching::NonmatchingRuntime<dstype, Int>> runtime;
        commonstruct requestCommon;
        requestCommon.components.nc = 3; requestCommon.components.ncu = 1;
        requestCommon.components.ncq = 2; requestCommon.components.ncx = 2;
        requestCommon.components.nco = 0; requestCommon.components.ncw = 0;
        requestCommon.meshsizes.ne = rs.mesh.ndims[1];
        requestCommon.meshsizes.ne1 = rs.mesh.ndims[1];
        Int interfaceMap = 0;
        rs.app.interfacefluxmap = &interfaceMap;
        std::vector<Int> intfaces;
        for (Int e = 0; e < rs.mesh.ndims[1]; ++e)
            for (Int lf = 0; lf < m.nfe; ++lf)
                if (rs.mesh.bf[lf + m.nfe * e] == 2) intfaces.push_back(lf + m.nfe * e);
        rs.mesh.intfaces = intfaces.data();
        rs.mesh.szintfaces = static_cast<Int>(intfaces.size());
        DonorSearchResult<dstype, Int> res;
        check(rebuild_nonmatching_interface(rs.app, rs.master, rs.mesh, rs.sol, requestCommon,
                  Int(2), Int(worldRank), Int(model), nm, runtime, Int(0), &res),
              "extension: setup ran");
        const dstype tol = 1e-10;
        auto poly = [](dstype x, dstype y) { return 1.0 + 2.0 * x - 3.0 * y + x * y + y * y - 0.5 * x * x; };
        for (int direction = 0; direction < 2; ++direction) {
            const nonmatchinginterface& d = nm.dir[direction];
            if (!d.isdonor) continue;
            for (Int p = 0; p < d.npts; ++p) {
                const dstype y = d.px[p + static_cast<std::size_t>(d.npts) * 1];
                bool outside = false;
                for (Int k = 0; k < d.nd; ++k) {
                    const dstype z = d.pzeta[p + static_cast<std::size_t>(d.npts) * k];
                    outside = outside || z < -tol || z > 1 + tol;
                }
                if (direction == 0) {
                    const bool predicted = y > 1.0 + tol;
                    counts[0] += predicted; counts[1] += outside; counts[2] += predicted != outside;
                    counts[4] += 1;
                }
                else counts[5] += outside;
                // Polynomial reproduction (Q2 polynomial, affine elements): the stored
                // basis values at pzeta must reproduce f(x) also outside the element.
                const Int e = d.pe[p];
                dstype value = 0;
                for (Int j = 0; j < d.npe; ++j) {
                    const dstype xj = rs.sol.xdg[j + static_cast<std::size_t>(m.npe) * (0 + m.nd * e)];
                    const dstype yj = rs.sol.xdg[j + static_cast<std::size_t>(m.npe) * (1 + m.nd * e)];
                    value += d.pphi[j + static_cast<std::size_t>(d.npe) * p] * poly(xj, yj);
                }
                const dstype exact = poly(d.px[p], y);
                const dstype error = std::abs(value - exact);
                maxReproduction = std::max(maxReproduction, static_cast<double>(error));
                counts[3] += error > 1e-12 * std::max<dstype>(1, std::abs(exact));
            }
        }
        exasim::nonmatching::free_nonmatching_data(nm);
        MPI_Comm_free(&sub);
        EXASIM_COMM_WORLD = MPI_COMM_WORLD;
    }
    long total[6];
    double maxGlobal = 0;
    MPI_Allreduce(counts, total, 6, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&maxReproduction, &maxGlobal, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    check(total[0] > 0, "extension: owner points beyond Gamma^2 exist");
    check(total[2] == 0 && total[0] == total[1],
          "extension: extrapolated points identified from pzeta equal the geometric prediction");
    check(total[5] == 0, "extension: no extrapolated point in the fully covered direction");
    check(total[3] == 0, "extension: stored basis values reproduce a Q2 polynomial at every point");
    if (worldRank == 0)
        std::printf("extension: %ld of %ld Gamma^1 points extrapolated (predicted %ld), direction-1 extrapolated %ld, "
                    "max reproduction error %.3e\n", total[1], total[4], total[0], total[5], maxGlobal);
}

} // namespace

#ifndef NONMATCHING_SETUP_NO_MAIN
int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    EXASIM_COMM_WORLD = MPI_COMM_WORLD;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    if (worldSize < 2 || (worldSize & (worldSize - 1)) != 0) {
        if (worldRank == 0) std::printf("nonmatching_setup needs 2, 4, 8, ... ranks\n");
        MPI_Finalize();
        return 1;
    }

    MasterData matchingMaster = make_master(1, 3);
    ModelMesh matching1, matching2;
    matching_meshes(matching1, matching2);
    run_phase3_case("matching", matching1, matching2, matchingMaster, true);

    ModelMesh scaledLarge1 = transformed_mesh(matching1, 1e4, 0.0, 0.0);
    ModelMesh scaledLarge2 = transformed_mesh(matching2, 1e4, 0.0, 0.0);
    run_phase3_case("matching-scale-1e4", scaledLarge1, scaledLarge2, matchingMaster, true);

    ModelMesh translated1 = transformed_mesh(matching1, 1.0, 1e4, 1e4);
    ModelMesh translated2 = transformed_mesh(matching2, 1.0, 1e4, 1e4);
    run_phase3_case("matching-translate-1e4", translated1, translated2, matchingMaster, true);

    ModelMesh scaledSmall1 = transformed_mesh(matching1, 1e-6, 0.0, 0.0);
    ModelMesh scaledSmall2 = transformed_mesh(matching2, 1e-6, 0.0, 0.0);
    run_phase3_case("matching-scale-1e-6", scaledSmall1, scaledSmall2, matchingMaster, true);

    MasterData nonmatchingMaster = make_master(1, 2);
    ModelMesh nonmatching1, nonmatching2;
    nonmatching1.xs = uniform(0.0, 0.5, 4); nonmatching1.ys = uniform(0.0, 1.0, 12);
    nonmatching1.xiface = 0.5; nonmatching1.xfar = 0.0;
    nonmatching2.xs = uniform(0.51, 1.0, 3);
    nonmatching2.ys = {0.0, 0.11, 0.27, 0.38, 0.55, 0.71, 0.82, 0.9, 1.0};
    nonmatching2.xiface = 0.51; nonmatching2.xfar = 1.0;
    run_phase3_case("nonmatching", nonmatching1, nonmatching2, nonmatchingMaster, false);

    ModelMesh curvedQuad1 = nonmatching1, curvedQuad2 = nonmatching2;
    curvedQuad2.xiface = 0.5; curvedQuad2.xs = uniform(0.5, 1.0, 3);
    curvedQuad2.amp = 0.01;
    run_phase3_case("curved-quads", curvedQuad1, curvedQuad2, nonmatchingMaster, false);

    MasterData triangleMaster = make_master(0, 2);
    ModelMesh triangle1, triangle2;
    triangle1.tri = true;
    triangle1.xs = uniform(0.0, 0.5, 4); triangle1.ys = uniform(0.0, 1.0, 12);
    triangle1.xiface = 0.5; triangle1.xfar = 0.0;
    triangle2.tri = true;
    triangle2.xs = uniform(0.5, 1.0, 3);
    triangle2.ys = {0.0, 0.11, 0.27, 0.38, 0.55, 0.71, 0.82, 0.9};
    triangle2.xiface = 0.5; triangle2.xfar = 1.0; triangle2.amp = 0.01;
    run_phase3_case("curved-triangles-beyond-edge", triangle1, triangle2,
                    triangleMaster, false, true);

    check_extension_case(nonmatchingMaster); // [nonmatching] Phase 9 C5

    int total = 0;
    MPI_Allreduce(&failures, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (worldRank == 0)
        std::printf("%s nonmatching_setup (%d ranks, up to %d per model)\n",
                    total == 0 ? "PASS" : "FAIL", worldSize, worldSize / 2);
    MPI_Finalize();
    return total == 0 ? 0 : 1;
}
#endif
