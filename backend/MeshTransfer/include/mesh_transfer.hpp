#pragma once

#include "geometry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace exasim::meshtransfer {

template <class Scalar = DefaultScalar, class Index = DefaultIndex>
class MeshAdapter {
public:
    using scalar_type = Scalar;
    using index_type = Index;
    using memory_space = typename Kokkos::DefaultExecutionSpace::memory_space;

    struct DeviceView {
        const Scalar* xe = nullptr;
        const Index* neighborRank = nullptr;
        const Index* neighborElem = nullptr;
        const Scalar* invV = nullptr;
        Index elemtype = 0;
        Index porder = 0;
        Index npe = 0;
        Index nfe = 0;
        Index nd = 0;
        Index ne = 0;
        Index myrank = 0;
        Index maxNewtonIterations = 20;
        Scalar newtonTolerance = Scalar(1.0e-12);
        Scalar insideTolerance = Scalar(1.0e-10);

        KOKKOS_INLINE_FUNCTION
        void modal_basis(const Scalar* xi, Scalar* phi, Scalar* dphi) const
        {
            detail::modal_basis(xi, phi, dphi, elemtype, porder, nd);
        }

        KOKKOS_INLINE_FUNCTION
        void shape_values(const Scalar* xi, Scalar* shape, Scalar* phi) const
        {
            modal_basis(xi, phi, nullptr);
            for (Index a = 0; a < npe; ++a) {
                Scalar value = 0;
                for (Index m = 0; m < npe; ++m)
                    value += phi[m] * invV[m * npe + a];
                shape[a] = value;
            }
        }

        KOKKOS_INLINE_FUNCTION
        void shape(
            const Scalar* xi,
            Scalar* shapeValues,
            Scalar* shapeDerivatives,
            Scalar* phi,
            Scalar* dphi) const
        {
            modal_basis(xi, phi, dphi);
            for (Index a = 0; a < npe; ++a) {
                Scalar value = 0;
                for (Index m = 0; m < npe; ++m)
                    value += phi[m] * invV[m * npe + a];
                shapeValues[a] = value;
                for (Index d = 0; d < nd; ++d) {
                    Scalar derivative = 0;
                    for (Index m = 0; m < npe; ++m)
                        derivative += dphi[m + npe * d] * invV[m * npe + a];
                    shapeDerivatives[a + npe * d] = derivative;
                }
            }
        }

        KOKKOS_INLINE_FUNCTION
        bool inside(const Scalar* xi) const
        {
            return detail::inside_reference(
                xi, elemtype, nd, insideTolerance);
        }

        KOKKOS_INLINE_FUNCTION
        Index exit_face(const Scalar* xi) const
        {
            return detail::exit_face(xi, elemtype, nd);
        }

        KOKKOS_INLINE_FUNCTION
        bool neighbor(
            Index elem, Index face, Index& rankNext, Index& elemNext) const
        {
            if (!neighborRank || !neighborElem || elem < 0 || elem >= ne ||
                face < 0 || face >= nfe) {
                rankNext = -1;
                elemNext = -1;
                return false;
            }
            const Index q = face + nfe * elem;
            rankNext = neighborRank[q];
            elemNext = neighborElem[q];
            return rankNext >= 0 && elemNext >= 0;
        }

        KOKKOS_INLINE_FUNCTION
        InverseMapStatus inverse_map(
            const Scalar* x,
            Index elem,
            Scalar* xi,
            Scalar* shapeValues,
            Scalar* shapeDerivatives,
            Scalar* phi,
            Scalar* dphi) const
        {
            if (elem < 0 || elem >= ne)
                return InverseMapStatus::Nonconverged;
            for (Index d = 0; d < nd; ++d) {
                if (!isfinite(xi[d]))
                    xi[d] = elemtype == 0
                        ? Scalar(1) / Scalar(nd + 1) : Scalar(0.5);
            }
            for (Index iteration = 0; iteration < maxNewtonIterations; ++iteration) {
                shape(xi, shapeValues, shapeDerivatives, phi, dphi);
                Scalar residual[3] = {Scalar(0), Scalar(0), Scalar(0)};
                Scalar jacobian[9] = {
                    Scalar(0), Scalar(0), Scalar(0),
                    Scalar(0), Scalar(0), Scalar(0),
                    Scalar(0), Scalar(0), Scalar(0)};
                for (Index d = 0; d < nd; ++d) {
                    Scalar mapped = 0;
                    for (Index a = 0; a < npe; ++a) {
                        const Scalar coordinate = xe[a + npe * d + npe * nd * elem];
                        mapped += shapeValues[a] * coordinate;
                        for (Index k = 0; k < nd; ++k)
                            jacobian[d * nd + k] +=
                                shapeDerivatives[a + npe * k] * coordinate;
                    }
                    residual[d] = x[d] - mapped;
                }
                Scalar residualNormSquared = 0;
                for (Index d = 0; d < nd; ++d)
                    residualNormSquared += residual[d] * residual[d];
                if (sqrt(residualNormSquared) <= newtonTolerance)
                    return InverseMapStatus::Converged;
                Scalar delta[3] = {Scalar(0), Scalar(0), Scalar(0)};
                if (!detail::solve_small_system(delta, jacobian, residual, nd))
                    return InverseMapStatus::Singular;
                Scalar deltaNormSquared = 0;
                for (Index d = 0; d < nd; ++d) {
                    xi[d] += delta[d];
                    deltaNormSquared += delta[d] * delta[d];
                }
                if (sqrt(deltaNormSquared) <= newtonTolerance)
                    return InverseMapStatus::Converged;
            }
            return InverseMapStatus::Nonconverged;
        }

        KOKKOS_INLINE_FUNCTION
        void evaluate_field(
            const Scalar* field,
            Index elem,
            const Scalar* xi,
            Index nc,
            Scalar* value,
            Scalar* shapeValues,
            Scalar* phi) const
        {
            shape_values(xi, shapeValues, phi);
            for (Index c = 0; c < nc; ++c) {
                Scalar sum = 0;
                for (Index a = 0; a < npe; ++a)
                    sum += shapeValues[a] *
                        field[a + npe * c + npe * nc * elem];
                value[c] = sum;
            }
        }
    };

    MeshAdapter(
        const Scalar* xe,
        const Scalar* xpe,
        const Index* neighborRank,
        const Index* neighborElem,
        Index elemtype,
        Index porder,
        Index npe,
        Index nfe,
        Index nd,
        Index ne,
        Index myrank)
        : xe_(xe), xpe_(xpe), neighborRank_(neighborRank),
          neighborElem_(neighborElem), elemtype_(elemtype), porder_(porder),
          npe_(npe), nfe_(nfe), nd_(nd), ne_(ne), myrank_(myrank)
    {
        validate();
        construct_inverse_vandermonde();
    }

    ~MeshAdapter() = default;
    MeshAdapter(const MeshAdapter&) = delete;
    MeshAdapter& operator=(const MeshAdapter&) = delete;
    MeshAdapter(MeshAdapter&&) noexcept = default;
    MeshAdapter& operator=(MeshAdapter&&) noexcept = default;

    Index elemtype() const noexcept { return elemtype_; }
    Index porder() const noexcept { return porder_; }
    Index nodes_per_element() const noexcept { return npe_; }
    Index faces_per_element() const noexcept { return nfe_; }
    Index spatial_dimension() const noexcept { return nd_; }
    Index num_elements() const noexcept { return ne_; }
    Index rank() const noexcept { return myrank_; }
    const Scalar* geometry_data() const noexcept { return xe_; }
    const Scalar* reference_nodes() const noexcept { return xpe_; }
    const Scalar* inverse_vandermonde() const noexcept { return invV_.data(); }
    Scalar vandermonde_condition_estimate() const noexcept { return conditionEstimate_; }
    DeviceView device_view() const noexcept
    {
        return {xe_, neighborRank_, neighborElem_, invV_.data(), elemtype_,
                porder_, npe_, nfe_, nd_, ne_, myrank_,
                maxNewtonIterations_, newtonTolerance_, insideTolerance_};
    }
    void set_newton_options(
        Index maxIterations, Scalar newtonTolerance, Scalar insideTolerance)
    {
        if (maxIterations <= 0 || newtonTolerance <= Scalar(0) ||
            insideTolerance < Scalar(0))
            throw MeshTransferError("invalid inverse-map tolerances");
        maxNewtonIterations_ = maxIterations;
        newtonTolerance_ = newtonTolerance;
        insideTolerance_ = insideTolerance;
    }

private:
    void validate() const
    {
        if (!xe_ || !xpe_)
            throw MeshTransferError("MeshAdapter requires non-null xe and xpe");
        if (nd_ < 1 || nd_ > 3)
            throw MeshTransferError("MeshAdapter supports spatial dimensions 1, 2, and 3");
        if (elemtype_ != 0 && elemtype_ != 1)
            throw MeshTransferError("MeshAdapter elemtype must be 0 (simplex) or 1 (tensor)");
        if (porder_ < 0 || npe_ <= 0 || nfe_ <= 0 || ne_ <= 0)
            throw MeshTransferError("MeshAdapter dimensions and polynomial order are invalid");
        if (porder_ > detail::max_polynomial_order ||
            npe_ > detail::max_element_nodes)
            throw MeshTransferError("MeshAdapter polynomial order or node count exceeds device scratch limits");
        if (npe_ != detail::expected_nodes<Scalar>(elemtype_, porder_, nd_))
            throw MeshTransferError("npe is inconsistent with elemtype, porder, and nd");
        if ((neighborRank_ == nullptr) != (neighborElem_ == nullptr))
            throw MeshTransferError("neighborRank and neighborElem must both be null or non-null");
    }

    void construct_inverse_vandermonde()
    {
        using unmanaged_view =
            Kokkos::View<const Scalar*, memory_space, Kokkos::MemoryUnmanaged>;
        unmanaged_view xpeDevice(xpe_, static_cast<std::size_t>(npe_ * nd_));
        auto xpeHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), xpeDevice);

        std::vector<Scalar> matrix(static_cast<std::size_t>(npe_ * npe_));
        std::vector<Scalar> inverse(static_cast<std::size_t>(npe_ * npe_), Scalar(0));
        std::vector<Scalar> phi(static_cast<std::size_t>(npe_));
        Scalar xi[3] = {Scalar(0), Scalar(0), Scalar(0)};
        for (Index a = 0; a < npe_; ++a) {
            for (Index d = 0; d < nd_; ++d)
                xi[d] = xpeHost(a + npe_ * d);
            detail::modal_basis(xi, phi.data(), static_cast<Scalar*>(nullptr),
                                elemtype_, porder_, nd_);
            for (Index m = 0; m < npe_; ++m)
                matrix[static_cast<std::size_t>(a * npe_ + m)] = phi[m];
            inverse[static_cast<std::size_t>(a * npe_ + a)] = Scalar(1);
        }

        Scalar matrixNorm = 0;
        for (Index row = 0; row < npe_; ++row) {
            Scalar sum = 0;
            for (Index col = 0; col < npe_; ++col)
                sum += std::abs(matrix[static_cast<std::size_t>(row * npe_ + col)]);
            matrixNorm = std::max(matrixNorm, sum);
        }

        const Scalar pivotTolerance =
            Scalar(64) * std::numeric_limits<Scalar>::epsilon() *
            std::max(Scalar(1), matrixNorm);
        for (Index col = 0; col < npe_; ++col) {
            Index pivot = col;
            Scalar pivotAbs = std::abs(matrix[static_cast<std::size_t>(col * npe_ + col)]);
            for (Index row = col + 1; row < npe_; ++row) {
                const Scalar candidate =
                    std::abs(matrix[static_cast<std::size_t>(row * npe_ + col)]);
                if (candidate > pivotAbs) {
                    pivot = row;
                    pivotAbs = candidate;
                }
            }
            if (pivotAbs <= pivotTolerance)
                throw MeshTransferError("reference-node Vandermonde matrix is singular");
            if (pivot != col) {
                for (Index j = 0; j < npe_; ++j) {
                    std::swap(matrix[static_cast<std::size_t>(col * npe_ + j)],
                              matrix[static_cast<std::size_t>(pivot * npe_ + j)]);
                    std::swap(inverse[static_cast<std::size_t>(col * npe_ + j)],
                              inverse[static_cast<std::size_t>(pivot * npe_ + j)]);
                }
            }
            const Scalar diagonal = matrix[static_cast<std::size_t>(col * npe_ + col)];
            for (Index j = 0; j < npe_; ++j) {
                matrix[static_cast<std::size_t>(col * npe_ + j)] /= diagonal;
                inverse[static_cast<std::size_t>(col * npe_ + j)] /= diagonal;
            }
            for (Index row = 0; row < npe_; ++row) {
                if (row == col) continue;
                const Scalar factor = matrix[static_cast<std::size_t>(row * npe_ + col)];
                for (Index j = 0; j < npe_; ++j) {
                    matrix[static_cast<std::size_t>(row * npe_ + j)] -=
                        factor * matrix[static_cast<std::size_t>(col * npe_ + j)];
                    inverse[static_cast<std::size_t>(row * npe_ + j)] -=
                        factor * inverse[static_cast<std::size_t>(col * npe_ + j)];
                }
            }
        }

        Scalar inverseNorm = 0;
        for (Index row = 0; row < npe_; ++row) {
            Scalar sum = 0;
            for (Index col = 0; col < npe_; ++col)
                sum += std::abs(inverse[static_cast<std::size_t>(row * npe_ + col)]);
            inverseNorm = std::max(inverseNorm, sum);
        }
        conditionEstimate_ = matrixNorm * inverseNorm;

        invV_ = Kokkos::View<Scalar*, memory_space>(
            "meshtransfer_inverse_vandermonde", static_cast<std::size_t>(npe_ * npe_));
        auto inverseHost = Kokkos::create_mirror_view(invV_);
        for (Index i = 0; i < npe_ * npe_; ++i)
            inverseHost(i) = inverse[static_cast<std::size_t>(i)];
        Kokkos::deep_copy(invV_, inverseHost);
    }

    const Scalar* xe_ = nullptr;
    const Scalar* xpe_ = nullptr;
    const Index* neighborRank_ = nullptr;
    const Index* neighborElem_ = nullptr;
    Kokkos::View<Scalar*, memory_space> invV_;
    Scalar conditionEstimate_ = Scalar(0);
    Index elemtype_ = 0;
    Index porder_ = 0;
    Index npe_ = 0;
    Index nfe_ = 0;
    Index nd_ = 0;
    Index ne_ = 0;
    Index myrank_ = 0;
    Index maxNewtonIterations_ = 20;
    Scalar newtonTolerance_ =
        Scalar(128) * std::numeric_limits<Scalar>::epsilon();
    Scalar insideTolerance_ =
        Scalar(1024) * std::numeric_limits<Scalar>::epsilon();
};

template <class Scalar = DefaultScalar, class Index = DefaultIndex>
class DistributedMeshTransfer {
public:
    using scalar_type = Scalar;
    using index_type = Index;
    using mesh_type = MeshAdapter<Scalar, Index>;
    using memory_space = typename Kokkos::DefaultExecutionSpace::memory_space;

    DistributedMeshTransfer(Comm comm, mesh_type mesh)
        : comm_(comm), mesh_(std::move(mesh))
    {
#ifdef HAVE_MPI
        if (comm_ == MPI_COMM_NULL)
            throw MeshTransferError("DistributedMeshTransfer requires a valid MPI communicator");
        MPI_Comm_rank(comm_, &myrankInt_);
        MPI_Comm_size(comm_, &nprocInt_);
        myrank_ = static_cast<Index>(myrankInt_);
        nproc_ = static_cast<Index>(nprocInt_);
#else
        (void)comm_;
        myrank_ = 0;
        nproc_ = 1;
#endif
        reserve_rank_workspace();
    }

    ~DistributedMeshTransfer() = default;
    DistributedMeshTransfer(const DistributedMeshTransfer&) = delete;
    DistributedMeshTransfer& operator=(const DistributedMeshTransfer&) = delete;
    DistributedMeshTransfer(DistributedMeshTransfer&&) noexcept = default;
    DistributedMeshTransfer& operator=(DistributedMeshTransfer&&) noexcept = default;

    void evaluate_shape(const Scalar* xi, Scalar* shap, Index np) const
    {
        if (np < 0 || (np > 0 && (!xi || !shap)))
            throw MeshTransferError("evaluate_shape received invalid arrays or point count");
        const auto mesh = mesh_.device_view();
        const Index nd = mesh.nd;
        const Index npe = mesh.npe;
        Kokkos::parallel_for(
            "MeshTransfer::evaluate_shape", range_policy<Index>(0, np),
            KOKKOS_LAMBDA(const Index i) {
                Scalar xiLocal[3] = {Scalar(0), Scalar(0), Scalar(0)};
                Scalar phi[detail::max_element_nodes];
                Scalar shape[detail::max_element_nodes];
                for (Index d = 0; d < nd; ++d)
                    xiLocal[d] = xi[d + nd * i];
                mesh.shape_values(xiLocal, shape, phi);
                for (Index a = 0; a < npe; ++a)
                    shap[a + npe * i] = shape[a];
            });
    }

    void locate(const Scalar* x, Index* erank, Index* e, Scalar* xi, Index np)
    {
        validate_location_arguments(x, erank, e, xi, np);
        validate_location_map(erank, e, np);
        reserve(np);
        Kokkos::deep_copy(pointState_, Index(0));
        const Index nactive = get_active_ids(erank, np);
        const Index unresolved = local_element_search(
            x, erank, e, xi, activeIds_, nactive, np);
#ifdef HAVE_MPI
        if (nproc_ > 1)
            distributed_locate(x, erank, e, xi, np, unresolved);
#else
        (void)unresolved;
#endif
    }

    void evaluate(
        const Scalar* source,
        Scalar* target,
        const Index* erank,
        const Index* e,
        const Scalar* xi,
        Index np)
    {
        if (nc_ <= 0)
            throw MeshTransferError("set_num_components must be called before evaluate");
        if (np < 0 || (np > 0 && (!source || !target || !erank || !e || !xi)))
            throw MeshTransferError("evaluate received invalid arrays or point count");
        validate_location_map(erank, e, np);
        evaluate_local(source, target, erank, e, xi, np);
#ifdef HAVE_MPI
        if (nproc_ > 1)
            evaluate_remote(source, target, erank, e, xi, np);
#endif
    }

    void evaluate(
        const Scalar* source,
        Scalar* target,
        const Index* erank,
        const Index* e,
        const Scalar* xi,
        Index np,
        Index nc)
    {
        set_num_components(nc);
        evaluate(source, target, erank, e, xi, np);
    }

    void transfer(
        const Scalar* x,
        const Scalar* source,
        Scalar* target,
        Index* erank,
        Index* e,
        Scalar* xi,
        Index np)
    {
        locate(x, erank, e, xi, np);
        evaluate(source, target, erank, e, xi, np);
    }

    void transfer(
        const Scalar* x,
        const Scalar* source,
        Scalar* target,
        Index* erank,
        Index* e,
        Scalar* xi,
        Index np,
        Index nc)
    {
        set_num_components(nc);
        transfer(x, source, target, erank, e, xi, np);
    }

    void set_target_points(
        const Scalar* x, Index np, Index* erank, Index* e, Scalar* xi)
    {
        if (np < 0)
            throw MeshTransferError("target point count cannot be negative");
        if (np > 0 && (!x || !erank || !e || !xi))
            throw MeshTransferError("registered target arrays cannot be null when np is positive");
        x_ = x;
        np_ = np;
        erank_ = erank;
        e_ = e;
        xi_ = xi;
        reserve(np);
    }

    void transfer(const Scalar* source, Scalar* target)
    {
        if (np_ < 0 || (np_ > 0 && (!x_ || !erank_ || !e_ || !xi_)))
            throw MeshTransferError("target points have not been registered");
        transfer(x_, source, target, erank_, e_, xi_, np_);
    }

    void transfer(const Scalar* source, Scalar* target, Index nc)
    {
        set_num_components(nc);
        transfer(source, target);
    }

    void set_num_components(Index nc)
    {
        if (nc <= 0)
            throw MeshTransferError("number of field components must be positive");
        nc_ = nc;
    }

    void reserve(Index np)
    {
        if (np < 0)
            throw MeshTransferError("reserve capacity cannot be negative");
        if (np <= pointCapacity_)
            return;
        activeIds_ = index_view("meshtransfer_active_ids", np);
        activeIdsNext_ = index_view("meshtransfer_active_ids_next", np);
        status_ = index_view("meshtransfer_status", np);
        pointState_ = index_view("meshtransfer_point_state", np);
        pointCapacity_ = np;
    }

    Index num_target_points() const noexcept { return np_; }
    Index spatial_dimension() const noexcept { return mesh_.spatial_dimension(); }
    Index num_components() const noexcept { return nc_; }
    Index point_capacity() const noexcept { return pointCapacity_; }
    Index send_query_capacity() const noexcept { return sendQueryCapacity_; }
    Index receive_query_capacity() const noexcept { return recvQueryCapacity_; }
    Index send_result_capacity() const noexcept { return sendResultCapacity_; }
    Index receive_result_capacity() const noexcept { return recvResultCapacity_; }
    const Scalar* registered_points() const noexcept { return x_; }
    Index* registered_ranks() const noexcept { return erank_; }
    Index* registered_elements() const noexcept { return e_; }
    Scalar* registered_reference_coordinates() const noexcept { return xi_; }

private:
    using index_view = Kokkos::View<Index*, memory_space>;
    using scalar_view = Kokkos::View<Scalar*, memory_space>;

    void reserve_rank_workspace()
    {
        const Index n = nproc_ + 1;
        sendCount_ = index_view("meshtransfer_send_count", n);
        recvCount_ = index_view("meshtransfer_recv_count", n);
        sendOffset_ = index_view("meshtransfer_send_offset", n);
        recvOffset_ = index_view("meshtransfer_recv_offset", n);
        cursor_ = index_view("meshtransfer_cursor", n);
        hostSendCount_.resize(static_cast<std::size_t>(nproc_));
        hostRecvCount_.resize(static_cast<std::size_t>(nproc_));
        hostSendOffset_.resize(static_cast<std::size_t>(nproc_ + 1));
        hostRecvOffset_.resize(static_cast<std::size_t>(nproc_ + 1));
        forwardSendCount_.resize(static_cast<std::size_t>(nproc_));
        forwardSendOffset_.resize(static_cast<std::size_t>(nproc_ + 1));
        resultSendCount_.resize(static_cast<std::size_t>(nproc_));
        resultSendOffset_.resize(static_cast<std::size_t>(nproc_ + 1));
    }

    void reserve_send_queries(Index n)
    {
        if (n <= sendQueryCapacity_) return;
        sendQueryCapacity_ = n;
        sendQueryInt_ = index_view("meshtransfer_send_query_int", 3 * n);
        sendQueryReal_ = scalar_view(
            "meshtransfer_send_query_real", 2 * mesh_.spatial_dimension() * n);
    }

    void reserve_recv_queries(Index n)
    {
        if (n <= recvQueryCapacity_) return;
        recvQueryCapacity_ = n;
        recvQueryInt_ = index_view("meshtransfer_recv_query_int", 3 * n);
        recvQueryReal_ = scalar_view(
            "meshtransfer_recv_query_real", 2 * mesh_.spatial_dimension() * n);
        queryDest_ = index_view("meshtransfer_query_destination", n);
        queryActiveIds_ = index_view("meshtransfer_query_active", n);
        queryActiveIdsNext_ = index_view("meshtransfer_query_active_next", n);
        queryStatus_ = index_view("meshtransfer_query_status", n);
    }

    void reserve_send_results(Index n, Index realWidth)
    {
        if (n <= sendResultCapacity_ && realWidth <= sendResultRealWidth_) return;
        sendResultCapacity_ = std::max(n, sendResultCapacity_);
        sendResultRealWidth_ = std::max(realWidth, sendResultRealWidth_);
        sendResultInt_ = index_view(
            "meshtransfer_send_result_int", 3 * sendResultCapacity_);
        sendResultReal_ = scalar_view(
            "meshtransfer_send_result_real",
            sendResultRealWidth_ * sendResultCapacity_);
    }

    void reserve_recv_results(Index n, Index realWidth)
    {
        if (n <= recvResultCapacity_ && realWidth <= recvResultRealWidth_) return;
        recvResultCapacity_ = std::max(n, recvResultCapacity_);
        recvResultRealWidth_ = std::max(realWidth, recvResultRealWidth_);
        recvResultInt_ = index_view(
            "meshtransfer_recv_result_int", 3 * recvResultCapacity_);
        recvResultReal_ = scalar_view(
            "meshtransfer_recv_result_real",
            recvResultRealWidth_ * recvResultCapacity_);
    }

    void copy_counts_to_host(const index_view& counts, std::vector<Index>& host) const
    {
        auto mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), counts);
        for (Index rank = 0; rank < nproc_; ++rank)
            host[static_cast<std::size_t>(rank)] = mirror(rank);
    }

    Index compute_offsets(
        const std::vector<Index>& counts,
        std::vector<Index>& offsets) const
    {
        offsets[0] = 0;
        for (Index rank = 0; rank < nproc_; ++rank)
            offsets[static_cast<std::size_t>(rank + 1)] =
                offsets[static_cast<std::size_t>(rank)] +
                counts[static_cast<std::size_t>(rank)];
        return offsets[static_cast<std::size_t>(nproc_)];
    }

    void set_device_offsets(const std::vector<Index>& offsets)
    {
        auto mirror = Kokkos::create_mirror_view(sendOffset_);
        for (Index rank = 0; rank <= nproc_; ++rank)
            mirror(rank) = offsets[static_cast<std::size_t>(rank)];
        Kokkos::deep_copy(sendOffset_, mirror);
        Kokkos::deep_copy(cursor_, Index(0));
    }

    void validate_location_arguments(
        const Scalar* x,
        const Index* erank,
        const Index* e,
        const Scalar* xi,
        Index np) const
    {
        if (np < 0 || (np > 0 && (!x || !erank || !e || !xi)))
            throw MeshTransferError("locate received invalid arrays or point count");
    }

    void validate_location_map(const Index* erank, const Index* e, Index np) const
    {
        const Index nproc = nproc_;
        const Index rank = myrank_;
        const Index ne = mesh_.num_elements();
        Kokkos::View<Index, memory_space> error("meshtransfer_map_error");
        Kokkos::deep_copy(error, Index(0));
        Kokkos::parallel_for(
            "MeshTransfer::ValidateLocationMap", range_policy<Index>(0, np),
            KOKKOS_LAMBDA(const Index i) {
                if (erank[i] < 0 || erank[i] >= nproc)
                    Kokkos::atomic_exchange(error.data(), Index(1));
                else if (erank[i] == rank && (e[i] < 0 || e[i] >= ne))
                    Kokkos::atomic_exchange(error.data(), Index(2));
            });
        auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), error);
        if (host() == 1)
            throw MeshTransferError("location map contains an invalid rank");
        if (host() == 2)
            throw MeshTransferError("location map contains an invalid local element");
    }

    Index get_active_ids(const Index* erank, Index np)
    {
        Index count = 0;
        const Index rank = myrank_;
        Index* active = activeIds_.data();
        Index* pointState = pointState_.data();
        Kokkos::parallel_scan(
            "MeshTransfer::GetActiveIds", range_policy<Index>(0, np),
            KOKKOS_LAMBDA(const Index i, Index& offset, const bool final) {
                const bool local = erank[i] == rank;
                if (final) {
                    pointState[i] = 0;
                    if (local) active[offset] = i;
                }
                if (local) ++offset;
            }, count);
        return count;
    }

    Index get_next_active_ids(
        const index_view& current,
        const index_view& next,
        Index nactive)
    {
        Index count = 0;
        const Index* active = current.data();
        const Index* status = status_.data();
        Index* output = next.data();
        Kokkos::parallel_scan(
            "MeshTransfer::GetNextActiveIds", range_policy<Index>(0, nactive),
            KOKKOS_LAMBDA(const Index k, Index& offset, const bool final) {
                if (status[k] == static_cast<Index>(LOCAL_NEXT)) {
                    if (final) output[offset] = active[k];
                    ++offset;
                }
            }, count);
        return count;
    }

    Index local_element_search(
        const Scalar* x,
        Index* erank,
        Index* e,
        Scalar* xi,
        index_view current,
        Index nactive,
        Index np)
    {
        const auto mesh = mesh_.device_view();
        const Index nd = mesh.nd;
        const Index rank = myrank_;
        index_view next = activeIdsNext_;
        Index* status = status_.data();
        Index* pointState = pointState_.data();
        Kokkos::View<Index, memory_space> error("meshtransfer_location_error");
        Kokkos::deep_copy(error, Index(0));
        const Index maxHops = 4 * mesh.ne + 8;
        Index hops = 0;
        while (nactive > 0 && hops < maxHops) {
            const Index* active = current.data();
            Kokkos::parallel_for(
                "MeshTransfer::ProcessActivePoints",
                range_policy<Index>(0, nactive),
                KOKKOS_LAMBDA(const Index k) {
                    const Index i = active[k];
                    const Index elem = e[i];
                    if (elem < 0 || elem >= mesh.ne) {
                        Kokkos::atomic_exchange(error.data(), Index(1));
                        status[k] = static_cast<Index>(OUTSIDE);
                        return;
                    }
                    Scalar xLocal[3] = {Scalar(0), Scalar(0), Scalar(0)};
                    Scalar xiLocal[3] = {Scalar(0), Scalar(0), Scalar(0)};
                    for (Index d = 0; d < nd; ++d) {
                        xLocal[d] = x[d + nd * i];
                        xiLocal[d] = xi[d + nd * i];
                    }
                    Scalar shape[detail::max_element_nodes];
                    Scalar dshape[3 * detail::max_element_nodes];
                    Scalar phi[detail::max_element_nodes];
                    Scalar dphi[3 * detail::max_element_nodes];
                    const auto inverseStatus = mesh.inverse_map(
                        xLocal, elem, xiLocal, shape, dshape, phi, dphi);
                    for (Index d = 0; d < nd; ++d)
                        xi[d + nd * i] = xiLocal[d];
                    if (inverseStatus != InverseMapStatus::Converged) {
                        Kokkos::atomic_exchange(
                            error.data(),
                            inverseStatus == InverseMapStatus::Singular ? Index(2) : Index(3));
                        status[k] = static_cast<Index>(OUTSIDE);
                        return;
                    }
                    if (mesh.inside(xiLocal)) {
                        erank[i] = rank;
                        pointState[i] = 1;
                        status[k] = static_cast<Index>(FOUND);
                        return;
                    }
                    const Index face = mesh.exit_face(xiLocal);
                    Index nextRank = -1;
                    Index nextElem = -1;
                    if (!mesh.neighbor(elem, face, nextRank, nextElem)) {
                        erank[i] = rank;
                        pointState[i] = 1;
                        status[k] = static_cast<Index>(OUTSIDE);
                    }
                    else if (nextRank == rank) {
                        e[i] = nextElem;
                        status[k] = static_cast<Index>(LOCAL_NEXT);
                    }
                    else {
                        erank[i] = nextRank;
                        e[i] = nextElem;
                        status[k] = static_cast<Index>(REMOTE);
                    }
                });
            Kokkos::fence();
            auto errorHost =
                Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), error);
            if (errorHost() != 0) {
                if (errorHost() == 1)
                    throw MeshTransferError("point location received an invalid local element");
                if (errorHost() == 2)
                    throw MeshTransferError("point location encountered a singular geometric Jacobian");
                throw MeshTransferError("inverse mapping failed to converge");
            }
            const Index nextCount = get_next_active_ids(current, next, nactive);
            std::swap(current, next);
            nactive = nextCount;
            ++hops;
        }
        if (nactive > 0)
            throw MeshTransferError("local element walk exceeded its hop limit");

        Index unresolved = 0;
        Kokkos::parallel_reduce(
            "MeshTransfer::CountRemotePoints", range_policy<Index>(0, np),
            KOKKOS_LAMBDA(const Index i, Index& sum) {
                if (pointState[i] == 0) ++sum;
            }, unresolved);
        return unresolved;
    }

    void evaluate_local(
        const Scalar* source,
        Scalar* target,
        const Index* erank,
        const Index* e,
        const Scalar* xi,
        Index np) const
    {
        const auto mesh = mesh_.device_view();
        const Index rank = myrank_;
        const Index nd = mesh.nd;
        const Index nc = nc_;
        Kokkos::parallel_for(
            "MeshTransfer::EvaluateLocal", range_policy<Index>(0, np),
            KOKKOS_LAMBDA(const Index i) {
                if (erank[i] != rank) return;
                Scalar xiLocal[3] = {Scalar(0), Scalar(0), Scalar(0)};
                Scalar shape[detail::max_element_nodes];
                Scalar phi[detail::max_element_nodes];
                for (Index d = 0; d < nd; ++d)
                    xiLocal[d] = xi[d + nd * i];
                mesh.shape_values(xiLocal, shape, phi);
                for (Index c = 0; c < nc; ++c) {
                    Scalar value = 0;
                    for (Index a = 0; a < mesh.npe; ++a)
                        value += shape[a] *
                            source[a + mesh.npe * c + mesh.npe * nc * e[i]];
                    target[c + nc * i] = value;
                }
            });
    }

#ifdef HAVE_MPI
    void exchange_counts(
        const std::vector<Index>& send,
        std::vector<Index>& receive) const
    {
        MPI_Alltoall(send.data(), 1, MpiType<Index>::value(),
                     receive.data(), 1, MpiType<Index>::value(), comm_);
    }

    void exchange_soa(
        const index_view& sendInt,
        Index sendCapacity,
        index_view& recvInt,
        Index recvCapacity,
        Index intFields,
        const scalar_view& sendReal,
        scalar_view& recvReal,
        Index realFields,
        int tagBase)
    {
        requests_.clear();
        requests_.reserve(static_cast<std::size_t>(
            2 * nproc_ * (intFields + realFields)));
        for (Index rank = 0; rank < nproc_; ++rank) {
            if (hostSendCount_[static_cast<std::size_t>(rank)] >
                    static_cast<Index>(std::numeric_limits<int>::max()) ||
                hostRecvCount_[static_cast<std::size_t>(rank)] >
                    static_cast<Index>(std::numeric_limits<int>::max()))
                throw MeshTransferError("MPI record count exceeds the int count range");
        }
        for (Index rank = 0; rank < nproc_; ++rank) {
            const Index count = hostRecvCount_[static_cast<std::size_t>(rank)];
            const Index offset = hostRecvOffset_[static_cast<std::size_t>(rank)];
            for (Index field = 0; field < intFields && count > 0; ++field) {
                requests_.push_back(MPI_REQUEST_NULL);
                MPI_Irecv(recvInt.data() + field * recvCapacity + offset,
                          static_cast<int>(count), MpiType<Index>::value(),
                          static_cast<int>(rank), tagBase + static_cast<int>(field),
                          comm_, &requests_.back());
            }
            for (Index field = 0; field < realFields && count > 0; ++field) {
                requests_.push_back(MPI_REQUEST_NULL);
                MPI_Irecv(recvReal.data() + field * recvCapacity + offset,
                          static_cast<int>(count), MpiType<Scalar>::value(),
                          static_cast<int>(rank),
                          tagBase + static_cast<int>(intFields + field),
                          comm_, &requests_.back());
            }
        }
        for (Index rank = 0; rank < nproc_; ++rank) {
            const Index count = hostSendCount_[static_cast<std::size_t>(rank)];
            const Index offset = hostSendOffset_[static_cast<std::size_t>(rank)];
            for (Index field = 0; field < intFields && count > 0; ++field) {
                requests_.push_back(MPI_REQUEST_NULL);
                MPI_Isend(sendInt.data() + field * sendCapacity + offset,
                          static_cast<int>(count), MpiType<Index>::value(),
                          static_cast<int>(rank), tagBase + static_cast<int>(field),
                          comm_, &requests_.back());
            }
            for (Index field = 0; field < realFields && count > 0; ++field) {
                requests_.push_back(MPI_REQUEST_NULL);
                MPI_Isend(sendReal.data() + field * sendCapacity + offset,
                          static_cast<int>(count), MpiType<Scalar>::value(),
                          static_cast<int>(rank),
                          tagBase + static_cast<int>(intFields + field),
                          comm_, &requests_.back());
            }
        }
        if (!requests_.empty())
            MPI_Waitall(static_cast<int>(requests_.size()), requests_.data(),
                        MPI_STATUSES_IGNORE);
    }

    Index pack_owner_queries(
        const Scalar* x,
        const Index* erank,
        const Index* e,
        const Scalar* xi,
        Index np)
    {
        Kokkos::deep_copy(sendCount_, Index(0));
        Index* counts = sendCount_.data();
        const Index* state = pointState_.data();
        const Index nproc = nproc_;
        Kokkos::parallel_for(
            "MeshTransfer::CountOwnerQueries", range_policy<Index>(0, np),
            KOKKOS_LAMBDA(const Index i) {
                if (state[i] == 0 && erank[i] >= 0 && erank[i] < nproc)
                    Kokkos::atomic_inc(&counts[erank[i]]);
            });
        copy_counts_to_host(sendCount_, hostSendCount_);
        const Index nsend = compute_offsets(hostSendCount_, hostSendOffset_);
        reserve_send_queries(nsend);
        set_device_offsets(hostSendOffset_);
        Index* queryInt = sendQueryInt_.data();
        Scalar* queryReal = sendQueryReal_.data();
        Index* cursor = cursor_.data();
        const Index* offsets = sendOffset_.data();
        const Index capacity = sendQueryCapacity_;
        const Index nd = mesh_.spatial_dimension();
        const Index ownerRank = myrank_;
        Kokkos::parallel_for(
            "MeshTransfer::PackOwnerQueries", range_policy<Index>(0, np),
            KOKKOS_LAMBDA(const Index i) {
                if (state[i] != 0) return;
                const Index destination = erank[i];
                const Index q = offsets[destination] +
                    Kokkos::atomic_fetch_add(&cursor[destination], Index(1));
                queryInt[q] = ownerRank;
                queryInt[capacity + q] = i;
                queryInt[2 * capacity + q] = e[i];
                for (Index d = 0; d < nd; ++d) {
                    queryReal[d * capacity + q] = x[d + nd * i];
                    queryReal[(nd + d) * capacity + q] = xi[d + nd * i];
                }
            });
        Kokkos::fence();
        return nsend;
    }

    Index exchange_queries(Index)
    {
        exchange_counts(hostSendCount_, hostRecvCount_);
        const Index nrecv = compute_offsets(hostRecvCount_, hostRecvOffset_);
        reserve_recv_queries(nrecv);
        exchange_soa(sendQueryInt_, sendQueryCapacity_,
                     recvQueryInt_, recvQueryCapacity_, 3,
                     sendQueryReal_, recvQueryReal_,
                     2 * mesh_.spatial_dimension(), 3100);
        return nrecv;
    }

    void process_received_queries(Index nquery)
    {
        const auto mesh = mesh_.device_view();
        const Index capacity = recvQueryCapacity_;
        const Index nd = mesh.nd;
        Index* queryInt = recvQueryInt_.data();
        Scalar* queryReal = recvQueryReal_.data();
        Index* status = queryStatus_.data();
        Index* destination = queryDest_.data();
        Index* active = queryActiveIds_.data();
        Kokkos::parallel_for(
            "MeshTransfer::InitializeWorkerQueries", range_policy<Index>(0, nquery),
            KOKKOS_LAMBDA(const Index q) {
                active[q] = q;
                status[q] = static_cast<Index>(LOCAL_NEXT);
                destination[q] = mesh.myrank;
            });
        Index nactive = nquery;
        index_view current = queryActiveIds_;
        index_view next = queryActiveIdsNext_;
        const Index maxHops = 4 * mesh.ne + 8;
        Index hops = 0;
        Kokkos::View<Index, memory_space> error("meshtransfer_worker_error");
        Kokkos::deep_copy(error, Index(0));
        while (nactive > 0 && hops < maxHops) {
            const Index* activeIds = current.data();
            Kokkos::parallel_for(
                "MeshTransfer::ProcessWorkerQueries",
                range_policy<Index>(0, nactive),
                KOKKOS_LAMBDA(const Index k) {
                    const Index q = activeIds[k];
                    const Index elem = queryInt[2 * capacity + q];
                    if (elem < 0 || elem >= mesh.ne) {
                        Kokkos::atomic_exchange(error.data(), Index(1));
                        status[q] = static_cast<Index>(OUTSIDE);
                        return;
                    }
                    Scalar xLocal[3] = {Scalar(0), Scalar(0), Scalar(0)};
                    Scalar xiLocal[3] = {Scalar(0), Scalar(0), Scalar(0)};
                    for (Index d = 0; d < nd; ++d) {
                        xLocal[d] = queryReal[d * capacity + q];
                        xiLocal[d] = queryReal[(nd + d) * capacity + q];
                    }
                    Scalar shape[detail::max_element_nodes];
                    Scalar dshape[3 * detail::max_element_nodes];
                    Scalar phi[detail::max_element_nodes];
                    Scalar dphi[3 * detail::max_element_nodes];
                    const auto inverseStatus = mesh.inverse_map(
                        xLocal, elem, xiLocal, shape, dshape, phi, dphi);
                    for (Index d = 0; d < nd; ++d)
                        queryReal[(nd + d) * capacity + q] = xiLocal[d];
                    if (inverseStatus != InverseMapStatus::Converged) {
                        Kokkos::atomic_exchange(
                            error.data(),
                            inverseStatus == InverseMapStatus::Singular ? Index(2) : Index(3));
                        status[q] = static_cast<Index>(OUTSIDE);
                        return;
                    }
                    if (mesh.inside(xiLocal)) {
                        destination[q] = queryInt[q];
                        status[q] = static_cast<Index>(FOUND);
                        return;
                    }
                    const Index face = mesh.exit_face(xiLocal);
                    Index nextRank = -1;
                    Index nextElem = -1;
                    if (!mesh.neighbor(elem, face, nextRank, nextElem)) {
                        destination[q] = queryInt[q];
                        status[q] = static_cast<Index>(OUTSIDE);
                    }
                    else if (nextRank == mesh.myrank) {
                        queryInt[2 * capacity + q] = nextElem;
                        status[q] = static_cast<Index>(LOCAL_NEXT);
                    }
                    else {
                        queryInt[2 * capacity + q] = nextElem;
                        destination[q] = nextRank;
                        status[q] = static_cast<Index>(REMOTE);
                    }
                });
            Kokkos::fence();
            auto errorHost =
                Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), error);
            if (errorHost() != 0) {
                if (errorHost() == 1)
                    throw MeshTransferError("distributed query contains an invalid element");
                throw MeshTransferError(errorHost() == 2
                    ? "distributed point location encountered a singular geometric Jacobian"
                    : "distributed inverse mapping failed to converge");
            }
            Index nextCount = 0;
            const Index* currentIds = current.data();
            Index* nextIds = next.data();
            Kokkos::parallel_scan(
                "MeshTransfer::CompactWorkerQueries",
                range_policy<Index>(0, nactive),
                KOKKOS_LAMBDA(const Index k, Index& offset, const bool final) {
                    const Index q = currentIds[k];
                    if (status[q] == static_cast<Index>(LOCAL_NEXT)) {
                        if (final) nextIds[offset] = q;
                        ++offset;
                    }
                }, nextCount);
            std::swap(current, next);
            nactive = nextCount;
            ++hops;
        }
        if (nactive > 0)
            throw MeshTransferError("distributed local walk exceeded its hop limit");
    }

    void pack_worker_outputs(Index nquery, Index& nforward, Index& nresult)
    {
        const Index* status = queryStatus_.data();
        const Index* destination = queryDest_.data();
        Index* counts = sendCount_.data();

        Kokkos::deep_copy(sendCount_, Index(0));
        Kokkos::parallel_for(
            "MeshTransfer::CountForwardQueries", range_policy<Index>(0, nquery),
            KOKKOS_LAMBDA(const Index q) {
                if (status[q] == static_cast<Index>(REMOTE))
                    Kokkos::atomic_inc(&counts[destination[q]]);
            });
        copy_counts_to_host(sendCount_, forwardSendCount_);
        nforward = compute_offsets(forwardSendCount_, forwardSendOffset_);

        Kokkos::deep_copy(sendCount_, Index(0));
        Kokkos::parallel_for(
            "MeshTransfer::CountLocationResults", range_policy<Index>(0, nquery),
            KOKKOS_LAMBDA(const Index q) {
                if (status[q] == static_cast<Index>(FOUND) ||
                    status[q] == static_cast<Index>(OUTSIDE))
                    Kokkos::atomic_inc(&counts[destination[q]]);
            });
        copy_counts_to_host(sendCount_, resultSendCount_);
        nresult = compute_offsets(resultSendCount_, resultSendOffset_);

        reserve_send_queries(nforward);
        reserve_send_results(nresult, mesh_.spatial_dimension());
        const Index inputCapacity = recvQueryCapacity_;
        const Index queryCapacity = sendQueryCapacity_;
        const Index resultCapacity = sendResultCapacity_;
        const Index nd = mesh_.spatial_dimension();
        const Index* inputInt = recvQueryInt_.data();
        const Scalar* inputReal = recvQueryReal_.data();
        Index* outputInt = sendQueryInt_.data();
        Scalar* outputReal = sendQueryReal_.data();
        Index* resultInt = sendResultInt_.data();
        Scalar* resultReal = sendResultReal_.data();

        set_device_offsets(forwardSendOffset_);
        const Index* offsets = sendOffset_.data();
        Index* cursor = cursor_.data();
        Kokkos::parallel_for(
            "MeshTransfer::PackForwardQueries", range_policy<Index>(0, nquery),
            KOKKOS_LAMBDA(const Index q) {
                if (status[q] != static_cast<Index>(REMOTE)) return;
                const Index rank = destination[q];
                const Index out = offsets[rank] +
                    Kokkos::atomic_fetch_add(&cursor[rank], Index(1));
                for (Index field = 0; field < 3; ++field)
                    outputInt[field * queryCapacity + out] =
                        inputInt[field * inputCapacity + q];
                for (Index field = 0; field < 2 * nd; ++field)
                    outputReal[field * queryCapacity + out] =
                        inputReal[field * inputCapacity + q];
            });

        set_device_offsets(resultSendOffset_);
        offsets = sendOffset_.data();
        cursor = cursor_.data();
        const auto mesh = mesh_.device_view();
        Kokkos::parallel_for(
            "MeshTransfer::PackLocationResults", range_policy<Index>(0, nquery),
            KOKKOS_LAMBDA(const Index q) {
                if (status[q] != static_cast<Index>(FOUND) &&
                    status[q] != static_cast<Index>(OUTSIDE)) return;
                const Index rank = destination[q];
                const Index out = offsets[rank] +
                    Kokkos::atomic_fetch_add(&cursor[rank], Index(1));
                resultInt[out] = inputInt[inputCapacity + q];
                resultInt[resultCapacity + out] = mesh.myrank;
                resultInt[2 * resultCapacity + out] =
                    inputInt[2 * inputCapacity + q];
                for (Index d = 0; d < nd; ++d)
                    resultReal[d * resultCapacity + out] =
                        inputReal[(nd + d) * inputCapacity + q];
            });
        Kokkos::fence();
    }

    Index exchange_location_results()
    {
        hostSendCount_ = resultSendCount_;
        hostSendOffset_ = resultSendOffset_;
        exchange_counts(hostSendCount_, hostRecvCount_);
        const Index nrecv = compute_offsets(hostRecvCount_, hostRecvOffset_);
        reserve_recv_results(nrecv, mesh_.spatial_dimension());
        exchange_soa(sendResultInt_, sendResultCapacity_,
                     recvResultInt_, recvResultCapacity_, 3,
                     sendResultReal_, recvResultReal_,
                     mesh_.spatial_dimension(), 3200);
        return nrecv;
    }

    void update_location_results(
        Index nresult, Index* erank, Index* e, Scalar* xi, Index np)
    {
        const Index capacity = recvResultCapacity_;
        const Index nd = mesh_.spatial_dimension();
        const Index* resultInt = recvResultInt_.data();
        const Scalar* resultReal = recvResultReal_.data();
        Index* state = pointState_.data();
        Kokkos::View<Index, memory_space> error("meshtransfer_result_error");
        Kokkos::deep_copy(error, Index(0));
        Kokkos::parallel_for(
            "MeshTransfer::UpdateLocationResults",
            range_policy<Index>(0, nresult),
            KOKKOS_LAMBDA(const Index q) {
                const Index i = resultInt[q];
                if (i < 0 || i >= np) {
                    Kokkos::atomic_exchange(error.data(), Index(1));
                    return;
                }
                erank[i] = resultInt[capacity + q];
                e[i] = resultInt[2 * capacity + q];
                for (Index d = 0; d < nd; ++d)
                    xi[d + nd * i] = resultReal[d * capacity + q];
                state[i] = 1;
            });
        auto errorHost =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), error);
        if (errorHost() != 0)
            throw MeshTransferError("location result contains an invalid owner index");
    }

    Index global_completed(Index np) const
    {
        const Index* state = pointState_.data();
        Index local = 0;
        Kokkos::parallel_reduce(
            "MeshTransfer::CountCompletedOwners", range_policy<Index>(0, np),
            KOKKOS_LAMBDA(const Index i, Index& sum) {
                if (state[i] != 0) ++sum;
            }, local);
        Index global = 0;
        MPI_Allreduce(&local, &global, 1, MpiType<Index>::value(), MPI_SUM, comm_);
        return global;
    }

    Index pack_evaluation_requests(
        const Index* erank,
        const Index* e,
        const Scalar* xi,
        Index np)
    {
        Kokkos::deep_copy(sendCount_, Index(0));
        Index* counts = sendCount_.data();
        const Index rank = myrank_;
        Kokkos::parallel_for(
            "MeshTransfer::CountEvaluationRequests", range_policy<Index>(0, np),
            KOKKOS_LAMBDA(const Index i) {
                if (erank[i] != rank)
                    Kokkos::atomic_inc(&counts[erank[i]]);
            });
        copy_counts_to_host(sendCount_, hostSendCount_);
        const Index nsend = compute_offsets(hostSendCount_, hostSendOffset_);
        reserve_send_queries(nsend);
        set_device_offsets(hostSendOffset_);
        Index* requestInt = sendQueryInt_.data();
        Scalar* requestReal = sendQueryReal_.data();
        const Index* offsets = sendOffset_.data();
        Index* cursor = cursor_.data();
        const Index capacity = sendQueryCapacity_;
        const Index nd = mesh_.spatial_dimension();
        const Index ownerRank = myrank_;
        Kokkos::parallel_for(
            "MeshTransfer::PackEvaluationRequests", range_policy<Index>(0, np),
            KOKKOS_LAMBDA(const Index i) {
                if (erank[i] == ownerRank) return;
                const Index destination = erank[i];
                const Index q = offsets[destination] +
                    Kokkos::atomic_fetch_add(&cursor[destination], Index(1));
                requestInt[q] = ownerRank;
                requestInt[capacity + q] = i;
                requestInt[2 * capacity + q] = e[i];
                for (Index d = 0; d < nd; ++d)
                    requestReal[d * capacity + q] = xi[d + nd * i];
            });
        Kokkos::fence();
        return nsend;
    }

    Index exchange_evaluation_requests()
    {
        exchange_counts(hostSendCount_, hostRecvCount_);
        const Index nrecv = compute_offsets(hostRecvCount_, hostRecvOffset_);
        reserve_recv_queries(nrecv);
        exchange_soa(sendQueryInt_, sendQueryCapacity_,
                     recvQueryInt_, recvQueryCapacity_, 3,
                     sendQueryReal_, recvQueryReal_,
                     mesh_.spatial_dimension(), 3300);
        return nrecv;
    }

    Index evaluate_received_requests(const Scalar* source, Index nrequest)
    {
        Kokkos::deep_copy(sendCount_, Index(0));
        Index* counts = sendCount_.data();
        const Index* requestInt = recvQueryInt_.data();
        const Index inputCapacity = recvQueryCapacity_;
        Kokkos::parallel_for(
            "MeshTransfer::CountEvaluationResults",
            range_policy<Index>(0, nrequest),
            KOKKOS_LAMBDA(const Index q) {
                Kokkos::atomic_inc(&counts[requestInt[q]]);
            });
        copy_counts_to_host(sendCount_, hostSendCount_);
        const Index nsend = compute_offsets(hostSendCount_, hostSendOffset_);
        reserve_send_results(nsend, nc_);
        set_device_offsets(hostSendOffset_);

        const auto mesh = mesh_.device_view();
        const Scalar* requestReal = recvQueryReal_.data();
        Index* resultInt = sendResultInt_.data();
        Scalar* resultReal = sendResultReal_.data();
        const Index resultCapacity = sendResultCapacity_;
        const Index nd = mesh.nd;
        const Index nc = nc_;
        const Index* offsets = sendOffset_.data();
        Index* cursor = cursor_.data();
        Kokkos::parallel_for(
            "MeshTransfer::EvaluateReceivedRequests",
            range_policy<Index>(0, nrequest),
            KOKKOS_LAMBDA(const Index q) {
                const Index owner = requestInt[q];
                const Index out = offsets[owner] +
                    Kokkos::atomic_fetch_add(&cursor[owner], Index(1));
                const Index elem = requestInt[2 * inputCapacity + q];
                Scalar xiLocal[3] = {Scalar(0), Scalar(0), Scalar(0)};
                Scalar shape[detail::max_element_nodes];
                Scalar phi[detail::max_element_nodes];
                for (Index d = 0; d < nd; ++d)
                    xiLocal[d] = requestReal[d * inputCapacity + q];
                mesh.shape_values(xiLocal, shape, phi);
                resultInt[out] = requestInt[inputCapacity + q];
                for (Index c = 0; c < nc; ++c) {
                    Scalar value = 0;
                    for (Index a = 0; a < mesh.npe; ++a)
                        value += shape[a] *
                            source[a + mesh.npe * c + mesh.npe * nc * elem];
                    resultReal[c * resultCapacity + out] = value;
                }
            });
        Kokkos::fence();
        return nsend;
    }

    Index exchange_evaluation_results()
    {
        exchange_counts(hostSendCount_, hostRecvCount_);
        const Index nrecv = compute_offsets(hostRecvCount_, hostRecvOffset_);
        reserve_recv_results(nrecv, nc_);
        exchange_soa(sendResultInt_, sendResultCapacity_,
                     recvResultInt_, recvResultCapacity_, 1,
                     sendResultReal_, recvResultReal_, nc_, 3400);
        return nrecv;
    }

    void update_evaluation_results(Scalar* target, Index nresult, Index np)
    {
        const Index* resultInt = recvResultInt_.data();
        const Scalar* resultReal = recvResultReal_.data();
        const Index capacity = recvResultCapacity_;
        const Index nc = nc_;
        Kokkos::View<Index, memory_space> error("meshtransfer_evaluation_result_error");
        Kokkos::deep_copy(error, Index(0));
        Kokkos::parallel_for(
            "MeshTransfer::UpdateEvaluationResults",
            range_policy<Index>(0, nresult),
            KOKKOS_LAMBDA(const Index q) {
                const Index i = resultInt[q];
                if (i < 0 || i >= np) {
                    Kokkos::atomic_exchange(error.data(), Index(1));
                    return;
                }
                for (Index c = 0; c < nc; ++c)
                    target[c + nc * i] = resultReal[c * capacity + q];
            });
        auto errorHost =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), error);
        if (errorHost() != 0)
            throw MeshTransferError("evaluation result contains an invalid owner index");
    }

    void evaluate_remote(
        const Scalar* source,
        Scalar* target,
        const Index* erank,
        const Index* e,
        const Scalar* xi,
        Index np)
    {
        (void)pack_evaluation_requests(erank, e, xi, np);
        const Index nrequest = exchange_evaluation_requests();
        (void)evaluate_received_requests(source, nrequest);
        const Index nresult = exchange_evaluation_results();
        update_evaluation_results(target, nresult, np);
    }

    void distributed_locate(
        const Scalar* x,
        Index* erank,
        Index* e,
        Scalar* xi,
        Index np,
        Index)
    {
        Index globalTargets = 0;
        MPI_Allreduce(&np, &globalTargets, 1, MpiType<Index>::value(), MPI_SUM, comm_);
        if (global_completed(np) == globalTargets) return;

        Index nsend = pack_owner_queries(x, erank, e, xi, np);
        const Index maxRounds = 16 * nproc_ + 64;
        for (Index round = 0; round < maxRounds; ++round) {
            const Index nrecv = exchange_queries(nsend);
            process_received_queries(nrecv);
            Index nforward = 0;
            Index nresults = 0;
            pack_worker_outputs(nrecv, nforward, nresults);
            (void)nresults;
            const Index receivedResults = exchange_location_results();
            update_location_results(receivedResults, erank, e, xi, np);
            if (global_completed(np) == globalTargets) return;
            hostSendCount_ = forwardSendCount_;
            hostSendOffset_ = forwardSendOffset_;
            nsend = nforward;
            Index globalMessages = 0;
            MPI_Allreduce(&nsend, &globalMessages, 1,
                          MpiType<Index>::value(), MPI_SUM, comm_);
            if (globalMessages == 0)
                throw MeshTransferError("distributed point location stalled before completion");
        }
        throw MeshTransferError(
            "distributed point location exceeded its communication-round limit");
    }
#endif

    Comm comm_ = null_comm;
    int myrankInt_ = 0;
    int nprocInt_ = 1;
    Index myrank_ = 0;
    Index nproc_ = 1;
    mesh_type mesh_;
    Index nc_ = 0;

    const Scalar* x_ = nullptr;
    Index* erank_ = nullptr;
    Index* e_ = nullptr;
    Scalar* xi_ = nullptr;
    Index np_ = -1;

    Index pointCapacity_ = 0;
    index_view activeIds_;
    index_view activeIdsNext_;
    index_view status_;
    index_view pointState_;
    index_view sendCount_;
    index_view recvCount_;
    index_view sendOffset_;
    index_view recvOffset_;
    index_view cursor_;

    Index sendQueryCapacity_ = 0;
    Index recvQueryCapacity_ = 0;
    Index sendResultCapacity_ = 0;
    Index recvResultCapacity_ = 0;
    Index sendResultRealWidth_ = 0;
    Index recvResultRealWidth_ = 0;
    index_view sendQueryInt_;
    index_view recvQueryInt_;
    scalar_view sendQueryReal_;
    scalar_view recvQueryReal_;
    index_view sendResultInt_;
    index_view recvResultInt_;
    scalar_view sendResultReal_;
    scalar_view recvResultReal_;
    index_view queryDest_;
    index_view queryActiveIds_;
    index_view queryActiveIdsNext_;
    index_view queryStatus_;
    std::vector<Index> hostSendCount_;
    std::vector<Index> hostRecvCount_;
    std::vector<Index> hostSendOffset_;
    std::vector<Index> hostRecvOffset_;
    std::vector<Index> forwardSendCount_;
    std::vector<Index> forwardSendOffset_;
    std::vector<Index> resultSendCount_;
    std::vector<Index> resultSendOffset_;
#ifdef HAVE_MPI
    std::vector<MPI_Request> requests_;
#endif
};

} // namespace exasim::meshtransfer
