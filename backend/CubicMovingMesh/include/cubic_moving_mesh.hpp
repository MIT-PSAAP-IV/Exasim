#pragma once

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace exasim::cubicmovingmesh {

using DefaultScalar = double;
using DefaultIndex = std::int32_t;

class CubicMovingMeshError : public std::runtime_error {
public:
    explicit CubicMovingMeshError(const std::string& message)
        : std::runtime_error(message)
    {
    }
};

template <class Scalar = DefaultScalar, class Index = DefaultIndex>
class CubicMovingMesh {
public:
    static_assert(std::is_floating_point_v<Scalar>,
                  "CubicMovingMesh Scalar must be a floating-point type");
    static_assert(std::is_integral_v<Index>,
                  "CubicMovingMesh Index must be an integral type");

    using scalar_type = Scalar;
    using index_type = Index;
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = typename execution_space::memory_space;
    using scalar_view =
        Kokkos::View<Scalar*, Kokkos::LayoutRight, memory_space>;
    using range_policy =
        Kokkos::RangePolicy<execution_space, Kokkos::IndexType<Index>>;

    CubicMovingMesh(Index npe, Index nd, Index ne)
        : npe_(npe), nd_(nd), ne_(ne)
    {
        if (npe_ <= 0)
            throw CubicMovingMeshError("npe must be positive");
        if (nd_ != Index(2) && nd_ != Index(3))
            throw CubicMovingMeshError("nd must be 2 or 3");
        if (ne_ <= 0)
            throw CubicMovingMeshError("ne must be positive");
        nc_ = nd_ + nd_ * nd_;
        if (npe_ > std::numeric_limits<Index>::max() / nc_ ||
            npe_ * nc_ > std::numeric_limits<Index>::max() / ne_)
            throw CubicMovingMeshError("packed mesh extent exceeds the Index range");

        const Index nmesh = mesh_extent();
        phi0_ = scalar_view("cubic_mesh_phi0", nmesh);
        v0_ = scalar_view("cubic_mesh_v0", nmesh);
        phi1_ = scalar_view("cubic_mesh_phi1", nmesh);
        phi2_ = scalar_view("cubic_mesh_phi2", nmesh);
        a_ = scalar_view("cubic_mesh_time_functions", Index(4));
        da_ = scalar_view("cubic_mesh_time_derivatives", Index(4));
    }

    ~CubicMovingMesh() = default;
    CubicMovingMesh(const CubicMovingMesh&) = delete;
    CubicMovingMesh& operator=(const CubicMovingMesh&) = delete;
    CubicMovingMesh(CubicMovingMesh&&) noexcept = default;
    CubicMovingMesh& operator=(CubicMovingMesh&&) noexcept = default;

    Index nodes_per_element() const noexcept { return npe_; }
    Index spatial_dimension() const noexcept { return nd_; }
    Index num_elements() const noexcept { return ne_; }
    Index num_components() const noexcept { return nc_; }
    Index coordinate_extent() const noexcept { return npe_ * nd_ * ne_; }
    Index mesh_extent() const noexcept { return npe_ * nc_ * ne_; }
    Index vector_extent() const noexcept { return npe_ * nd_ * ne_; }
    Index matrix_extent() const noexcept { return npe_ * nd_ * nd_ * ne_; }
    Index scalar_extent() const noexcept { return npe_ * ne_; }

    Scalar slab_initial_time() const noexcept { return tn_; }
    Scalar slab_duration() const noexcept { return dt_; }
    Scalar first_stage_time() const noexcept { return c1_; }
    Scalar second_stage_time() const noexcept { return c2_; }

    scalar_view phi0() const noexcept { return phi0_; }
    scalar_view v0() const noexcept { return v0_; }
    scalar_view phi1() const noexcept { return phi1_; }
    scalar_view phi2() const noexcept { return phi2_; }
    scalar_view time_functions() const noexcept { return a_; }
    scalar_view time_function_derivatives() const noexcept { return da_; }

    KOKKOS_INLINE_FUNCTION
    Index packed_index(Index p, Index c, Index e) const noexcept
    {
        return p + npe_ * c + npe_ * nc_ * e;
    }

    KOKKOS_INLINE_FUNCTION
    Index coordinate_index(Index p, Index j, Index e) const noexcept
    {
        return p + npe_ * j + npe_ * nd_ * e;
    }

    KOKKOS_INLINE_FUNCTION
    Index matrix_channel(Index i, Index j) const noexcept
    {
        return nd_ + i + nd_ * j;
    }

    KOKKOS_INLINE_FUNCTION
    Index matrix_output_index(Index p, Index i, Index j, Index e) const noexcept
    {
        return p + npe_ * (i + nd_ * j) + npe_ * nd_ * nd_ * e;
    }

    KOKKOS_INLINE_FUNCTION
    Index scalar_output_index(Index p, Index e) const noexcept
    {
        return p + npe_ * e;
    }

    void setTimeSlab(Scalar tn, Scalar dt, Scalar c1, Scalar c2)
    {
        if (!std::isfinite(tn) || !std::isfinite(dt) ||
            !std::isfinite(c1) || !std::isfinite(c2))
            throw CubicMovingMeshError("time-slab parameters must be finite");
        if (!(dt > Scalar(0)))
            throw CubicMovingMeshError("dt must be positive");
        if (!(c1 > Scalar(0) && c1 < c2))
            throw CubicMovingMeshError("time stages must satisfy 0 < c1 < c2");
        tn_ = tn;
        dt_ = dt;
        c1_ = c1;
        c2_ = c2;
        timeSlabSet_ = true;
    }

    void calculateTimeFunctions(Scalar t)
    {
        validate_evaluation_time(t);
        const Scalar tau = t - tn_;
        const Scalar dt = dt_;
        const Scalar c1 = c1_;
        const Scalar c2 = c2_;
        const auto a = a_;
        const auto da = da_;

        Kokkos::parallel_for(
            "CubicMovingMesh::CalculateTimeFunctions", range_policy(0, Index(1)),
            KOKKOS_LAMBDA(const Index) {
                const Scalar c1dt = c1 * dt;
                const Scalar c2dt = c2 * dt;
                const Scalar dt2 = dt * dt;
                const Scalar dt3 = dt2 * dt;
                const Scalar tau2 = tau * tau;
                const Scalar c1sq = c1 * c1;
                const Scalar c2sq = c2 * c2;
                const Scalar dc = c2 - c1;

                a(0) = ((c1dt - tau) * (c2dt - tau) *
                        (c1 * c2 * dt + (c1 + c2) * tau)) /
                       (c1sq * c2sq * dt3);
                a(1) = tau * (c1dt - tau) * (c2dt - tau) /
                       (c1 * c2 * dt2);
                a(2) = tau2 * (c2dt - tau) /
                       (c1sq * dc * dt3);
                a(3) = tau2 * (tau - c1dt) /
                       (c2sq * dc * dt3);

                da(0) = tau *
                        (-Scalar(2) * c1sq * dt
                         - Scalar(2) * c1 * c2 * dt
                         - Scalar(2) * c2sq * dt
                         + Scalar(3) * (c1 + c2) * tau) /
                        (c1sq * c2sq * dt3);
                da(1) = (c1 * c2 * dt2
                         - Scalar(2) * (c1 + c2) * dt * tau
                         + Scalar(3) * tau2) /
                        (c1 * c2 * dt2);
                da(2) = tau * (Scalar(2) * c2dt - Scalar(3) * tau) /
                        (c1sq * dc * dt3);
                da(3) = tau * (Scalar(3) * tau - Scalar(2) * c1dt) /
                        (c2sq * dc * dt3);
            });
    }

    void updateMeshDeformation0(const Scalar* X, const Scalar* d0)
    {
        update_deformation(phi0_, X, d0,
                           "CubicMovingMesh::UpdateMeshDeformation0");
    }

    void updateMeshVelocity0(const Scalar* v0)
    {
        validate_pointer(v0, "v0 input");
        const Index nmesh = mesh_extent();
        const auto output = v0_;
        Kokkos::parallel_for(
            "CubicMovingMesh::UpdateMeshVelocity0", range_policy(0, nmesh),
            KOKKOS_LAMBDA(const Index k) { output(k) = v0[k]; });
    }

    void updateMeshDeformation1(const Scalar* X, const Scalar* d1)
    {
        update_deformation(phi1_, X, d1,
                           "CubicMovingMesh::UpdateMeshDeformation1");
    }

    void updateMeshDeformation2(const Scalar* X, const Scalar* d2)
    {
        update_deformation(phi2_, X, d2,
                           "CubicMovingMesh::UpdateMeshDeformation2");
    }

    void initializeFirstSlabMesh(const Scalar* X)
    {
        validate_pointer(X, "coordinate input");
        const Index npe = npe_;
        const Index nd = nd_;
        const Index nc = nc_;
        const Index nmesh = mesh_extent();
        const auto phi0 = phi0_;
        const auto v0 = v0_;
        const auto phi1 = phi1_;
        const auto phi2 = phi2_;
        Kokkos::parallel_for(
            "CubicMovingMesh::InitializeFirstSlabMesh", range_policy(0, nmesh),
            KOKKOS_LAMBDA(const Index k) {
                const Index p = k % npe;
                const Index c = (k / npe) % nc;
                const Index e = k / (npe * nc);
                Scalar value;
                if (c < nd) {
                    const Index xk = p + npe * c + npe * nd * e;
                    value = X[xk];
                }
                else {
                    const Index q = c - nd;
                    value = q % nd == q / nd ? Scalar(1) : Scalar(0);
                }
                phi0(k) = value;
                phi1(k) = value;
                phi2(k) = value;
                v0(k) = Scalar(0);
            });
    }

    void initializeFirstSlabMesh(const Scalar* X, const Scalar* d)
    {
        validate_pointer(X, "coordinate input");
        validate_pointer(d, "mesh deformation");
        const Index npe = npe_;
        const Index nd = nd_;
        const Index nc = nc_;
        const Index nmesh = mesh_extent();
        const auto phi0 = phi0_;
        const auto v0 = v0_;
        const auto phi1 = phi1_;
        const auto phi2 = phi2_;
        Kokkos::parallel_for(
            "CubicMovingMesh::InitializeFirstSlabMeshDeformed",
            range_policy(0, nmesh), KOKKOS_LAMBDA(const Index k) {
                const Index c = (k / npe) % nc;
                Scalar value;
                if (c < nd) {
                    const Index p = k % npe;
                    const Index e = k / (npe * nc);
                    const Index xk = p + npe * c + npe * nd * e;
                    value = X[xk] + d[k];
                }
                else {
                    const Index matrixComponent = c - nd;
                    const Index i = matrixComponent % nd;
                    const Index j = matrixComponent / nd;
                    value = (i == j ? Scalar(1) : Scalar(0)) - d[k];
                }
                phi0(k) = value;
                phi1(k) = value;
                phi2(k) = value;
                v0(k) = Scalar(0);
            });
    }

    void updateMeshDeformationAndVelocity0()
    {
        if (!timeSlabSet_)
            throw CubicMovingMeshError(
                "setTimeSlab must be called before slab rollover");
        calculateTimeFunctions(tn_ + dt_);
        const Index nmesh = mesh_extent();
        const auto phi0 = phi0_;
        const auto v0 = v0_;
        const auto phi1 = phi1_;
        const auto phi2 = phi2_;
        const auto a = a_;
        const auto da = da_;
        Kokkos::parallel_for(
            "CubicMovingMesh::UpdateMeshDeformationAndVelocity0",
            range_policy(0, nmesh), KOKKOS_LAMBDA(const Index k) {
                const Scalar oldPhi0 = phi0(k);
                const Scalar oldV0 = v0(k);
                const Scalar oldPhi1 = phi1(k);
                const Scalar oldPhi2 = phi2(k);
                const Scalar newPhi0 = a(0) * oldPhi0 + a(1) * oldV0 +
                                       a(2) * oldPhi1 + a(3) * oldPhi2;
                const Scalar newV0 = da(0) * oldPhi0 + da(1) * oldV0 +
                                     da(2) * oldPhi1 + da(3) * oldPhi2;
                phi0(k) = newPhi0;
                v0(k) = newV0;
            });
    }

    void updateSlabMesh(
        const Scalar* X, const Scalar* d1, const Scalar* d2)
    {
        validate_pointer(X, "coordinate input");
        validate_pointer(d1, "first-stage deformation");
        validate_pointer(d2, "second-stage deformation");
        updateMeshDeformationAndVelocity0();
        updateMeshDeformation1(X, d1);
        updateMeshDeformation2(X, d2);
    }

    void calculateMeshDeformation(Scalar* phi, Scalar t)
    {
        evaluate_packed(phi, t, false,
                        "CubicMovingMesh::CalculateMeshDeformation");
    }

    void calculateMeshVelocity(Scalar* velocity, Scalar t)
    {
        evaluate_packed(velocity, t, true,
                        "CubicMovingMesh::CalculateMeshVelocity");
    }

    void calculateMeshDeformationGradient(Scalar* G, Scalar t)
    {
        evaluate_matrix(G, t, false,
                        "CubicMovingMesh::CalculateMeshDeformationGradient");
    }

    void calculateMeshDeformationGradientDerivative(
        Scalar* dGdt, Scalar t)
    {
        evaluate_matrix(
            dGdt, t, true,
            "CubicMovingMesh::CalculateMeshDeformationGradientDerivative");
    }

    void calculateMeshDeformationGradientInverse(Scalar* Ginv, Scalar t)
    {
        validate_pointer(Ginv, "inverse-gradient output");
        calculateTimeFunctions(t);
        const Index npe = npe_;
        const Index nd = nd_;
        const Index nc = nc_;
        const Index nnode = scalar_extent();
        const auto phi0 = phi0_;
        const auto v0 = v0_;
        const auto phi1 = phi1_;
        const auto phi2 = phi2_;
        const auto a = a_;
        const Scalar epsilon = std::numeric_limits<Scalar>::epsilon();
        Index invalid = 0;
        Kokkos::parallel_reduce(
            "CubicMovingMesh::CalculateMeshDeformationGradientInverse",
            range_policy(0, nnode),
            KOKKOS_LAMBDA(const Index node, Index& invalidCount) {
                const Index p = node % npe;
                const Index e = node / npe;
                Scalar G[9];
                Scalar inverse[9];
                load_matrix(p, e, npe, nd, nc, phi0, v0, phi1, phi2, a, G);
                const Scalar determinant = determinant_small(G, nd);
                const Scalar tolerance = determinant_tolerance(G, nd, epsilon);
                if (!(determinant > tolerance)) {
                    invalidCount += Index(1);
                    for (Index q = 0; q < nd * nd; ++q)
                        Ginv[p + npe * q + npe * nd * nd * e] = Scalar(0);
                    return;
                }
                inverse_small(G, nd, determinant, inverse);
                for (Index q = 0; q < nd * nd; ++q)
                    Ginv[p + npe * q + npe * nd * nd * e] = inverse[q];
            }, invalid);
        if (invalid != Index(0))
            throw CubicMovingMeshError(
                "deformation gradient is singular, near-singular, or inverted");
    }

    void calculateMeshJacobian(Scalar* J, Scalar t)
    {
        validate_pointer(J, "Jacobian output");
        calculateTimeFunctions(t);
        const Index npe = npe_;
        const Index nd = nd_;
        const Index nc = nc_;
        const Index nnode = scalar_extent();
        const auto phi0 = phi0_;
        const auto v0 = v0_;
        const auto phi1 = phi1_;
        const auto phi2 = phi2_;
        const auto a = a_;
        Kokkos::parallel_for(
            "CubicMovingMesh::CalculateMeshJacobian", range_policy(0, nnode),
            KOKKOS_LAMBDA(const Index node) {
                const Index p = node % npe;
                const Index e = node / npe;
                Scalar G[9];
                load_matrix(p, e, npe, nd, nc, phi0, v0, phi1, phi2, a, G);
                J[node] = determinant_small(G, nd);
            });
    }

    void calculateMeshJacobianDerivative(Scalar* dJdt, Scalar t)
    {
        validate_pointer(dJdt, "Jacobian-derivative output");
        calculateTimeFunctions(t);
        const Index npe = npe_;
        const Index nd = nd_;
        const Index nc = nc_;
        const Index nnode = scalar_extent();
        const auto phi0 = phi0_;
        const auto v0 = v0_;
        const auto phi1 = phi1_;
        const auto phi2 = phi2_;
        const auto a = a_;
        const auto da = da_;
        Kokkos::parallel_for(
            "CubicMovingMesh::CalculateMeshJacobianDerivative",
            range_policy(0, nnode), KOKKOS_LAMBDA(const Index node) {
                const Index p = node % npe;
                const Index e = node / npe;
                Scalar G[9];
                Scalar dGdt[9];
                Scalar cofactor[9];
                load_matrix(p, e, npe, nd, nc, phi0, v0, phi1, phi2, a, G);
                load_matrix(
                    p, e, npe, nd, nc, phi0, v0, phi1, phi2, da, dGdt);
                cofactor_small(G, nd, cofactor);
                Scalar value = Scalar(0);
                for (Index q = 0; q < nd * nd; ++q)
                    value += cofactor[q] * dGdt[q];
                dJdt[node] = value;
            });
    }

private:
    static void validate_pointer(const Scalar* pointer, const char* name)
    {
        if (pointer == nullptr)
            throw CubicMovingMeshError(std::string(name) + " cannot be null");
    }

    void update_deformation(
        scalar_view output, const Scalar* X, const Scalar* deformation,
        const char* kernelName)
    {
        validate_pointer(X, "coordinate input");
        validate_pointer(deformation, "mesh deformation");
        const Index npe = npe_;
        const Index nd = nd_;
        const Index nc = nc_;
        const Index nmesh = mesh_extent();
        Kokkos::parallel_for(
            kernelName, range_policy(0, nmesh), KOKKOS_LAMBDA(const Index k) {
                const Index p = k % npe;
                const Index c = (k / npe) % nc;
                const Index e = k / (npe * nc);
                if (c < nd) {
                    const Index xk = p + npe * c + npe * nd * e;
                    output(k) = X[xk] + deformation[k];
                }
                else {
                    const Index matrixComponent = c - nd;
                    const Index i = matrixComponent % nd;
                    const Index j = matrixComponent / nd;
                    output(k) = (i == j ? Scalar(1) : Scalar(0)) -
                                deformation[k];
                }
            });
    }

    KOKKOS_INLINE_FUNCTION
    static void load_matrix(
        Index p, Index e, Index npe, Index nd, Index nc,
        scalar_view phi0, scalar_view v0, scalar_view phi1, scalar_view phi2,
        scalar_view coefficients, Scalar* matrix) noexcept
    {
        for (Index q = 0; q < nd * nd; ++q) {
            const Index k = p + npe * (nd + q) + npe * nc * e;
            matrix[q] = coefficients(0) * phi0(k) +
                        coefficients(1) * v0(k) +
                        coefficients(2) * phi1(k) +
                        coefficients(3) * phi2(k);
        }
    }

    KOKKOS_INLINE_FUNCTION
    static Scalar determinant_small(const Scalar* G, Index nd) noexcept
    {
        if (nd == Index(2))
            return G[0] * G[3] - G[2] * G[1];
        return G[0] * (G[4] * G[8] - G[7] * G[5])
             - G[3] * (G[1] * G[8] - G[7] * G[2])
             + G[6] * (G[1] * G[5] - G[4] * G[2]);
    }

    KOKKOS_INLINE_FUNCTION
    static void cofactor_small(const Scalar* G, Index nd, Scalar* cofactor) noexcept
    {
        if (nd == Index(2)) {
            cofactor[0] = G[3];
            cofactor[1] = -G[2];
            cofactor[2] = -G[1];
            cofactor[3] = G[0];
            return;
        }
        cofactor[0] = G[4] * G[8] - G[7] * G[5];
        cofactor[1] = G[6] * G[5] - G[3] * G[8];
        cofactor[2] = G[3] * G[7] - G[6] * G[4];
        cofactor[3] = G[7] * G[2] - G[1] * G[8];
        cofactor[4] = G[0] * G[8] - G[6] * G[2];
        cofactor[5] = G[6] * G[1] - G[0] * G[7];
        cofactor[6] = G[1] * G[5] - G[4] * G[2];
        cofactor[7] = G[3] * G[2] - G[0] * G[5];
        cofactor[8] = G[0] * G[4] - G[3] * G[1];
    }

    KOKKOS_INLINE_FUNCTION
    static void inverse_small(
        const Scalar* G, Index nd, Scalar determinant, Scalar* inverse) noexcept
    {
        Scalar cofactor[9];
        cofactor_small(G, nd, cofactor);
        for (Index j = 0; j < nd; ++j)
            for (Index i = 0; i < nd; ++i)
                inverse[i + nd * j] = cofactor[j + nd * i] / determinant;
    }

    KOKKOS_INLINE_FUNCTION
    static Scalar determinant_tolerance(
        const Scalar* G, Index nd, Scalar epsilon) noexcept
    {
        Scalar scale = Scalar(0);
        for (Index q = 0; q < nd * nd; ++q) {
            const Scalar magnitude = G[q] < Scalar(0) ? -G[q] : G[q];
            if (magnitude > scale) scale = magnitude;
        }
        Scalar determinantScale = scale * scale;
        if (nd == Index(3)) determinantScale *= scale;
        return Scalar(128) * epsilon * determinantScale;
    }

    void evaluate_packed(
        Scalar* output, Scalar t, bool derivative, const char* kernelName)
    {
        validate_pointer(output, "packed mesh output");
        calculateTimeFunctions(t);
        const Index nmesh = mesh_extent();
        const auto phi0 = phi0_;
        const auto v0 = v0_;
        const auto phi1 = phi1_;
        const auto phi2 = phi2_;
        const auto coefficients = derivative ? da_ : a_;
        Kokkos::parallel_for(
            kernelName, range_policy(0, nmesh), KOKKOS_LAMBDA(const Index k) {
                output[k] = coefficients(0) * phi0(k) +
                            coefficients(1) * v0(k) +
                            coefficients(2) * phi1(k) +
                            coefficients(3) * phi2(k);
            });
    }

    void evaluate_matrix(
        Scalar* output, Scalar t, bool derivative, const char* kernelName)
    {
        validate_pointer(output, "matrix output");
        calculateTimeFunctions(t);
        const Index npe = npe_;
        const Index nd = nd_;
        const Index nc = nc_;
        const Index nd2 = nd * nd;
        const Index nmatrix = matrix_extent();
        const auto phi0 = phi0_;
        const auto v0 = v0_;
        const auto phi1 = phi1_;
        const auto phi2 = phi2_;
        const auto coefficients = derivative ? da_ : a_;
        Kokkos::parallel_for(
            kernelName, range_policy(0, nmatrix),
            KOKKOS_LAMBDA(const Index m) {
                const Index p = m % npe;
                const Index q = (m / npe) % nd2;
                const Index e = m / (npe * nd2);
                const Index k = p + npe * (nd + q) + npe * nc * e;
                output[m] = coefficients(0) * phi0(k) +
                            coefficients(1) * v0(k) +
                            coefficients(2) * phi1(k) +
                            coefficients(3) * phi2(k);
            });
    }

    void validate_evaluation_time(Scalar t) const
    {
        if (!timeSlabSet_)
            throw CubicMovingMeshError("setTimeSlab must be called before evaluation");
        if (!std::isfinite(t))
            throw CubicMovingMeshError("evaluation time must be finite");
        const Scalar scale = std::abs(tn_) + std::abs(dt_) + Scalar(1);
        const Scalar tolerance =
            Scalar(64) * std::numeric_limits<Scalar>::epsilon() * scale;
        if (t < tn_ - tolerance || t > tn_ + dt_ + tolerance)
            throw CubicMovingMeshError("evaluation time lies outside the current slab");
    }

    scalar_view phi0_;
    scalar_view v0_;
    scalar_view phi1_;
    scalar_view phi2_;
    scalar_view a_;
    scalar_view da_;

    Index npe_ = 0;
    Index nd_ = 0;
    Index ne_ = 0;
    Index nc_ = 0;
    Scalar tn_ = Scalar(0);
    Scalar dt_ = Scalar(0);
    Scalar c1_ = Scalar(0);
    Scalar c2_ = Scalar(1);
    bool timeSlabSet_ = false;
};

} // namespace exasim::cubicmovingmesh
