#include <cubic_moving_mesh.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

using Mesh = exasim::cubicmovingmesh::CubicMovingMesh<double, std::int64_t>;

double maxGeometryError = 0.0;
double maxInverseResidual = 0.0;
double maxFiniteDifferenceError = 0.0;

void check_close(double actual, double expected, double tolerance)
{
    maxGeometryError = std::max(maxGeometryError, std::abs(actual - expected));
    if (std::abs(actual - expected) > tolerance)
        throw std::runtime_error("geometry check failed");
}

double value(double b0, double b1, double b2, double b3, double t)
{
    return b0 + t * (b1 + t * (b2 + t * b3));
}

double derivative(double b1, double b2, double b3, double t)
{
    return b1 + 2.0 * b2 * t + 3.0 * b3 * t * t;
}

void coefficients(
    std::int64_t i, std::int64_t j, int mode,
    double& b0, double& b1, double& b2, double& b3)
{
    const double tag = 1.0 + i + 2.0 * j;
    b0 = (i == j ? 1.0 : 0.0) + (mode >= 1 ? 0.025 * tag : 0.0);
    b1 = mode >= 2 ? 0.012 * (i + 1.0) - 0.007 * (j + 1.0) : 0.0;
    b2 = mode >= 2 ? 0.0015 * tag : 0.0;
    b3 = mode >= 2 ? -0.0002 * tag : 0.0;
}

void analytic_matrix(std::int64_t nd, int mode, double t, double* G, double* dGdt)
{
    for (std::int64_t j = 0; j < nd; ++j)
        for (std::int64_t i = 0; i < nd; ++i) {
            double b0, b1, b2, b3;
            coefficients(i, j, mode, b0, b1, b2, b3);
            const auto q = i + nd * j;
            G[q] = value(b0, b1, b2, b3, t);
            dGdt[q] = derivative(b1, b2, b3, t);
        }
}

double determinant(const double* G, std::int64_t nd)
{
    if (nd == 2)
        return G[0] * G[3] - G[2] * G[1];
    return G[0] * (G[4] * G[8] - G[7] * G[5])
         - G[3] * (G[1] * G[8] - G[7] * G[2])
         + G[6] * (G[1] * G[5] - G[4] * G[2]);
}

void cofactor(const double* G, std::int64_t nd, double* C)
{
    if (nd == 2) {
        C[0] = G[3]; C[1] = -G[2]; C[2] = -G[1]; C[3] = G[0];
        return;
    }
    C[0] = G[4] * G[8] - G[7] * G[5];
    C[1] = G[6] * G[5] - G[3] * G[8];
    C[2] = G[3] * G[7] - G[6] * G[4];
    C[3] = G[7] * G[2] - G[1] * G[8];
    C[4] = G[0] * G[8] - G[6] * G[2];
    C[5] = G[6] * G[1] - G[0] * G[7];
    C[6] = G[1] * G[5] - G[4] * G[2];
    C[7] = G[3] * G[2] - G[0] * G[5];
    C[8] = G[0] * G[4] - G[3] * G[1];
}

void fill_geometry_state(
    Mesh& mesh, int mode, double tn, double dt, double c1, double c2)
{
    auto h0 = Kokkos::create_mirror_view(mesh.phi0());
    auto hv = Kokkos::create_mirror_view(mesh.v0());
    auto h1 = Kokkos::create_mirror_view(mesh.phi1());
    auto h2 = Kokkos::create_mirror_view(mesh.phi2());
    for (std::int64_t k = 0; k < mesh.mesh_extent(); ++k) {
        h0(k) = 0.0; hv(k) = 0.0; h1(k) = 0.0; h2(k) = 0.0;
    }
    for (std::int64_t e = 0; e < mesh.num_elements(); ++e)
        for (std::int64_t p = 0; p < mesh.nodes_per_element(); ++p)
            for (std::int64_t j = 0; j < mesh.spatial_dimension(); ++j)
                for (std::int64_t i = 0; i < mesh.spatial_dimension(); ++i) {
                    double b0, b1, b2, b3;
                    coefficients(i, j, mode, b0, b1, b2, b3);
                    const auto k = mesh.packed_index(p, mesh.matrix_channel(i, j), e);
                    h0(k) = value(b0, b1, b2, b3, tn);
                    hv(k) = derivative(b1, b2, b3, tn);
                    h1(k) = value(b0, b1, b2, b3, tn + c1 * dt);
                    h2(k) = value(b0, b1, b2, b3, tn + c2 * dt);
                }
    Kokkos::deep_copy(mesh.phi0(), h0);
    Kokkos::deep_copy(mesh.v0(), hv);
    Kokkos::deep_copy(mesh.phi1(), h1);
    Kokkos::deep_copy(mesh.phi2(), h2);
}

void check_geometry(std::int64_t nd, double c2, int mode)
{
    Mesh mesh(2, nd, 2);
    const double tn = 0.1;
    const double dt = 0.8;
    const double c1 = 0.32;
    mesh.setTimeSlab(tn, dt, c1, c2);
    fill_geometry_state(mesh, mode, tn, dt, c1, c2);
    Mesh::scalar_view Ginv("geometry_inverse", mesh.matrix_extent());
    Mesh::scalar_view J("geometry_jacobian", mesh.scalar_extent());
    Mesh::scalar_view dJdt("geometry_jacobian_derivative", mesh.scalar_extent());
    const double normalizedTimes[] = {0.0, 0.27, c1, 0.64, 1.0};
    for (double s : normalizedTimes) {
        const double t = tn + s * dt;
        mesh.calculateMeshDeformationGradientInverse(Ginv.data(), t);
        mesh.calculateMeshJacobian(J.data(), t);
        mesh.calculateMeshJacobianDerivative(dJdt.data(), t);
        auto hInverse = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Ginv);
        auto hJ = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), J);
        auto hdJdt = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dJdt);
        double G[9] = {}, dG[9] = {}, C[9] = {};
        analytic_matrix(nd, mode, t, G, dG);
        const double expectedJ = determinant(G, nd);
        cofactor(G, nd, C);
        double expectedDerivative = 0.0;
        for (std::int64_t q = 0; q < nd * nd; ++q)
            expectedDerivative += C[q] * dG[q];
        for (std::int64_t node = 0; node < mesh.scalar_extent(); ++node) {
            check_close(hJ(node), expectedJ, 2e-12);
            check_close(hdJdt(node), expectedDerivative, 3e-12);
            const auto p = node % mesh.nodes_per_element();
            const auto e = node / mesh.nodes_per_element();
            for (std::int64_t j = 0; j < nd; ++j)
                for (std::int64_t i = 0; i < nd; ++i) {
                    double product = 0.0;
                    for (std::int64_t k = 0; k < nd; ++k) {
                        const auto m = p + mesh.nodes_per_element() * (k + nd * j) +
                            mesh.nodes_per_element() * nd * nd * e;
                        product += G[i + nd * k] * hInverse(m);
                    }
                    const double residual = std::abs(product - (i == j ? 1.0 : 0.0));
                    maxInverseResidual = std::max(maxInverseResidual, residual);
                    if (residual > 3e-12)
                        throw std::runtime_error("inverse identity check failed");
                }
        }
    }

    const double center = tn + 0.55 * dt;
    const double h = 1e-6 * dt;
    mesh.calculateMeshJacobian(J.data(), center - h);
    auto minus = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), J);
    const double Jminus = minus(0);
    mesh.calculateMeshJacobian(J.data(), center + h);
    auto plus = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), J);
    mesh.calculateMeshJacobianDerivative(dJdt.data(), center);
    auto derivativeView =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dJdt);
    const double finiteDifference = (plus(0) - Jminus) / (2.0 * h);
    maxFiniteDifferenceError = std::max(
        maxFiniteDifferenceError, std::abs(finiteDifference - derivativeView(0)));
    if (std::abs(finiteDifference - derivativeView(0)) > 2e-9)
        throw std::runtime_error("Jacobian finite-difference check failed");
}

void check_singular_rejection()
{
    Mesh mesh(1, 2, 1);
    mesh.setTimeSlab(0.0, 1.0, 0.3, 1.2);
    Kokkos::deep_copy(mesh.phi0(), 0.0);
    Kokkos::deep_copy(mesh.v0(), 0.0);
    Kokkos::deep_copy(mesh.phi1(), 0.0);
    Kokkos::deep_copy(mesh.phi2(), 0.0);
    Mesh::scalar_view inverse("singular_inverse", mesh.matrix_extent());
    bool caught = false;
    try {
        mesh.calculateMeshDeformationGradientInverse(inverse.data(), 0.5);
    }
    catch (const exasim::cubicmovingmesh::CubicMovingMeshError&) {
        caught = true;
    }
    if (!caught)
        throw std::runtime_error("singular gradient was not rejected");
}

} // namespace

int main(int argc, char** argv)
{
    Kokkos::initialize(argc, argv);
    try {
        for (std::int64_t nd : {std::int64_t(2), std::int64_t(3)}) {
            check_geometry(nd, 1.0, 0);
            check_geometry(nd, 1.0, 1);
            for (double c2 : {0.78, 1.0, 1.4})
                check_geometry(nd, c2, 2);
        }
        check_singular_rejection();
        std::cout << "max_geometry_error " << maxGeometryError << '\n';
        std::cout << "max_inverse_residual " << maxInverseResidual << '\n';
        std::cout << "max_jacobian_fd_error " << maxFiniteDifferenceError << '\n';
    }
    catch (...) {
        Kokkos::finalize();
        throw;
    }
    Kokkos::finalize();
    return 0;
}
