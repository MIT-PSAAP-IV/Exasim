#include <cubic_moving_mesh.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <type_traits>

namespace {

using Mesh = exasim::cubicmovingmesh::CubicMovingMesh<double, std::int64_t>;

static_assert(!std::is_copy_constructible_v<Mesh>);
static_assert(std::is_move_constructible_v<Mesh>);
static_assert(std::is_same_v<
    decltype(&Mesh::updateMeshDeformation0),
    void (Mesh::*)(const double*, const double*)>);
static_assert(std::is_same_v<
    decltype(&Mesh::updateMeshVelocity0), void (Mesh::*)(const double*)>);
static_assert(std::is_same_v<
    decltype(&Mesh::updateMeshDeformation1),
    void (Mesh::*)(const double*, const double*)>);
static_assert(std::is_same_v<
    decltype(&Mesh::updateMeshDeformation2),
    void (Mesh::*)(const double*, const double*)>);
static_assert(std::is_same_v<
    decltype(static_cast<void (Mesh::*)(const double*)>(
        &Mesh::initializeFirstSlabMesh)),
    void (Mesh::*)(const double*)>);
static_assert(std::is_same_v<
    decltype(static_cast<void (Mesh::*)(const double*, const double*)>(
        &Mesh::initializeFirstSlabMesh)),
    void (Mesh::*)(const double*, const double*)>);
static_assert(std::is_same_v<
    decltype(&Mesh::updateSlabMesh),
    void (Mesh::*)(const double*, const double*, const double*)>);
static_assert(std::is_same_v<
    decltype(&Mesh::calculateMeshDeformation),
    void (Mesh::*)(double*, double)>);
static_assert(std::is_same_v<
    decltype(&Mesh::calculateMeshVelocity), void (Mesh::*)(double*, double)>);
static_assert(std::is_same_v<
    decltype(&Mesh::calculateMeshDeformationGradient),
    void (Mesh::*)(double*, double)>);
static_assert(std::is_same_v<
    decltype(&Mesh::calculateMeshDeformationGradientDerivative),
    void (Mesh::*)(double*, double)>);
static_assert(std::is_same_v<
    decltype(&Mesh::calculateMeshDeformationGradientInverse),
    void (Mesh::*)(double*, double)>);
static_assert(std::is_same_v<
    decltype(&Mesh::calculateMeshJacobian), void (Mesh::*)(double*, double)>);
static_assert(std::is_same_v<
    decltype(&Mesh::calculateMeshJacobianDerivative),
    void (Mesh::*)(double*, double)>);

double maxTimeFunctionError = 0.0;
double maxStateError = 0.0;
double maxTrajectoryError = 0.0;

void check_close(double actual, double expected, double tolerance = 3e-13)
{
    maxTimeFunctionError = std::max(maxTimeFunctionError, std::abs(actual - expected));
    if (std::abs(actual - expected) > tolerance)
        throw std::runtime_error("time-function check failed");
}

void check_state_close(double actual, double expected, double tolerance = 3e-12)
{
    maxStateError = std::max(maxStateError, std::abs(actual - expected));
    if (std::abs(actual - expected) > tolerance)
        throw std::runtime_error("mesh-state check failed");
}

void check_trajectory_close(double actual, double expected, double tolerance = 8e-12)
{
    maxTrajectoryError = std::max(maxTrajectoryError, std::abs(actual - expected));
    if (std::abs(actual - expected) > tolerance)
        throw std::runtime_error("analytic trajectory check failed");
}

double polynomial_value(double b0, double b1, double b2, double b3, double t)
{
    return b0 + t * (b1 + t * (b2 + t * b3));
}

double polynomial_derivative(double b1, double b2, double b3, double t)
{
    return b1 + 2.0 * t * b2 + 3.0 * t * t * b3;
}

void fill_analytic_state(Mesh& mesh, double tn, double dt, double c1, double c2)
{
    auto h0 = Kokkos::create_mirror_view(mesh.phi0());
    auto hv = Kokkos::create_mirror_view(mesh.v0());
    auto h1 = Kokkos::create_mirror_view(mesh.phi1());
    auto h2 = Kokkos::create_mirror_view(mesh.phi2());
    for (std::int64_t e = 0; e < mesh.num_elements(); ++e)
        for (std::int64_t c = 0; c < mesh.num_components(); ++c)
            for (std::int64_t p = 0; p < mesh.nodes_per_element(); ++p) {
                const auto k = mesh.packed_index(p, c, e);
                const double tag = 1.0 + p + 3.0 * c + 7.0 * e;
                const double b0 = 0.13 * tag;
                const double b1 = -0.07 * tag + 0.2;
                const double b2 = 0.025 * tag - 0.03;
                const double b3 = -0.004 * tag + 0.006;
                h0(k) = polynomial_value(b0, b1, b2, b3, tn);
                hv(k) = polynomial_derivative(b1, b2, b3, tn);
                h1(k) = polynomial_value(b0, b1, b2, b3, tn + c1 * dt);
                h2(k) = polynomial_value(b0, b1, b2, b3, tn + c2 * dt);
            }
    Kokkos::deep_copy(mesh.phi0(), h0);
    Kokkos::deep_copy(mesh.v0(), hv);
    Kokkos::deep_copy(mesh.phi1(), h1);
    Kokkos::deep_copy(mesh.phi2(), h2);
}

void check_analytic_trajectory(std::int64_t nd, double c2)
{
    Mesh mesh(3, nd, 2);
    const double tn = 0.2;
    const double dt = 0.65;
    const double c1 = 0.3;
    mesh.setTimeSlab(tn, dt, c1, c2);
    fill_analytic_state(mesh, tn, dt, c1, c2);
    Mesh::scalar_view phi("analytic_phi", mesh.mesh_extent());
    Mesh::scalar_view velocity("analytic_velocity", mesh.mesh_extent());
    Mesh::scalar_view G("analytic_G", mesh.matrix_extent());
    Mesh::scalar_view dGdt("analytic_dGdt", mesh.matrix_extent());
    const double normalizedTimes[] = {0.0, c1, std::min(c2, 1.0), 0.53, 1.0};
    for (double s : normalizedTimes) {
        const double t = tn + s * dt;
        mesh.calculateMeshDeformation(phi.data(), t);
        mesh.calculateMeshVelocity(velocity.data(), t);
        mesh.calculateMeshDeformationGradient(G.data(), t);
        mesh.calculateMeshDeformationGradientDerivative(dGdt.data(), t);
        auto hphi = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), phi);
        auto hvelocity = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), velocity);
        auto hG = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), G);
        auto hdGdt = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dGdt);
        for (std::int64_t e = 0; e < mesh.num_elements(); ++e)
            for (std::int64_t c = 0; c < mesh.num_components(); ++c)
                for (std::int64_t p = 0; p < mesh.nodes_per_element(); ++p) {
                    const auto k = mesh.packed_index(p, c, e);
                    const double tag = 1.0 + p + 3.0 * c + 7.0 * e;
                    const double b0 = 0.13 * tag;
                    const double b1 = -0.07 * tag + 0.2;
                    const double b2 = 0.025 * tag - 0.03;
                    const double b3 = -0.004 * tag + 0.006;
                    check_trajectory_close(
                        hphi(k), polynomial_value(b0, b1, b2, b3, t));
                    check_trajectory_close(
                        hvelocity(k), polynomial_derivative(b1, b2, b3, t));
                    if (c >= nd) {
                        const auto q = c - nd;
                        const auto m = p + mesh.nodes_per_element() * q +
                            mesh.nodes_per_element() * nd * nd * e;
                        check_trajectory_close(hG(m), hphi(k));
                        check_trajectory_close(hdGdt(m), hvelocity(k));
                    }
                }
    }
}

Mesh::scalar_view make_view(const char* name, std::int64_t n, double shift)
{
    Mesh::scalar_view view(name, n);
    auto host = Kokkos::create_mirror_view(view);
    for (std::int64_t k = 0; k < n; ++k)
        host(k) = shift + 0.031 * static_cast<double>(k);
    Kokkos::deep_copy(view, host);
    return view;
}

void check_initialization_and_setters()
{
    Mesh mesh(2, 2, 2);
    const auto n = mesh.mesh_extent();
    auto X = make_view("state_X", mesh.coordinate_extent(), 2.0);
    auto d = make_view("state_d", n, -0.2);
    auto velocity = make_view("state_velocity", n, 0.7);

    mesh.initializeFirstSlabMesh(X.data());
    auto hX = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), X);
    auto hp0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mesh.phi0());
    auto hv0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mesh.v0());
    auto hp1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mesh.phi1());
    auto hp2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mesh.phi2());
    for (std::int64_t e = 0; e < mesh.num_elements(); ++e)
        for (std::int64_t c = 0; c < mesh.num_components(); ++c)
            for (std::int64_t p = 0; p < mesh.nodes_per_element(); ++p) {
                const auto k = mesh.packed_index(p, c, e);
                double expected;
                if (c < mesh.spatial_dimension()) {
                    expected = hX(mesh.coordinate_index(p, c, e));
                }
                else {
                    const auto q = c - mesh.spatial_dimension();
                    const auto i = q % mesh.spatial_dimension();
                    const auto j = q / mesh.spatial_dimension();
                    expected = i == j ? 1.0 : 0.0;
                }
                check_state_close(hp0(k), expected);
                check_state_close(hp1(k), expected);
                check_state_close(hp2(k), expected);
                check_state_close(hv0(k), 0.0);
            }

    mesh.initializeFirstSlabMesh(X.data(), d.data());
    auto hd = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d);
    hp0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mesh.phi0());
    hv0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mesh.v0());
    hp1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mesh.phi1());
    hp2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mesh.phi2());
    for (std::int64_t e = 0; e < mesh.num_elements(); ++e)
        for (std::int64_t c = 0; c < mesh.num_components(); ++c)
            for (std::int64_t p = 0; p < mesh.nodes_per_element(); ++p) {
                const auto k = mesh.packed_index(p, c, e);
                double expected;
                if (c < mesh.spatial_dimension()) {
                    expected = hX(mesh.coordinate_index(p, c, e)) + hd(k);
                }
                else {
                    const auto q = c - mesh.spatial_dimension();
                    expected = (q % 2 == q / 2 ? 1.0 : 0.0) - hd(k);
                }
                check_state_close(hp0(k), expected);
                check_state_close(hp1(k), expected);
                check_state_close(hp2(k), expected);
                check_state_close(hv0(k), 0.0);
            }

    mesh.updateMeshDeformation0(X.data(), d.data());
    mesh.updateMeshVelocity0(velocity.data());
    mesh.updateMeshDeformation1(X.data(), d.data());
    mesh.updateMeshDeformation2(X.data(), d.data());
    auto hvelocity = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), velocity);
    hv0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mesh.v0());
    for (std::int64_t k = 0; k < n; ++k)
        check_state_close(hv0(k), hvelocity(k));
}

void check_rollover(double c2)
{
    Mesh mesh(2, 3, 1);
    const auto n = mesh.mesh_extent();
    auto p0 = make_view("roll_p0", n, 0.2);
    auto w0 = make_view("roll_w0", n, -0.4);
    auto p1 = make_view("roll_p1", n, 1.1);
    auto p2 = make_view("roll_p2", n, 2.3);
    Kokkos::deep_copy(mesh.phi0(), p0);
    Kokkos::deep_copy(mesh.v0(), w0);
    Kokkos::deep_copy(mesh.phi1(), p1);
    Kokkos::deep_copy(mesh.phi2(), p2);
    mesh.setTimeSlab(0.3, 0.6, 0.31, c2);
    mesh.calculateTimeFunctions(0.9);
    auto ha = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), mesh.time_functions());
    auto hda = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), mesh.time_function_derivatives());
    auto hp0old = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p0);
    auto hw0old = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), w0);
    auto hp1old = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p1);
    auto hp2old = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p2);

    mesh.updateMeshDeformationAndVelocity0();
    auto hp0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mesh.phi0());
    auto hv0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mesh.v0());
    for (std::int64_t k = 0; k < n; ++k) {
        const double expectedP = ha(0) * hp0old(k) + ha(1) * hw0old(k) +
                                 ha(2) * hp1old(k) + ha(3) * hp2old(k);
        const double expectedV = hda(0) * hp0old(k) + hda(1) * hw0old(k) +
                                 hda(2) * hp1old(k) + hda(3) * hp2old(k);
        check_state_close(hp0(k), expectedP);
        check_state_close(hv0(k), expectedV);
        if (c2 == 1.0)
            check_state_close(hp0(k), hp2old(k));
    }
}

void check_update_slab_ordering()
{
    Mesh mesh(1, 2, 1);
    const auto n = mesh.mesh_extent();
    auto old0 = make_view("order_p0", n, 0.1);
    auto oldv = make_view("order_v0", n, -0.7);
    auto old1 = make_view("order_p1", n, 1.4);
    auto old2 = make_view("order_p2", n, 2.8);
    Kokkos::deep_copy(mesh.phi0(), old0);
    Kokkos::deep_copy(mesh.v0(), oldv);
    Kokkos::deep_copy(mesh.phi1(), old1);
    Kokkos::deep_copy(mesh.phi2(), old2);
    mesh.setTimeSlab(0.0, 0.5, 0.25, 1.4);
    mesh.calculateTimeFunctions(0.5);
    auto ha = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mesh.time_functions());
    auto h0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), old0);
    auto hv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), oldv);
    auto h1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), old1);
    auto h2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), old2);
    auto X = make_view("order_X", mesh.coordinate_extent(), 10.0);
    auto d1 = make_view("order_d1", n, -3.0);
    auto d2 = make_view("order_d2", n, 4.0);
    mesh.updateSlabMesh(X.data(), d1.data(), d2.data());
    auto result = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mesh.phi0());
    for (std::int64_t k = 0; k < n; ++k)
        check_state_close(result(k), ha(0) * h0(k) + ha(1) * hv(k) +
                                     ha(2) * h1(k) + ha(3) * h2(k));
}

void check_time_functions(double c1, double c2)
{
    Mesh mesh(2, 2, 1);
    const double tn = 1.25;
    const double dt = 0.7;
    mesh.setTimeSlab(tn, dt, c1, c2);

    const auto check = [&](double t, int cardinal) {
        mesh.calculateTimeFunctions(t);
        auto a = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), mesh.time_functions());
        auto da = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), mesh.time_function_derivatives());
        if (cardinal >= 0)
            for (int i = 0; i < 4; ++i)
                check_close(a(i), i == cardinal ? 1.0 : 0.0);
        check_close(a(0) + a(2) + a(3), 1.0);
        check_close(da(0) + da(2) + da(3), 0.0, 2e-12);
        if (t == tn) {
            check_close(da(0), 0.0);
            check_close(da(1), 1.0);
            check_close(da(2), 0.0);
            check_close(da(3), 0.0);
        }
    };

    check(tn, 0);
    check(tn + c1 * dt, 2);
    if (c2 <= 1.0)
        check(tn + c2 * dt, 3);
    check(tn + 0.37 * dt, -1);
    check(tn + dt, c2 == 1.0 ? 3 : -1);

    // For c2 > 1 the second prescribed state is outside the evaluation slab.
    // Check its cardinal basis identity algebraically without accepting an
    // out-of-slab public evaluation time.
    const double tau = c2 * dt;
    const double b0 = ((c1 * dt - tau) * (c2 * dt - tau) *
                       (c1 * c2 * dt + (c1 + c2) * tau)) /
                      (c1 * c1 * c2 * c2 * dt * dt * dt);
    const double b1 = tau * (c1 * dt - tau) * (c2 * dt - tau) /
                      (c1 * c2 * dt * dt);
    const double b2 = tau * tau * (c2 * dt - tau) /
                      (c1 * c1 * (c2 - c1) * dt * dt * dt);
    const double b3 = tau * tau * (tau - c1 * dt) /
                      (c2 * c2 * (c2 - c1) * dt * dt * dt);
    check_close(b0, 0.0);
    check_close(b1, 0.0);
    check_close(b2, 0.0);
    check_close(b3, 1.0);
}

Mesh::scalar_view make_target_deformation(
    const Mesh& mesh, const char* name, double t)
{
    Mesh::scalar_view deformation(name, mesh.mesh_extent());
    auto host = Kokkos::create_mirror_view(deformation);
    for (std::int64_t e = 0; e < mesh.num_elements(); ++e)
        for (std::int64_t c = 0; c < mesh.num_components(); ++c)
            for (std::int64_t p = 0; p < mesh.nodes_per_element(); ++p) {
                const auto k = mesh.packed_index(p, c, e);
                const double tag = 1.0 + p + 3.0 * c + 7.0 * e;
                const double p0 = 0.13 * tag;
                const double p1 = -0.07 * tag + 0.2;
                const double p2 = 0.025 * tag - 0.03;
                const double p3 = -0.004 * tag + 0.006;
                const double target = polynomial_value(p0, p1, p2, p3, t);
                if (c < mesh.spatial_dimension()) {
                    host(k) = target;
                }
                else {
                    const auto q = c - mesh.spatial_dimension();
                    const auto i = q % mesh.spatial_dimension();
                    const auto j = q / mesh.spatial_dimension();
                    host(k) = (i == j ? 1.0 : 0.0) - target;
                }
            }
    Kokkos::deep_copy(deformation, host);
    return deformation;
}

void check_repeated_slabs(std::int64_t nd)
{
    Mesh mesh(2, nd, 2);
    double tn = 0.0;
    double dt = 0.31;
    double c1 = 0.24;
    double c2 = 0.76;
    mesh.setTimeSlab(tn, dt, c1, c2);
    fill_analytic_state(mesh, tn, dt, c1, c2);
    Mesh::scalar_view X("multislab_X", mesh.coordinate_extent());
    Kokkos::deep_copy(X, 0.0);

    const double nextDt[] = {0.27, 0.43, 0.36, 0.29, 0.41};
    const double nextC1[] = {0.31, 0.28, 0.35, 0.22, 0.4};
    const double nextC2[] = {1.0, 1.35, 0.82, 1.5, 0.91};
    for (int slab = 0; slab < 5; ++slab) {
        const double nextTn = tn + dt;
        auto d1 = make_target_deformation(
            mesh, "multislab_d1", nextTn + nextC1[slab] * nextDt[slab]);
        auto d2 = make_target_deformation(
            mesh, "multislab_d2", nextTn + nextC2[slab] * nextDt[slab]);
        mesh.updateSlabMesh(X.data(), d1.data(), d2.data());

        auto hp0 = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), mesh.phi0());
        auto hv0 = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), mesh.v0());
        for (std::int64_t e = 0; e < mesh.num_elements(); ++e)
            for (std::int64_t c = 0; c < mesh.num_components(); ++c)
                for (std::int64_t p = 0; p < mesh.nodes_per_element(); ++p) {
                    const auto k = mesh.packed_index(p, c, e);
                    const double tag = 1.0 + p + 3.0 * c + 7.0 * e;
                    const double p0 = 0.13 * tag;
                    const double p1 = -0.07 * tag + 0.2;
                    const double p2 = 0.025 * tag - 0.03;
                    const double p3 = -0.004 * tag + 0.006;
                    check_trajectory_close(
                        hp0(k), polynomial_value(p0, p1, p2, p3, nextTn), 2e-11);
                    check_trajectory_close(
                        hv0(k), polynomial_derivative(p1, p2, p3, nextTn), 2e-11);
                }
        tn = nextTn;
        dt = nextDt[slab];
        c1 = nextC1[slab];
        c2 = nextC2[slab];
        mesh.setTimeSlab(tn, dt, c1, c2);
    }
}

void check_pointer_validation()
{
    Mesh mesh(2, 2, 1);
    mesh.setTimeSlab(0.0, 1.0, 0.3, 1.2);
    bool caught = false;
    try {
        mesh.initializeFirstSlabMesh(nullptr);
    }
    catch (const exasim::cubicmovingmesh::CubicMovingMeshError&) {
        caught = true;
    }
    if (!caught)
        throw std::runtime_error("null pointer was not rejected");

    caught = false;
    try {
        mesh.calculateMeshJacobian(nullptr, 0.5);
    }
    catch (const exasim::cubicmovingmesh::CubicMovingMeshError&) {
        caught = true;
    }
    if (!caught)
        throw std::runtime_error("null output pointer was not rejected");
}

void check_validation()
{
    bool caught = false;
    try {
        Mesh invalid(1, 1, 1);
    }
    catch (const exasim::cubicmovingmesh::CubicMovingMeshError&) {
        caught = true;
    }
    assert(caught);

    Mesh mesh(2, 3, 2);
    assert(mesh.num_components() == 12);
    assert(mesh.coordinate_extent() == 12);
    assert(mesh.mesh_extent() == 48);
    assert(mesh.coordinate_index(1, 2, 1) == 11);
    assert(mesh.packed_index(1, 11, 1) == 47);
    assert(mesh.matrix_channel(2, 1) == 8);

    caught = false;
    try {
        mesh.setTimeSlab(0.0, 1.0, 0.5, 0.5);
    }
    catch (const exasim::cubicmovingmesh::CubicMovingMeshError&) {
        caught = true;
    }
    assert(caught);

    mesh.setTimeSlab(0.0, 1.0, 0.4, 1.3);
    caught = false;
    try {
        mesh.calculateTimeFunctions(1.1);
    }
    catch (const exasim::cubicmovingmesh::CubicMovingMeshError&) {
        caught = true;
    }
    assert(caught);
}

} // namespace

int main(int argc, char** argv)
{
    Kokkos::initialize(argc, argv);
    try {
        check_time_functions(0.25, 0.8);
        check_time_functions(0.35, 1.0);
        check_time_functions(0.4, 1.4);
        check_validation();
        check_initialization_and_setters();
        check_rollover(0.78);
        check_rollover(1.0);
        check_rollover(1.45);
        check_update_slab_ordering();
        for (std::int64_t nd : {std::int64_t(2), std::int64_t(3)}) {
            check_analytic_trajectory(nd, 0.75);
            check_analytic_trajectory(nd, 1.0);
            check_analytic_trajectory(nd, 1.35);
            check_repeated_slabs(nd);
        }
        check_pointer_validation();
        using FloatMesh =
            exasim::cubicmovingmesh::CubicMovingMesh<float, std::int32_t>;
        FloatMesh floatMesh(1, 2, 1);
        floatMesh.setTimeSlab(0.0f, 1.0f, 0.25f, 1.25f);
        floatMesh.calculateTimeFunctions(0.5f);
        std::cout << "max_time_function_error " << maxTimeFunctionError << '\n';
        std::cout << "max_state_error " << maxStateError << '\n';
        std::cout << "max_trajectory_error " << maxTrajectoryError << '\n';
    }
    catch (...) {
        Kokkos::finalize();
        throw;
    }
    Kokkos::finalize();
    return 0;
}
