#include "backend/Model/Helmholtz/helmholtz_model.hpp"
#include "backend/Model/Helmholtz/helmholtzprovider.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>

template <int nd>
bool check_model()
{
    using Model = HelmholtzModel<nd>;
    constexpr int nq = 1 + nd;
    std::array<dstype, nq> uq{};
    std::array<dstype, 2> v{{2.5, 0.125}};
    std::array<dstype, nd> flux{};
    std::array<dstype, nd*nq> jac{};

    for (int d = 0; d < nd; ++d) uq[1 + d] = 0.25 * (d + 1);
    Model::flux(flux.data(), nullptr, uq.data(), v.data(), nullptr,
                nullptr, nullptr, 0.0);
    Model::flux_jac_uq(jac.data(), nullptr, uq.data(), v.data(), nullptr,
                       nullptr, nullptr, 0.0);

    const dstype eps = (sizeof(dstype) == sizeof(float)) ? 1.0e-3f : 1.0e-7;
    const dstype tolerance = (sizeof(dstype) == sizeof(float))
        ? 20.0f * std::numeric_limits<dstype>::epsilon() / eps
        : 1.0e-9;
    for (int j = 0; j < nq; ++j) {
        auto plus = uq;
        auto minus = uq;
        plus[j] += eps;
        minus[j] -= eps;
        std::array<dstype, nd> fp{}, fm{};
        Model::flux(fp.data(), nullptr, plus.data(), v.data(), nullptr,
                    nullptr, nullptr, 0.0);
        Model::flux(fm.data(), nullptr, minus.data(), v.data(), nullptr,
                    nullptr, nullptr, 0.0);
        for (int i = 0; i < nd; ++i) {
            const dstype fd = (fp[i] - fm[i]) / (2.0*eps);
            const dstype exact = jac[i + nd*j];
            if (std::abs(fd - exact) > tolerance) return false;
        }
    }

    const auto& abi = GetHelmholtzModelABI(nd);
    return abi.ncu == 1 && abi.nco == 2 && abi.ncw == 0;
}

int main()
{
    if (!check_model<1>() || !check_model<2>() || !check_model<3>()) {
        std::cerr << "Helmholtz model value/Jacobian/ABI check failed\n";
        return 1;
    }
    return 0;
}
