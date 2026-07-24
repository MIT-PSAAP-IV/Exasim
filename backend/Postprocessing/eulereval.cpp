#include "eulereval.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace {

std::size_t stateIndex(int i, int c, int e, int npe, int nc)
{
    return static_cast<std::size_t>(i) +
           static_cast<std::size_t>(npe) *
               (static_cast<std::size_t>(c) +
                static_cast<std::size_t>(nc) * static_cast<std::size_t>(e));
}

std::size_t scalarIndex(int i, int e, int npe)
{
    return static_cast<std::size_t>(i) +
           static_cast<std::size_t>(npe) * static_cast<std::size_t>(e);
}

void validateComponentCount(int nd, int nc, int required)
{
    if (nc < required) {
        std::ostringstream oss;
        oss << "eulereval" << nd << "d requires nc >= " << required
            << " for nd = " << nd << ", but nc = " << nc << ".";
        throw std::runtime_error(oss.str());
    }
}

[[noreturn]] void invalidSelector(const std::string& quantity, int nd)
{
    std::ostringstream oss;
    oss << "Invalid Euler selector \"" << quantity << "\" for nd = " << nd << ". "
        << "Supported selectors: ";
    if (nd == 1) {
        oss << "\"r\", \"u\", \"p\", \"c\", \"c2\", \"M\", \"s\", \"t\", \"h\".";
    } else if (nd == 2) {
        oss << "\"r\", \"u\", \"v\", \"p\", \"c\", \"c2\", \"M\", \"s\", \"t\", \"h\".";
    } else if (nd == 3) {
        oss << "\"r\", \"u\", \"v\", \"w\", \"p\", \"c\", \"c2\", \"M\", \"s\", \"t\", \"h\".";
    } else {
        oss << "nd must be 1, 2, or 3.";
    }
    throw std::runtime_error(oss.str());
}

} // namespace

void eulereval1d(double* sca,
                 const double* u,
                 const std::string& quantity,
                 double gamma,
                 int npe,
                 int nc,
                 int ne)
{
    validateComponentCount(1, nc, 3);

    if (quantity != "r" && quantity != "u" && quantity != "p" &&
        quantity != "c" && quantity != "c2" && quantity != "M" &&
        quantity != "s" && quantity != "t" && quantity != "h") {
        invalidSelector(quantity, 1);
    }

    for (int e = 0; e < ne; ++e) {
        for (int i = 0; i < npe; ++i) {
            const double rho = u[stateIndex(i, 0, e, npe, nc)];
            const double rhou = u[stateIndex(i, 1, e, npe, nc)];
            const double rhoE = u[stateIndex(i, 2, e, npe, nc)];
            const double ux = rhou / rho;

            double value = 0.0;
            if (quantity == "r") {
                value = rho;
            } else if (quantity == "u") {
                value = ux;
            } else {
                const double kinetic = 0.5 * rhou * ux;
                const double p = (gamma - 1.0) * (rhoE - kinetic);

                if (quantity == "p") {
                    value = p;
                } else if (quantity == "c2") {
                    value = gamma * p / rho;
                } else if (quantity == "c") {
                    value = std::sqrt(gamma * p / rho);
                } else if (quantity == "M") {
                    value = std::abs(ux) / std::sqrt(gamma * p / rho);
                } else if (quantity == "s") {
                    value = p / std::pow(rho, gamma);
                } else if (quantity == "t") {
                    value = p / ((gamma - 1.0) * rho);
                } else if (quantity == "h") {
                    value = (rhoE + p) / rho;
                }
            }

            sca[scalarIndex(i, e, npe)] = value;
        }
    }
}

void eulereval2d(double* sca,
                 const double* u,
                 const std::string& quantity,
                 double gamma,
                 int npe,
                 int nc,
                 int ne)
{
    validateComponentCount(2, nc, 4);

    if (quantity != "r" && quantity != "u" && quantity != "v" &&
        quantity != "p" && quantity != "c" && quantity != "c2" &&
        quantity != "M" && quantity != "s" && quantity != "t" &&
        quantity != "h") {
        invalidSelector(quantity, 2);
    }

    for (int e = 0; e < ne; ++e) {
        for (int i = 0; i < npe; ++i) {
            const double rho = u[stateIndex(i, 0, e, npe, nc)];
            const double rhou = u[stateIndex(i, 1, e, npe, nc)];
            const double rhov = u[stateIndex(i, 2, e, npe, nc)];
            const double rhoE = u[stateIndex(i, 3, e, npe, nc)];
            const double ux = rhou / rho;
            const double uy = rhov / rho;

            double value = 0.0;
            if (quantity == "r") {
                value = rho;
            } else if (quantity == "u") {
                value = ux;
            } else if (quantity == "v") {
                value = uy;
            } else {
                const double kinetic = 0.5 * (rhou * ux + rhov * uy);
                const double p = (gamma - 1.0) * (rhoE - kinetic);

                if (quantity == "p") {
                    value = p;
                } else if (quantity == "c2") {
                    value = gamma * p / rho;
                } else if (quantity == "c") {
                    value = std::sqrt(gamma * p / rho);
                } else if (quantity == "M") {
                    value = std::sqrt(ux * ux + uy * uy) / std::sqrt(gamma * p / rho);
                } else if (quantity == "s") {
                    value = p / std::pow(rho, gamma);
                } else if (quantity == "t") {
                    value = p / ((gamma - 1.0) * rho);
                } else if (quantity == "h") {
                    value = (rhoE + p) / rho;
                }
            }

            sca[scalarIndex(i, e, npe)] = value;
        }
    }
}

void eulereval3d(double* sca,
                 const double* u,
                 const std::string& quantity,
                 double gamma,
                 int npe,
                 int nc,
                 int ne)
{
    validateComponentCount(3, nc, 5);

    if (quantity != "r" && quantity != "u" && quantity != "v" &&
        quantity != "w" && quantity != "p" && quantity != "c" &&
        quantity != "c2" && quantity != "M" && quantity != "s" &&
        quantity != "t" && quantity != "h") {
        invalidSelector(quantity, 3);
    }

    for (int e = 0; e < ne; ++e) {
        for (int i = 0; i < npe; ++i) {
            const double rho = u[stateIndex(i, 0, e, npe, nc)];
            const double rhou = u[stateIndex(i, 1, e, npe, nc)];
            const double rhov = u[stateIndex(i, 2, e, npe, nc)];
            const double rhow = u[stateIndex(i, 3, e, npe, nc)];
            const double rhoE = u[stateIndex(i, 4, e, npe, nc)];
            const double ux = rhou / rho;
            const double uy = rhov / rho;
            const double uz = rhow / rho;

            double value = 0.0;
            if (quantity == "r") {
                value = rho;
            } else if (quantity == "u") {
                value = ux;
            } else if (quantity == "v") {
                value = uy;
            } else if (quantity == "w") {
                value = uz;
            } else {
                const double kinetic = 0.5 * (rhou * ux + rhov * uy + rhow * uz);
                const double p = (gamma - 1.0) * (rhoE - kinetic);

                if (quantity == "p") {
                    value = p;
                } else if (quantity == "c2") {
                    value = gamma * p / rho;
                } else if (quantity == "c") {
                    value = std::sqrt(gamma * p / rho);
                } else if (quantity == "M") {
                    value = std::sqrt(ux * ux + uy * uy + uz * uz) /
                            std::sqrt(gamma * p / rho);
                } else if (quantity == "s") {
                    value = p / std::pow(rho, gamma);
                } else if (quantity == "t") {
                    value = p / ((gamma - 1.0) * rho);
                } else if (quantity == "h") {
                    value = (rhoE + p) / rho;
                }
            }

            sca[scalarIndex(i, e, npe)] = value;
        }
    }
}

void eulereval(double* sca,
               const double* u,
               const std::string& quantity,
               double gamma,
               int npe,
               int nc,
               int ne,
               int nd)
{
    if (nd == 1) {
        eulereval1d(sca, u, quantity, gamma, npe, nc, ne);
        return;
    }
    if (nd == 2) {
        eulereval2d(sca, u, quantity, gamma, npe, nc, ne);
        return;
    }
    if (nd == 3) {
        eulereval3d(sca, u, quantity, gamma, npe, nc, ne);
        return;
    }

    std::ostringstream oss;
    oss << "eulereval requires nd = 1, 2, or 3, but nd = " << nd << ".";
    throw std::runtime_error(oss.str());
}
