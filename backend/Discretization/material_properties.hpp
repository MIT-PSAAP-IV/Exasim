#include <stdexcept>
#include <cmath>
#include <iomanip>
#include <sstream>

template <class T>
KOKKOS_INLINE_FUNCTION T materialproperties_abs(T x)
{
    return x < static_cast<T>(0) ? -x : x;
}

template <class T>
KOKKOS_INLINE_FUNCTION T materialproperties_max3(T a, T b, T c)
{
    T m = a > b ? a : b;
    return m > c ? m : c;
}

template <class T>
KOKKOS_INLINE_FUNCTION T materialproperties_boundary_tolerance(T lo, T hi)
{
    const T eps = (sizeof(T) == sizeof(float)) ? static_cast<T>(1.0e-5) : static_cast<T>(1.0e-12);
    return static_cast<T>(64) * eps *
           materialproperties_max3(static_cast<T>(1), materialproperties_abs(lo), materialproperties_abs(hi));
}

template <class I>
inline void materialproperties_check_status(const I* elementIndex, I ng, const char* message)
{
    I bad = 0;
    Kokkos::parallel_reduce(
        "materialproperties_check_status",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::IndexType<I>>(0, ng),
        KOKKOS_LAMBDA(const I ig, I& local_bad) {
            const I e = elementIndex[ig];
            const I code = (e < 0) ? -e : static_cast<I>(0);
            if (code > local_bad) {
                local_bad = code;
            }
        },
        Kokkos::Max<I>(bad));
    if (bad != 0) {
        throw std::runtime_error(message);
    }
}

template <class T, class I>
inline std::string materialproperties_yesno(bool value)
{
    return value ? "yes" : "no";
}

template <class T, class I>
inline void materialproperties_check_locate_status(
    const I* elementIndex,
    const T* X,
    const T* xelem,
    const I* elementCounts,
    const I* xelemoffset,
    I ng,
    I nstate)
{
    using ExecSpace = Kokkos::DefaultExecutionSpace;

    I badIg = ng;
    Kokkos::parallel_reduce(
        "materialproperties_check_locate_status_index",
        Kokkos::RangePolicy<ExecSpace, Kokkos::IndexType<I>>(0, ng),
        KOKKOS_LAMBDA(const I ig, I& local_bad_ig) {
            if (elementIndex[ig] < 0 && ig < local_bad_ig) {
                local_bad_ig = ig;
            }
        },
        Kokkos::Min<I>(badIg));

    if (badIg >= ng) {
        return;
    }

    Kokkos::View<T*, typename ExecSpace::memory_space> diagT(
        "materialproperties_locate_diag_T", 5 * nstate);
    Kokkos::View<I*, typename ExecSpace::memory_space> diagI(
        "materialproperties_locate_diag_I", 4 + 4 * nstate);

    Kokkos::parallel_for(
        "materialproperties_check_locate_status_values",
        Kokkos::RangePolicy<ExecSpace, Kokkos::IndexType<I>>(0, 1),
        KOKKOS_LAMBDA(const I) {
            diagI(0) = elementIndex[badIg];
            diagI(1) = badIg;
            diagI(2) = ng;
            diagI(3) = nstate;

            for (I is = 0; is < nstate; ++is) {
                const T Xis0 = X[badIg + ng * is];
                const I nel = elementCounts[is];
                const I offset = xelemoffset[is];

                I iel = static_cast<I>(-1);
                T xmin = static_cast<T>(0);
                T xmax = static_cast<T>(0);
                T h = static_cast<T>(0);
                T xleft = static_cast<T>(0);
                T xright = static_cast<T>(0);

                if (nel >= 1) {
                    xmin = xelem[offset];
                    xmax = xelem[offset + nel];
                    if (Xis0 == xmax) {
                        iel = nel - 1;
                    } else {
                        I lo = 0;
                        I hi = nel + 1;
                        while (lo < hi) {
                            const I mid = lo + (hi - lo) / 2;
                            if (Xis0 < xelem[offset + mid]) {
                                hi = mid;
                            } else {
                                lo = mid + 1;
                            }
                        }
                        iel = lo - 1;
                    }
                    if (iel >= 0 && iel < nel) {
                        xleft = xelem[offset + iel];
                        xright = xelem[offset + iel + 1];
                        h = xright - xleft;
                    } else if (nel > 0) {
                        xleft = xelem[offset];
                        xright = xelem[offset + nel];
                        h = (nel > 0) ? (xright - xleft) / static_cast<T>(nel) : static_cast<T>(0);
                    }
                }

                diagT(is) = Xis0;
                diagT(nstate + is) = xmin;
                diagT(2 * nstate + is) = xmax;
                diagT(3 * nstate + is) = h;
                diagT(4 * nstate + is) = xleft;

                diagI(4 + is) = iel;
                diagI(4 + nstate + is) = nel;
                diagI(4 + 2 * nstate + is) = offset;
                diagI(4 + 3 * nstate + is) = static_cast<I>(0);
            }
        });

    auto hDiagT = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), diagT);
    auto hDiagI = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), diagI);

    const I code = -hDiagI(0);
    const I ig = hDiagI(1);

    std::ostringstream os;
    os << std::setprecision(17);
    os << "materialproperties_kokkos: invalid material database query\n\n";
    os << "query:\n";
    if (nstate > 0) {
        const T xi = hDiagT(0);
        const bool xiFinite = std::isfinite(static_cast<double>(xi));
        os << "  xi  = " << xi << "\n";
        os << "  rho = ";
        if (xiFinite) {
            os << std::exp(static_cast<double>(xi));
        } else {
            os << "NaN/Inf";
        }
        os << " kg/m^3\n";
        os << "  xi finite  = " << materialproperties_yesno<T,I>(xiFinite) << "\n";
        os << "  rho finite = " << materialproperties_yesno<T,I>(
            xiFinite && std::isfinite(std::exp(static_cast<double>(xi)))) << "\n";
    }
    if (nstate > 1) {
        const T e = hDiagT(1);
        os << "  e   = " << e << " J/kg\n";
        os << "  e finite   = " << materialproperties_yesno<T,I>(
            std::isfinite(static_cast<double>(e))) << "\n";
    }
    for (I is = 2; is < nstate; ++is) {
        const T x = hDiagT(is);
        os << "  state[" << is << "] = " << x << "\n";
        os << "  state[" << is << "] finite = "
           << materialproperties_yesno<T,I>(std::isfinite(static_cast<double>(x))) << "\n";
    }

    os << "\ndatabase:\n";
    if (nstate > 0) {
        const T xiMin = hDiagT(nstate);
        const T xiMax = hDiagT(2 * nstate);
        os << "  xi range  = [" << xiMin << ", " << xiMax << "]\n";
        os << "  rho range = [" << std::exp(static_cast<double>(xiMin))
           << ", " << std::exp(static_cast<double>(xiMax)) << "] kg/m^3\n";
    }
    if (nstate > 1) {
        os << "  e range   = [" << hDiagT(nstate + 1)
           << ", " << hDiagT(2 * nstate + 1) << "] J/kg\n";
    }
    for (I is = 2; is < nstate; ++is) {
        os << "  state[" << is << "] range = [" << hDiagT(nstate + is)
           << ", " << hDiagT(2 * nstate + is) << "]\n";
    }

    os << "\nlookup:\n";
    if (nstate > 0) {
        os << "  ix  = " << hDiagI(4) << "\n";
        os << "  nx  = " << hDiagI(4 + nstate) << "\n";
        os << "  dxi = " << hDiagT(3 * nstate) << "\n";
    }
    if (nstate > 1) {
        os << "  ie  = " << hDiagI(5) << "\n";
        os << "  ne  = " << hDiagI(5 + nstate) << "\n";
        os << "  de  = " << hDiagT(3 * nstate + 1) << "\n";
    }
    for (I is = 2; is < nstate; ++is) {
        os << "  i[" << is << "]  = " << hDiagI(4 + is) << "\n";
        os << "  n[" << is << "]  = " << hDiagI(4 + nstate + is) << "\n";
        os << "  d[" << is << "]  = " << hDiagT(3 * nstate + is) << "\n";
    }
    os << "  material element index = " << hDiagI(0) << "\n";
    os << "  query-point index      = " << ig << "\n";
    os << "  number of query points = " << ng << "\n";
    os << "  MPI rank               = unavailable in materialproperties_kokkos\n";

    os << "\nreason:\n";
    bool wroteReason = false;
    for (I is = 0; is < nstate; ++is) {
        const T value = hDiagT(is);
        const T xmin = hDiagT(nstate + is);
        const T xmax = hDiagT(2 * nstate + is);
        const bool finite = std::isfinite(static_cast<double>(value));
        const char* name = (is == 0) ? "xi" : ((is == 1) ? "e" : "state");
        if (!finite) {
            os << "  " << name;
            if (is >= 2) os << "[" << is << "]";
            os << " is NaN or Inf";
            os << "\n";
            wroteReason = true;
        } else if (hDiagI(4 + nstate + is) < 1) {
            os << "  invalid material element count in ";
            os << name;
            if (is >= 2) os << "[" << is << "]";
            os << " direction\n";
            wroteReason = true;
        } else if (value < xmin) {
            os << "  " << name;
            if (is >= 2) os << "[" << is << "]";
            os << " below database minimum\n";
            wroteReason = true;
        } else if (value > xmax) {
            os << "  " << name;
            if (is >= 2) os << "[" << is << "]";
            os << " above database maximum\n";
            wroteReason = true;
        }
    }
    if (code == static_cast<I>(3)) {
        os << "  invalid material element index\n";
        wroteReason = true;
    } else if (code == static_cast<I>(4)) {
        os << "  non-positive material element size\n";
        wroteReason = true;
    } else if (code == static_cast<I>(1)) {
        os << "  invalid material element count\n";
        wroteReason = true;
    }
    if (!wroteReason) {
        os << "  material lookup failed with status code " << code << "\n";
    }

    throw std::runtime_error(os.str());
}

template <class T=dstype, class I=Int>
inline void materialproperties_kokkos(
    T* U,
    const T* X,
    const T* dgnodes,
    const T* udg,
    const T* xelem,
    const I* elementCounts,
    const I* xelemoffset,
    T* tmd,
    I* tmi,
    I ng,
    I ne,
    I npe,
    I porder,
    I nstate,
    I nprop) {
    (void)ne;

    using ExecSpace = Kokkos::DefaultExecutionSpace;

    const I np = porder + 1;

    // Integer workspace:
    //   ie(ng,nstate)
    //   elementIndex(ng)
    // Required ntmi = ng*(nstate + 1).
    I* ie = tmi;
    I* elementIndex = ie + ng * nstate;

    // Floating-point workspace:
    //   he(ng,nstate), xref(ng,nstate), xi(ng,np,nstate),
    //   shap1d(ng,np,nstate).
    // Required ntmd = ng*nstate*(2 + 2*np).
    T* he = tmd;
    T* xref = he + ng * nstate;
    T* xi = xref + ng * nstate;
    T* shap1d = xi + ng * np * nstate;

    Kokkos::parallel_for(
        "materialproperties_value_locate",
        Kokkos::RangePolicy<ExecSpace, Kokkos::IndexType<I>>(0, ng),
        KOKKOS_LAMBDA(const I ig) {
            I e = 0;
            I stride = 1;

            for (I is = 0; is < nstate; ++is) {
                const T Xis0 = X[ig + ng * is];
                const I nel = elementCounts[is];
                const I offset = xelemoffset[is];

                if (nel < 1) {
                    elementIndex[ig] = -1;
                    return;
                }
                const T xmin = xelem[offset];
                const T xmax = xelem[offset + nel];
                const T tol = materialproperties_boundary_tolerance(xmin, xmax);
                T Xis = Xis0;
                if (Xis0 < xmin) {
                    if (xmin - Xis0 <= tol) {
                        Xis = xmin;
                    } else {
                        elementIndex[ig] = -2;
                        return;
                    }
                } else if (Xis0 > xmax) {
                    if (Xis0 - xmax <= tol) {
                        Xis = xmax;
                    } else {
                        elementIndex[ig] = -2;
                        return;
                    }
                }

                I iel = 0;
                if (Xis == xmax) {
                    iel = nel - 1;
                } else {
                    I lo = 0;
                    I hi = nel + 1;
                    while (lo < hi) {
                        const I mid = lo + (hi - lo) / 2;
                        if (Xis < xelem[offset + mid]) {
                            hi = mid;
                        } else {
                            lo = mid + 1;
                        }
                    }
                    iel = lo - 1;
                    if (iel < 0 || iel >= nel) {
                        elementIndex[ig] = -3;
                        return;
                    }
                }

                ie[ig + ng * is] = iel;

                const T xl = xelem[offset + iel];
                const T xr = xelem[offset + iel + 1];
                const T h = xr - xl;
                if (!(h > static_cast<T>(0))) {
                    elementIndex[ig] = -4;
                    return;
                }
                he[ig + ng * is] = h;
                xref[ig + ng * is] = (Xis - xl) / h;

                // Structured tensor-product element numbering, dimension 0
                // varying fastest.
                e += iel * stride;
                stride *= nel;
            }

            elementIndex[ig] = e;
        });
    materialproperties_check_locate_status(elementIndex, X, xelem, elementCounts, xelemoffset, ng, nstate);

    const I N2 = ng * np * nstate;
    Kokkos::parallel_for(
        "materialproperties_value_reference_nodes",
        Kokkos::RangePolicy<ExecSpace, Kokkos::IndexType<I>>(0, N2),
        KOKKOS_LAMBDA(const I idx) {
            const I ig = idx % ng;
            const I q = idx / ng;
            const I i = q % np;
            const I is = q / np;

            const I e = elementIndex[ig];
            const I iel = ie[ig + ng * is];
            const T h = he[ig + ng * is];
            const T xl = xelem[xelemoffset[is] + iel];

            I tensorStride = 1;
            for (I d = 0; d < is; ++d) {
                tensorStride *= np;
            }

            // Tensor-product node ordering:
            // a = i0 + np*i1 + np^2*i2 + ...
            const I a = i * tensorStride;
            const T Xnode = dgnodes[a + npe * (is + nstate * e)];
            xi[idx] = (Xnode - xl) / h;
        });

    const I N3 = ng * np * nstate;
    Kokkos::parallel_for(
        "materialproperties_value_shape1d",
        Kokkos::RangePolicy<ExecSpace, Kokkos::IndexType<I>>(0, N3),
        KOKKOS_LAMBDA(const I idx) {
            const I ig = idx % ng;
            const I q = idx / ng;
            const I i = q % np;
            const I is = q / np;

            const T xx = xref[ig + ng * is];
            const T xii = xi[idx];

            T Li = static_cast<T>(1);
            for (I j = 0; j < np; ++j) {
                if (j == i) {
                    continue;
                }
                const T xij = xi[ig + ng * (j + np * is)];
                const T denom = xii - xij;
                if (denom == static_cast<T>(0)) {
                    elementIndex[ig] = -5;
                    Li = static_cast<T>(0);
                } else {
                    Li *= (xx - xij) / denom;
                }
            }
            shap1d[idx] = Li;
        });
    materialproperties_check_status(elementIndex, ng,
        "materialproperties_kokkos: duplicated material interpolation nodes");

    const I N4 = ng * nprop;
    Kokkos::parallel_for(
        "materialproperties_value_interpolate",
        Kokkos::RangePolicy<ExecSpace, Kokkos::IndexType<I>>(0, N4),
        KOKKOS_LAMBDA(const I idx) {
            const I ig = idx % ng;
            const I ip = idx / ng;
            const I e = elementIndex[ig];

            T value = static_cast<T>(0);
            for (I a = 0; a < npe; ++a) {
                I rem = a;
                T Na = static_cast<T>(1);
                for (I is = 0; is < nstate; ++is) {
                    const I i = rem % np;
                    rem /= np;
                    Na *= shap1d[ig + ng * (i + np * is)];
                }
                value += Na * udg[a + npe * (ip + nprop * e)];
            }
            U[ig + ng * ip] = value;
        });

    // No fence here; callers can batch this with surrounding Kokkos work.
}

template <class T=dstype, class I=Int>
inline void materialproperties_kokkos(
    T* U,
    T* dUdX,
    const T* X,
    const T* dgnodes,
    const T* udg,
    const T* xelem,
    const I* elementCounts,
    const I* xelemoffset,
    T* tmd,
    I* tmi,
    I ng,
    I ne,
    I npe,
    I porder,
    I nstate,
    I nprop) {
    (void)ne;

    using ExecSpace = Kokkos::DefaultExecutionSpace;

    const I np = porder + 1;

    // Integer workspace:
    //   ie(ng,nstate)
    //   elementIndex(ng)
    // Required ntmi = ng*(nstate + 1).
    I* ie = tmi;
    I* elementIndex = ie + ng * nstate;

    // Floating-point workspace:
    //   he(ng,nstate), xref(ng,nstate), xi(ng,np,nstate),
    //   shap1d(ng,np,nstate), dshap1d(ng,np,nstate).
    // Required ntmd = ng*nstate*(2 + 3*np).
    T* he = tmd;
    T* xref = he + ng * nstate;
    T* xi = xref + ng * nstate;
    T* shap1d = xi + ng * np * nstate;
    T* dshap1d = shap1d + ng * np * nstate;

    Kokkos::parallel_for(
        "materialproperties_locate",
        Kokkos::RangePolicy<ExecSpace, Kokkos::IndexType<I>>(0, ng),
        KOKKOS_LAMBDA(const I ig) {
            I e = 0;
            I stride = 1;

            for (I is = 0; is < nstate; ++is) {
                const T Xis0 = X[ig + ng * is];
                const I nel = elementCounts[is];
                const I offset = xelemoffset[is];

                // Device equivalent of the CPU upper_bound interval rule:
                // [xelem[ie], xelem[ie+1]) for interiors, with the global upper
                // boundary included in the final element.
                if (nel < 1) {
                    elementIndex[ig] = -1;
                    return;
                }
                const T xmin = xelem[offset];
                const T xmax = xelem[offset + nel];
                const T tol = materialproperties_boundary_tolerance(xmin, xmax);
                T Xis = Xis0;
                if (Xis0 < xmin) {
                    if (xmin - Xis0 <= tol) {
                        Xis = xmin;
                    } else {
                        elementIndex[ig] = -2;
                        return;
                    }
                } else if (Xis0 > xmax) {
                    if (Xis0 - xmax <= tol) {
                        Xis = xmax;
                    } else {
                        elementIndex[ig] = -2;
                        return;
                    }
                }

                I iel = 0;
                if (Xis == xmax) {
                    iel = nel - 1;
                } else {
                    I lo = 0;
                    I hi = nel + 1;
                    while (lo < hi) {
                        const I mid = lo + (hi - lo) / 2;
                        if (Xis < xelem[offset + mid]) {
                            hi = mid;
                        } else {
                            lo = mid + 1;
                        }
                    }
                    iel = lo - 1;
                    if (iel < 0 || iel >= nel) {
                        elementIndex[ig] = -3;
                        return;
                    }
                }

                ie[ig + ng * is] = iel;

                const T xl = xelem[offset + iel];
                const T xr = xelem[offset + iel + 1];
                const T h = xr - xl;
                if (!(h > static_cast<T>(0))) {
                    elementIndex[ig] = -4;
                    return;
                }

                he[ig + ng * is] = h;
                xref[ig + ng * is] = (Xis - xl) / h;

                // Structured tensor-product element numbering, dimension 0
                // varying fastest.
                e += iel * stride;
                stride *= nel;
            }

            elementIndex[ig] = e;
        });
    materialproperties_check_locate_status(elementIndex, X, xelem, elementCounts, xelemoffset, ng, nstate);

    const I N2 = ng * np * nstate;
    Kokkos::parallel_for(
        "materialproperties_reference_nodes",
        Kokkos::RangePolicy<ExecSpace, Kokkos::IndexType<I>>(0, N2),
        KOKKOS_LAMBDA(const I idx) {
            const I ig = idx % ng;
            const I q = idx / ng;
            const I i = q % np;
            const I is = q / np;

            const I e = elementIndex[ig];
            const I iel = ie[ig + ng * is];
            const T h = he[ig + ng * is];
            const T xl = xelem[xelemoffset[is] + iel];

            I tensorStride = 1;
            for (I d = 0; d < is; ++d) {
                tensorStride *= np;
            }

            // Tensor-product node ordering:
            // a = i0 + np*i1 + np^2*i2 + ...
            // The 1D nodal line in dimension is is a = i*np^is.
            const I a = i * tensorStride;
            const T Xnode = dgnodes[a + npe * (is + nstate * e)];
            xi[idx] = (Xnode - xl) / h;
        });

    const I N3 = ng * np * nstate;
    Kokkos::parallel_for(
        "materialproperties_shape1d",
        Kokkos::RangePolicy<ExecSpace, Kokkos::IndexType<I>>(0, N3),
        KOKKOS_LAMBDA(const I idx) {
            const I ig = idx % ng;
            const I q = idx / ng;
            const I i = q % np;
            const I is = q / np;

            const T xx = xref[ig + ng * is];
            const T h = he[ig + ng * is];
            const T xii = xi[idx];

            T Li = static_cast<T>(1);
            for (I j = 0; j < np; ++j) {
                if (j == i) {
                    continue;
                }
                const T xij = xi[ig + ng * (j + np * is)];
                const T denom = xii - xij;
                if (denom == static_cast<T>(0)) {
                    elementIndex[ig] = -5;
                    Li = static_cast<T>(0);
                } else {
                    Li *= (xx - xij) / denom;
                }
            }
            shap1d[idx] = Li;

            T dLi = static_cast<T>(0);
            for (I m = 0; m < np; ++m) {
                if (m == i) {
                    continue;
                }
                const T xim = xi[ig + ng * (m + np * is)];
                const T denom_m = xii - xim;
                if (denom_m == static_cast<T>(0)) {
                    elementIndex[ig] = -5;
                    continue;
                }
                T term = static_cast<T>(1) / denom_m;
                for (I j = 0; j < np; ++j) {
                    if (j == i || j == m) {
                        continue;
                    }
                    const T xij = xi[ig + ng * (j + np * is)];
                    const T denom_j = xii - xij;
                    if (denom_j == static_cast<T>(0)) {
                        elementIndex[ig] = -5;
                        term = static_cast<T>(0);
                    } else {
                        term *= (xx - xij) / denom_j;
                    }
                }
                dLi += term;
            }
            dshap1d[idx] = dLi / h;
        });
    materialproperties_check_status(elementIndex, ng,
        "materialproperties_kokkos: duplicated material interpolation nodes");

    const I N4 = ng * nprop;
    Kokkos::parallel_for(
        "materialproperties_interpolate",
        Kokkos::RangePolicy<ExecSpace, Kokkos::IndexType<I>>(0, N4),
        KOKKOS_LAMBDA(const I idx) {
            const I ig = idx % ng;
            const I ip = idx / ng;
            const I e = elementIndex[ig];

            T value = static_cast<T>(0);
            for (I a = 0; a < npe; ++a) {
                I rem = a;
                T Na = static_cast<T>(1);
                for (I is = 0; is < nstate; ++is) {
                    const I i = rem % np;
                    rem /= np;
                    Na *= shap1d[ig + ng * (i + np * is)];
                }
                value += Na * udg[a + npe * (ip + nprop * e)];
            }
            U[ig + ng * ip] = value;

            for (I r = 0; r < nstate; ++r) {
                T deriv = static_cast<T>(0);
                for (I a = 0; a < npe; ++a) {
                    I rem = a;
                    T dNa = static_cast<T>(1);
                    for (I is = 0; is < nstate; ++is) {
                        const I i = rem % np;
                        rem /= np;
                        const I i1d = ig + ng * (i + np * is);
                        dNa *= (is == r) ? dshap1d[i1d] : shap1d[i1d];
                    }
                    deriv += dNa * udg[a + npe * (ip + nprop * e)];
                }
                dUdX[ig + ng * (ip + nprop * r)] = deriv;
            }
        });

    // No fence here; callers can batch this with surrounding Kokkos work.
}
