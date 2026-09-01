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
                const T Xis = X[ig + ng * is];
                const I nel = elementCounts[is];
                const I offset = xelemoffset[is];

                if (nel < 1) {
                    Kokkos::abort("materialproperties_kokkos: invalid material element count");
                }
                if (Xis < xelem[offset] || Xis > xelem[offset + nel]) {
                    Kokkos::abort("materialproperties_kokkos: query point outside material database domain");
                }

                I iel = 0;
                if (Xis == xelem[offset + nel]) {
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
                        Kokkos::abort("materialproperties_kokkos: failed to locate material element");
                    }
                }

                ie[ig + ng * is] = iel;

                const T xl = xelem[offset + iel];
                const T xr = xelem[offset + iel + 1];
                const T h = xr - xl;
                if (!(h > static_cast<T>(0))) {
                    Kokkos::abort("materialproperties_kokkos: non-positive material element size");
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
                Li *= (xx - xij) / (xii - xij);
            }
            shap1d[idx] = Li;
        });

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
                const T Xis = X[ig + ng * is];
                const I nel = elementCounts[is];
                const I offset = xelemoffset[is];

                // Device equivalent of the CPU upper_bound interval rule:
                // [xelem[ie], xelem[ie+1]) for interiors, with the global upper
                // boundary included in the final element.
                if (nel < 1) {
                    Kokkos::abort("materialproperties_kokkos: invalid material element count");
                }
                if (Xis < xelem[offset] || Xis > xelem[offset + nel]) {
                    Kokkos::abort("materialproperties_kokkos: query point outside material database domain");
                }

                I iel = 0;
                if (Xis == xelem[offset + nel]) {
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
                        Kokkos::abort("materialproperties_kokkos: failed to locate material element");
                    }
                }

                ie[ig + ng * is] = iel;

                const T xl = xelem[offset + iel];
                const T xr = xelem[offset + iel + 1];
                const T h = xr - xl;
                if (!(h > static_cast<T>(0))) {
                    Kokkos::abort("materialproperties_kokkos: non-positive material element size");
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
                Li *= (xx - xij) / (xii - xij);
            }
            shap1d[idx] = Li;

            T dLi = static_cast<T>(0);
            for (I m = 0; m < np; ++m) {
                if (m == i) {
                    continue;
                }
                const T xim = xi[ig + ng * (m + np * is)];
                T term = static_cast<T>(1) / (xii - xim);
                for (I j = 0; j < np; ++j) {
                    if (j == i || j == m) {
                        continue;
                    }
                    const T xij = xi[ig + ng * (j + np * is)];
                    term *= (xx - xij) / (xii - xij);
                }
                dLi += term;
            }
            dshap1d[idx] = dLi / h;
        });

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
