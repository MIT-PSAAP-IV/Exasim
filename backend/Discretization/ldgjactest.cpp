void LDGJacTestPerturbU(dstype* ueps, const dstype* u, const Int col,
        const dstype eps, const Int npe, const Int ncu, const Int ne)
{
    Int N = npe*ncu*ne;
    Kokkos::parallel_for("LDGJacTestPerturbU", N, KOKKOS_LAMBDA(const size_t idx) {
        ueps[idx] = u[idx] + ((idx == col) ? eps : zero);
    });
}

void LDGJacTestStoreFDColumn(dstype* Jfd, const dstype* Rup,
        const dstype* Rum, const Int col, const Int ncol,
        const dstype eps, const Int npe, const Int ncu)
{
    Int nrow = npe*ncu;
    Int e = col / nrow;
    Int c = col - e*nrow;
    dstype scale = one/(2.0*eps);

    Kokkos::parallel_for("LDGJacTestStoreFDColumn", nrow, KOKKOS_LAMBDA(const size_t idx) {
        Jfd[idx + nrow*c + nrow*ncol*e] =
            scale*(Rup[idx + nrow*e] - Rum[idx + nrow*e]);
    });
}

void LDGJacTestRuFromU(dstype* Ru, dstype* u, solstruct &sol,
        resstruct &res, appstruct &app, ExasimDriverABI& driver_abi,
        masterstruct &master, meshstruct &mesh, tempstruct &tmp,
        commonstruct &common)
{
    Int backend = common.backend;

    ArrayInsert(sol.udg, u, common.grid.npe, common.components.nc,
            common.meshsizes.ne, 0, common.grid.npe, 0,
            common.components.ncu, 0, common.meshsizes.ne1);

    GetUhat<exasim::detail::AbiAdapter>(sol, res, app, master, mesh, tmp, common,
            common.cublasHandle, 0, common.meshsizes.nbf, backend);

    if (common.components.ncq > 0)
        GetQ(sol, res, app, master, mesh, tmp, common, common.cublasHandle,
                0, common.meshsizes.nbe, 0, common.meshsizes.nbf, backend);

    if (common.components.ncw > 0)
        GetW<exasim::detail::AbiAdapter>(sol, res, app, master, mesh, tmp, common,
                common.cublasHandle, 0, common.meshsizes.nbe, 0,
                common.meshsizes.nbf, backend);

    if (common.physicsparams.ncAV > 0 &&
            common.physicsparams.frozenAVflag == 0)
        GetAv<exasim::detail::AbiAdapter>(sol, res, app, master, mesh, tmp, common,
                common.cublasHandle, backend);

    ArraySetValue(res.Ru, zero,
            common.grid.npe*common.components.ncu*common.meshsizes.ne1);
    ArraySetValue(res.Rh, zero,
            common.grid.npf*common.components.ncu*common.meshsizes.nf);

    RuElem<exasim::detail::AbiAdapter>(sol, res, app, master, mesh, tmp, common,
            common.cublasHandle, 0, common.meshsizes.nbe, backend);

    RuFace<exasim::detail::AbiAdapter>(sol, res, app, master, mesh, tmp, common,
            common.cublasHandle, 0, common.meshsizes.nbf, backend);

    if (common.meshsizes.nbf > 0) {
        Int f1 = common.fblks[0]-1;
        Int f2 = common.fblks[3*(common.meshsizes.nbf-1)+1];
        PutFaceNodes(res.Ru, res.Rh, mesh.facecon, common.grid.npf,
                common.components.ncu, common.grid.npe,
                common.components.ncu, f1, f2);
    }

    ArrayCopy(Ru, res.Ru,
            common.grid.npe*common.components.ncu*common.meshsizes.ne1);
}

void VerifyRuDerivFromUFiniteDifference(dstype* A, dstype* u,
        solstruct &sol, resstruct &res, appstruct &app,
        ExasimDriverABI& driver_abi, masterstruct &master, meshstruct &mesh,
        tempstruct &tmp, commonstruct &common, dstype eps)
{
    Int backend = common.backend;
    Int npe = common.grid.npe;
    Int ncu = common.components.ncu;
    Int ne = common.meshsizes.ne1;
    Int nlocu = npe*ncu;
    Int Nu = nlocu*ne;
    Int szA = nlocu*nlocu*ne;

    if (backend > 1)
        return;

    if (eps <= zero)
        eps = 1.0e-6;

    dstype *Afd = nullptr, *diff = nullptr;
    dstype *up = nullptr, *um = nullptr, *Rup = nullptr, *Rum = nullptr;

    TemplateMalloc(&Afd, szA, backend);
    TemplateMalloc(&diff, szA, backend);
    TemplateMalloc(&up, Nu, backend);
    TemplateMalloc(&um, Nu, backend);
    TemplateMalloc(&Rup, Nu, backend);
    TemplateMalloc(&Rum, Nu, backend);

    ArraySetValue(Afd, zero, szA);

    for (Int col = 0; col < Nu; col++) {
        LDGJacTestPerturbU(up, u, col, eps, npe, ncu, ne);
        LDGJacTestPerturbU(um, u, col, -eps, npe, ncu, ne);
        LDGJacTestRuFromU(Rup, up, sol, res, app, driver_abi, master, mesh,
                tmp, common);
        LDGJacTestRuFromU(Rum, um, sol, res, app, driver_abi, master, mesh,
                tmp, common);
        LDGJacTestStoreFDColumn(Afd, Rup, Rum, col, nlocu, eps, npe, ncu);
    }

    for (Int e = 0; e < ne; e++) {
        Int n = nlocu;
        ArrayAXPBY(diff, &A[n*n*e], &Afd[n*n*e], one, minusone, n*n);
        dstype normA = NORM(common.cublasHandle, n*n, &A[n*n*e], backend);
        dstype normAfd = NORM(common.cublasHandle, n*n, &Afd[n*n*e], backend);
        dstype errMinus = NORM(common.cublasHandle, n*n, diff, backend);

        ArrayAXPBY(diff, &A[n*n*e], &Afd[n*n*e], one, one, n*n);
        dstype errPlus = NORM(common.cublasHandle, n*n, diff, backend);

        cout << "Rank " << common.mpiRank << ", element " << e
             << ": BlockJacobianLDG finite difference comparison: "
             << "||A|| = " << scientific << normA
             << ", ||Afd|| = " << normAfd
             << "||A-Afd|| = " << scientific << errMinus
             << ", rel = " << errMinus/(normA + 1.0e-14)
             << ", ||A+Afd|| = " << errPlus
             << ", rel = " << errPlus/(normA + 1.0e-14)
             << endl;
    }

    if (Afd != nullptr) TemplateFree(Afd, backend);
    if (diff != nullptr) TemplateFree(diff, backend);
    if (up != nullptr) TemplateFree(up, backend);
    if (um != nullptr) TemplateFree(um, backend);
    if (Rup != nullptr) TemplateFree(Rup, backend);
    if (Rum != nullptr) TemplateFree(Rum, backend);
}
