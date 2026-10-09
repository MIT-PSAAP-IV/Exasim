// mk_setup.inc -- shared case setup for the harnesses (textually included inside main's body lambda):
// the state exactly as the first DIRK stage leaves it before the nonlinear solve.
    // ---- state exactly as the first DIRK stage leaves it before the nonlinear solve ----
    if (common.timeparams.tdep == 1) {
        common.timestate.currentstep = 0;
        PreviousSolutions(sol, m->solv.sys, common, backend);
        common.timestate.currentstage = 0;
        common.timestate.time = common.timestate.time + common.dt[0] * common.DIRKcoeff_t[0];
        UpdateSource(sol, m->solv.sys, app, d.driver_abi, res, common, backend);
    }
    if (common.physicsparams.ncAV > 0 && common.physicsparams.frozenAVflag > 0) {   // SteadyProblem's frozen-AV preparation, verbatim
        const Int ncAV = common.physicsparams.ncAV;
        dstype* avField = &res.Rq[0]; dstype* utm = &res.Rq[npe * ncAV * ne];
        m->residual.evalAVfield(avField, backend);
        if (common.physicsparams.AVsmoothingMethod != 0) return fail("AV smoothing method not replicated");
        for (Int iav = 0; iav < common.physicsparams.AVsmoothingIter; iav++) {
            d.DG2CG2(avField, avField, utm, ncAV, ncAV, ncAV, backend);
#ifdef HAVE_MPI
            if (nprocs > 1) {   // halo exchange of the smoothed AV field (SteadyProblem)
                const Int bsz = npe*ncAV;
                GetArrayAtIndex(tmp.buffsend, avField, mesh.elemsendodg, bsz*common.nelemsend);
                hipDeviceSynchronize();
                Int psend = 0, precv = 0, rq = 0;
                for (Int n = 0; n < common.nnbsd; n++) { const Int nsend = common.elemsendpts[n]*bsz;
                    if (nsend > 0) { MPI_Isend(&tmp.buffsend[psend], nsend, mpi_type<dstype>(), common.nbsd[n], 0, EXASIM_COMM_LOCAL, &common.requests[rq]);
                                     psend += nsend; rq++; } }
                for (Int n = 0; n < common.nnbsd; n++) { const Int nrecv = common.elemrecvpts[n]*bsz;
                    if (nrecv > 0) { MPI_Irecv(&tmp.buffrecv[precv], nrecv, mpi_type<dstype>(), common.nbsd[n], 0, EXASIM_COMM_LOCAL, &common.requests[rq]);
                                     precv += nrecv; rq++; } }
                MPI_Waitall(rq, common.requests, common.statuses);
                PutArrayAtIndex(avField, tmp.buffrecv, mesh.elemrecvodg, bsz*common.nelemrecv);
            }
#endif
        }
        ArrayInsert(sol.odg, avField, npe, nco, ne, 0, npe, nco - ncAV, nco, 0, ne);
    }
    if (nco > 0) {
        for (Int j = 0; j < nbe; j++) { Int e1 = common.eblks[3*j] - 1, e2 = common.eblks[3*j+1];
            GetElemNodes(tmp.tempn, sol.odg, npe, nco, 0, nco, e1, e2);
            Node2Gauss(handle, &sol.odgg[nge*nco*e1], tmp.tempn, master.shapegt, nge, npe, (e2-e1)*nco, backend); }
        for (Int j = 0; j < nbf; j++) { Int f1 = common.fblks[3*j] - 1, f2 = common.fblks[3*j+1];
            GetFaceNodes(tmp.tempn, sol.odg, mesh.facecon, npf, nco, npe, nco, f1, f2, 1);
            Node2Gauss(handle, &sol.og1[common.grid.ngf*nco*f1], tmp.tempn, master.shapfgt, common.grid.ngf, npf, (f2-f1)*nco, backend);
            GetFaceNodes(tmp.tempn, sol.odg, mesh.facecon, npf, nco, npe, nco, f1, f2, 2);
            Node2Gauss(handle, &sol.og2[common.grid.ngf*nco*f1], tmp.tempn, master.shapfgt, common.grid.ngf, npf, (f2-f1)*nco, backend); }
    }
    Kokkos::fence();
