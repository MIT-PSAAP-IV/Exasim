inline void WriteAcceptedVerificationFields(
    CSolution<exasim::detail::AbiAdapter>& model, Int continuationIteration,
    Int backend)
{
  const char *verificationEnvironment = std::getenv("EXASIM_MESHADAPT_VERIFY");
  if (verificationEnvironment == nullptr || string(verificationEnvironment) == "0" ||
      string(verificationEnvironment) == "") return;

  const Int rank = model.disc.common.mpiRank-
                   model.disc.common.outputparams.fileoffset;
  const string filename = model.disc.common.fileout + "_meshadapt_aviter" +
    NumberToString(continuationIteration) + "_flow_solution_np" +
    NumberToString(rank) + ".bin";
  writearray2file(filename, model.disc.sol.udg,
                  model.disc.sol.szudg, backend);
}

void avdistfunc(CSolution<exasim::detail::AbiAdapter>** pdemodel, ofstream* out, Int nummodels, Int backend)
{  
  for (int i=0; i<nummodels; i++) 
    pdemodel[i]->InitSolution(backend); 
  
  Int aviter = pdemodel[0]->disc.app.szavparam/2;
  for (Int n=0; n<aviter; n++) {
    if (pdemodel[0]->disc.common.mpiRank==0)
      printf("AV continuation iteration: %d\n", n+1);
    
    for (Int i=0; i<nummodels; i++)
      pdemodel[i]->SaveContinuationState(backend);

    bool localAccepted = true;
    for (Int i=0; i<nummodels; i++) {

      auto& model = *pdemodel[i];
      const Int m = model.disc.app.szphysicsparam;
      ArrayCopy(&model.disc.app.physicsparam[m-2],
                &model.disc.app.avparam[2*n], 2);
      if (model.disc.common.meshadaptparams.enabled)
          model.UpdateWallDistance(n+1, backend);
      model.PrepareArtificialViscosity(n == 0, n+1, backend);
      const SolveStatus status = model.disc.common.timeparams.tdep == 1
          ? model.DIRKonly(out[i], backend)
          : model.SteadyProblem(out[i], backend, true);
      // Nonlinear convergence is advisory here; finite, physically valid states are acceptable.
      const bool physicallyValid = status.finite && model.ValidatePhysicalState(backend);
      localAccepted = localAccepted && status.finite && physicallyValid;
    }

    int accepted = localAccepted ? 1 : 0;
#ifdef HAVE_MPI
    MPI_Allreduce(MPI_IN_PLACE, &accepted, 1, MPI_INT, MPI_MIN, EXASIM_COMM_WORLD);
#endif
    if (accepted == 0) {
      for (Int i=0; i<nummodels; i++)
        pdemodel[i]->RestoreContinuationState(backend);
      if (pdemodel[0]->disc.common.mpiRank==0)
        printf("AV continuation iteration %d rejected; restored iteration %d state.\n",
               n+1, n);
      break;
    }

    for (Int i=0; i<nummodels; i++)
      WriteAcceptedVerificationFields(*pdemodel[i], n+1, backend);

    if (n+1 < aviter) {
      bool meshAccepted = true;
      for (Int i=0; i<nummodels; i++) {
        if (pdemodel[i]->disc.common.meshadaptparams.enabled)
          meshAccepted = pdemodel[i]->AdaptMesh(backend, n+1) && meshAccepted;
      }
#ifdef HAVE_MPI
      int acceptedMesh = meshAccepted ? 1 : 0;
      MPI_Allreduce(MPI_IN_PLACE, &acceptedMesh, 1, MPI_INT,
                    MPI_MIN, EXASIM_COMM_WORLD);
      meshAccepted = acceptedMesh != 0;
#endif
      if (!meshAccepted) {
        for (Int i=0; i<nummodels; i++)
          pdemodel[i]->RestoreContinuationState(backend);
        if (pdemodel[0]->disc.common.mpiRank==0)
          printf("Mesh adaptation after AV continuation iteration %d rejected; "
                 "restored iteration %d state.\n", n+1, n);
        break;
      }
    }
  }

  // No rollback can occur after the continuation loop; release its full-state snapshots.
  for (Int i=0; i<nummodels; i++)
    pdemodel[i]->ClearContinuationState();
  
  for (int i=0; i<nummodels; i++) {
    string fn1 = pdemodel[i]->disc.common.fileout + "vdg_np" + NumberToString(pdemodel[i]->disc.common.mpiRank-pdemodel[i]->disc.common.outputparams.fileoffset) + ".bin";
    writearray2file(fn1, pdemodel[i]->disc.sol.odg, pdemodel[i]->disc.common.sizes.ndofodg1, backend);

    if (pdemodel[i]->disc.common.meshadaptparams.enabled)
    {
      string fn2 = pdemodel[i]->disc.common.fileout + "xdg_np" + NumberToString(pdemodel[i]->disc.common.mpiRank-pdemodel[i]->disc.common.outputparams.fileoffset) + ".bin";
      const Int ndofxdg1 = pdemodel[i]->disc.common.grid.npe *
                           pdemodel[i]->disc.common.components.ncx *
                           pdemodel[i]->disc.common.meshsizes.ne1;
      writearray2file(fn2, pdemodel[i]->disc.sol.xdg, ndofxdg1, backend);
    }

    pdemodel[i]->writer.SaveSolutions(backend);    
    pdemodel[i]->writer.SaveSolutionsOnBoundary(backend);         
    if (pdemodel[i]->vis.savemode > 0)
      pdemodel[i]->writer.SaveParaview(backend);
    if (pdemodel[i]->disc.common.components.nce>0)
      pdemodel[i]->writer.SaveOutputCG(backend);            
  }
}
