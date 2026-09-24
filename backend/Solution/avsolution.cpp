void avdistfunc(CSolution<exasim::detail::AbiAdapter>** pdemodel, ofstream* out, Int nummodels, Int backend)
{  
  for (int i=0; i<nummodels; i++) 
    pdemodel[i]->InitSolution(backend); 
  
  Int aviter = pdemodel[0]->disc.app.szavparam/2;
  for (Int n=0; n<aviter; n++) {
    if (pdemodel[0]->disc.common.mpiRank==0)
      printf("AV continuation iteration: %d\n", n+1);
    
    for (Int i=0; i<nummodels; i++) {
      Int m = pdemodel[i]->disc.app.szphysicsparam;      
      ArrayCopy(&pdemodel[i]->disc.app.physicsparam[m-2], &pdemodel[i]->disc.app.avparam[2*n], 2);
      pdemodel[i]->UpdateWallDistance(n+1, backend);
      pdemodel[i]->PrepareArtificialViscosity(n == 0, n+1, backend);
      if (pdemodel[i]->disc.common.timeparams.tdep == 1) 
          pdemodel[i]->DIRKonly(out[i], backend);      
      else 
          pdemodel[i]->SteadyProblem(out[i], backend);

      const char *verificationEnvironment = std::getenv("EXASIM_MESHADAPT_VERIFY");
      if (verificationEnvironment != nullptr && string(verificationEnvironment) != "0" &&
          string(verificationEnvironment) != "") {
        const Int rank = pdemodel[i]->disc.common.mpiRank-
                         pdemodel[i]->disc.common.outputparams.fileoffset;
        const string filename = pdemodel[i]->disc.common.fileout + "_meshadapt_aviter" +
          NumberToString(n+1) + "_flow_solution_np" + NumberToString(rank) + ".bin";
        writearray2file(filename, pdemodel[i]->disc.sol.udg,
                        pdemodel[i]->disc.sol.szudg, backend);
      }

      if (n+1 < aviter && pdemodel[i]->disc.common.meshadaptparams.enabled)
        pdemodel[i]->AdaptMesh(backend, n+1);
    }
  }
  
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
