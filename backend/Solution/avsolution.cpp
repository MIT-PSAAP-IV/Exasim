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
      pdemodel[i]->SteadyProblem(out[i], backend);        
    }
  }
  
  for (int i=0; i<nummodels; i++) {
    string fn1 = pdemodel[i]->disc.common.fileout + "vdg_np" + NumberToString(pdemodel[i]->disc.common.mpiRank-pdemodel[i]->disc.common.outputparams.fileoffset) + ".bin";
    writearray2file(fn1, pdemodel[i]->disc.sol.odg, pdemodel[i]->disc.common.sizes.ndofodg1, backend);

    pdemodel[i]->writer.SaveSolutions(backend);    
    pdemodel[i]->writer.SaveSolutionsOnBoundary(backend);         
    if (pdemodel[i]->vis.savemode > 0)
      pdemodel[i]->writer.SaveParaview(backend);
    if (pdemodel[i]->disc.common.components.nce>0)
      pdemodel[i]->writer.SaveOutputCG(backend);            
  }
}

