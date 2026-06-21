import os
from .. import Preprocessing
from .. import Gencode
from .. import config
from .fetchsolution import fetchsolution
from .getsolutions import getsolutions
from .generatecode import generatecode

def exasim(pde,mesh):

    if isinstance(pde, list):
        nmodels = len(pde);      
    else:
        nmodels = 1
    
    res = None;
    if nmodels == 1:
        # search compilers and set options
        pde = Gencode.setcompilers(pde);

        # generate input files and store them in datain folder
        pde, mesh, master, dmd = Preprocessing.preprocessing(pde,mesh);

        # generate source codes and store them in app folder
        Gencode.gencode(pde);

        # compile source codes to build an executable file and store it in build folder
        compilerstr = Gencode.cmakecompile(pde);
        # compilerstr = Gencode.compilepdemodel(pde);

        # When pde['exportapp'] is set, package a relocatable "data transfer
        # app" bundle INSTEAD of running the simulation here: a production run
        # can be prohibitively expensive on the local machine, so we export it
        # to be built and run elsewhere. pde['buildandrun'] (default True)
        # controls whether exportapp builds+runs the bundle in a scratch dir to
        # verify it; set it False to export without any local build/run.
        pde['vistime'] = [];
        if pde.get('exportapp'):
            Gencode.exportapp(pde, pde['exportapp'], build=pde.get('buildandrun', True));
            runstr = None;
            sol = None;
        else:
            runstr = Gencode.runcode(pde, 1);

            # get solution from output files in dataout folder (per-model strn
            # dir; strn="" for model 0 keeps the historical datapath/dataout).
            strn = config.model_strn(pde)
            dataout_dir = os.path.join(pde['datapath'], "dataout", strn) if strn \
                else (pde['datapath'] + "/dataout")
            sol = fetchsolution(pde,master,dmd, dataout_dir);
            #sol, _, _ = getsolutions(pde, dmd);

            if pde['saveResNorm']:
                fn = "dataout/out_residualnorms0.bin";
                tm = fromfile(open(fn, "r"), dtype=float64);
                ne = int(round(size(tm)/(4)));
                tm = reshape(tm,[4,ne],'F');
                res = tm.transpose();

    else:        
        master = [None] * nmodels
        dmd = [None] * nmodels
        sol = [None] * nmodels
        res = [None] * nmodels

        # Combined multi-PDE: model m occupies slot m. Assigning modelnumber=m
        # gives each model a distinct modelid (100+m), a distinct kernel/build
        # dir (config.model_builddir) and distinct datain/dataout subdirs, so
        # the N models generate without clobbering and link into one exasimapp.
        for m in range(0, nmodels):
            pde[m]['modelnumber'] = m

        # preprocess and generate code (own kernel dir) for all PDE models
        for m in range(0, nmodels):
            pde[m],mesh[m],master[m],dmd[m] = generatecode(pde[m],mesh[m])[0:4];

        # build ONE executable that dispatches all model ids, then run it with
        # one (datain, dataout) pair per model (the solver's multi-model CLI).
        compilerstr = Gencode.cmakecompile_combined(pde);

        # Export the combined bundle INSTEAD of running locally when requested
        # (see the single-model branch). pde[0]['buildandrun'] (default True)
        # gates the scratch build+run verification.
        if pde[0].get('exportapp'):
            Gencode.exportapp_combined(pde, pde[0]['exportapp'], build=pde[0].get('buildandrun', True));
            runstr = None;
        else:
            runstr = Gencode.runcode_combined(pde);

            # get solution from each model's own dataout subdir
            for m in range(0, nmodels):
                strn = config.model_strn(pde[m])
                doutdir = os.path.join(pde[m]['datapath'], "dataout", strn) if strn \
                    else (pde[m]['datapath'] + "/dataout")
                sol[m] = fetchsolution(pde[m],master[m],dmd[m], doutdir);

                if pde[m]['saveResNorm']:
                    fn = os.path.join(doutdir, "out_residualnorms0.bin");
                    tm = fromfile(open(fn, "r"), dtype=float64);
                    ne = int(round(size(tm)/(4)));
                    tm = reshape(tm,[4,ne],'F');
                    res[m] = tm.transpose();

    return sol,pde,mesh,master,dmd,compilerstr,runstr,res
