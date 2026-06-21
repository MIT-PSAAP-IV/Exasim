function [sol,pde,mesh,master,dmd,compilerstr,runstr,res] = exasim(pde,mesh)

if iscell(pde)
    nmodels = length(pde);
    if nmodels==1
        error("Number of PDE models must be greater than 1");
    end
else
    nmodels = 1;
end

res = [];
if nmodels==1
    % External-model path: isolate this model's kernels/build/datain/dataout by
    % its modelnumber so it can coexist with others in one working dir (model 0
    % is byte-identical to the historical single-model layout).
    pde.combinedmodel = true;
    pde.modelid = resolve_modelid(pde);   % resolve auto (-1) -> 100 + modelnumber

    % generate input files and store them in datain folder
    [pde,mesh,master,dmd] = preprocessing(pde,mesh);

    % generate source codes and store them in app folder
    if pde.gencode==1
      %gencode(pde);
      kkgencode(pde);
      compilerstr = cmakecompile(pde); % use cmake to compile C++ source codes
      %compilerstr = compilepdemodel(pde);
    end

    % When pde.exportapp is set, package a relocatable "data transfer app"
    % bundle INSTEAD of running the simulation here: a production run can be
    % prohibitively expensive on the local machine, so we export it to be built
    % and run elsewhere. pde.buildandrun (default true) controls whether
    % exportapp builds+runs the bundle in a throwaway scratch dir to verify it
    % before hand-off; set it false to export without any local build/run.
    if isfield(pde,'exportapp') && ~isempty(pde.exportapp)
        if isfield(pde,'buildandrun') && ~isempty(pde.buildandrun)
            exportapp(pde, pde.exportapp, pde.buildandrun);
        else
            exportapp(pde, pde.exportapp, true);
        end
        runstr = [];
        sol = [];
    else
        runstr = runcode(pde, 1); % run C++ code

        % get solution from output files in this model's dataout dir
        sol = fetchsolution(pde,master,dmd, pde.datapath + "/dataout" + model_strn(pde));

        % get residual norms from output files in dataout folder
        if pde.saveResNorm
            fileID = fopen('dataout/out_residualnorms0.bin','r'); res = fread(fileID,'double'); fclose(fileID);
            res = reshape(res,4,[])';
        end
    end
else
    master = cell(nmodels,1);
    dmd = cell(nmodels,1);
    sol = cell(nmodels,1);
    res = cell(nmodels,1);

    is_coupling = isfield(mesh{1}, 'interfacecondition') && ...
                  max(mesh{1}.interfacecondition) >= 1;

    if is_coupling
        % ---- legacy two-domain coupling path (interfacecondition) ----------
        % Unchanged: suffixed kernels via kkgencodeall + interface partition +
        % the nummodels>100 MPI-split CLI. (combinedmodel stays unset.)
        nummodels = 100 + pde{1}.mpiprocs;
        mpiprocs = pde{1}.mpiprocs + pde{2}.mpiprocs;

        for m = 1:nmodels
            [pde{m},mesh{m},master{m},dmd{m}] = preprocessing(pde{m},mesh{m});
        end

        [dmd{1},dmd{2},isd1,isd2]=interfacepartition(mesh{1}, dmd{1}, mesh{2}, dmd{2});
        writedmd(dmd{1}, pde{1}, isd1);
        writedmd(dmd{2}, pde{2}, isd2);

        if pde{1}.gencode==1
          for m = 1:nmodels
            kkgencode(pde{m});
          end
          kkgencodeall(nmodels, pde{1}.backendpath + "/Model");
          compilerstr = cmakecompile(pde{1}, mpiprocs);
        end

        runstr = runcode(pde{1}, nummodels, mpiprocs);

        for m = 1:nmodels
            sol{m} = fetchsolution(pde{m},master{m},dmd{m}, pde{m}.buildpath + "/dataout" + num2str(m));
        end
    else
        % ---- combined multi-PDE through the external-model path -----------
        % Each model occupies slot m: modelnumber=m-1 gives a distinct modelid
        % (100+slot), kernel dir and datain/dataout, so the N models generate
        % without clobbering and link into ONE exasimapp (one provider that
        % dispatches all ids). Replaces the broken legacy kkgencodeall path.
        for m = 1:nmodels
            pde{m}.modelnumber = m - 1;
            pde{m}.combinedmodel = true;
            pde{m}.modelid = resolve_modelid(pde{m});   % 100 + slot
            [pde{m},mesh{m},master{m},dmd{m}] = preprocessing(pde{m},mesh{m});
            if pde{m}.gencode==1
                kkgencode(pde{m});
            end
        end

        compilerstr = cmakecompile_combined(pde);
        runstr = runcode_combined(pde);

        for m = 1:nmodels
            sol{m} = fetchsolution(pde{m},master{m},dmd{m}, ...
                                   pde{m}.datapath + "/dataout" + model_strn(pde{m}));
        end
    end
end



