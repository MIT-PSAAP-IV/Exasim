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
    [physicsparamcases, hasPhysicsParamSweep] = normalizephysicsparamsweep(pde);
    if hasPhysicsParamSweep
        pde.physicsparam = physicsparamcases(1,:);
    end

    % generate input files and store them in datain folder
    [pde,mesh,master,dmd] = preprocessing(pde,mesh);
    writeappbase = [];
    if hasPhysicsParamSweep
        writeappbase = writeapptemplate(pde);
    end

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
        if hasPhysicsParamSweep
            ncases = size(physicsparamcases, 1);
            sol = cell(ncases, 1);
            runstr = cell(ncases, 1);
            res = cell(ncases, 1);
            pde.paramcaseoutputdirs = strings(ncases, 1);
            for icase = 1:ncases
                pdecase = writeappbase;
                pdecase.physicsparam = physicsparamcases(icase,:);
                pdecase.dataoutpath = paramcaseoutputdir(pdecase, icase);
                if exist(char(pdecase.dataoutpath), 'dir') == 0
                    mkdir(char(pdecase.dataoutpath));
                end
                writephysicsparamcase(pdecase.dataoutpath, pdecase.physicsparam);

                pdecase = writeapp(pdecase, appbinfile(pdecase), 'native');
                runstr{icase} = runcode(pdecase, 1);
                sol{icase} = fetchsolution(pdecase,master,dmd,pdecase.dataoutpath);
                if pdecase.saveResNorm
                    fn = pdecase.dataoutpath + "/out_residualnorms0.bin";
                    fileID = fopen(char(fn),'r'); res{icase} = fread(fileID,'double'); fclose(fileID);
                    res{icase} = reshape(res{icase},4,[])';
                end
                pde.paramcaseoutputdirs(icase) = pdecase.dataoutpath;
            end
            writephysicsparamsweepmanifest(pde.datapath + "/dataout" + model_strn(pde), physicsparamcases, pde.paramcaseoutputdirs);
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

end

function [cases, hassweep] = normalizephysicsparamsweep(pde)
base = pde.physicsparam(:).';
hassweep = false;
if isfield(pde, 'physicsparamsweep') && ~isempty(pde.physicsparamsweep)
    cases = physicsparamsweepcases(pde.physicsparamsweep, numel(base));
    hassweep = size(cases, 1) > 1;
elseif isnumeric(pde.physicsparam) && ismatrix(pde.physicsparam) && ...
        size(pde.physicsparam,1) > 1 && size(pde.physicsparam,2) > 1
    cases = pde.physicsparam;
    validatephysicsparamcases(cases, size(cases,2));
    hassweep = true;
else
    cases = base;
end
end

function cases = physicsparamsweepcases(spec, nparam)
if isstruct(spec)
    if isfield(spec, 'samples')
        cases = physicsparamsweepcases(spec.samples, nparam);
    elseif isfield(spec, 'values')
        cases = physicsparamsweepcases(spec.values, nparam);
    elseif isfield(spec, 'grid')
        cases = physicsparamgridcases(spec.grid, nparam);
    else
        error("physicsparamsweep struct must contain samples, values, or grid.");
    end
elseif iscell(spec)
    cases = zeros(numel(spec), nparam);
    for i = 1:numel(spec)
        v = spec{i}(:).';
        if numel(v) ~= nparam
            error("physicsparamsweep case %d has %d parameters; expected %d.", i, numel(v), nparam);
        end
        cases(i,:) = v;
    end
elseif isnumeric(spec)
    if isvector(spec)
        if nparam == 1
            cases = spec(:);
        else
            cases = spec(:).';
        end
    else
        cases = spec;
    end
    validatephysicsparamcases(cases, nparam);
else
    error("physicsparamsweep must be numeric, a cell array, or a struct.");
end
end

function cases = physicsparamgridcases(grid, nparam)
if ~iscell(grid) || numel(grid) ~= nparam
    error("physicsparamsweep.grid must be a cell array with one value vector per physics parameter.");
end
[meshgridvalues{1:nparam}] = ndgrid(grid{:});
ncases = numel(meshgridvalues{1});
cases = zeros(ncases, nparam);
for j = 1:nparam
    cases(:,j) = meshgridvalues{j}(:);
end
end

function validatephysicsparamcases(cases, nparam)
if ndims(cases) ~= 2 || size(cases,2) ~= nparam
    error("Each physicsparamsweep row must contain %d physics parameters.", nparam);
end
if ~isnumeric(cases) || any(~isfinite(cases(:)))
    error("physicsparamsweep cases must contain finite numeric values.");
end
end

function outdir = paramcaseoutputdir(pde, icase)
outdir = pde.datapath + "/dataout" + model_strn(pde) + "/paramcase_" + sprintf("%04d", icase);
end

function writephysicsparamcase(outdir, values)
dlmwrite(char(outdir + "/physicsparam.txt"), values(:).', 'delimiter', ' ', 'precision', '%.17g');
end

function writephysicsparamsweepmanifest(baseout, cases, outdirs)
fid = fopen(char(baseout + "/physicsparam_sweep_manifest.txt"), 'w');
if fid < 0
    error("Unable to write physicsparam sweep manifest in %s.", baseout);
end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, "ncases %d\n", size(cases,1));
fprintf(fid, "nparam %d\n", size(cases,2));
for i = 1:size(cases,1)
    fprintf(fid, "case %d %s", i, outdirs(i));
    fprintf(fid, " %.17g", cases(i,:));
    fprintf(fid, "\n");
end
end

function app = writeapptemplate(app)
app.flag = app.flag(19:end);
app.problem = app.problem(33:end);
app.factor = app.factor(6:end);
app.solversparam = app.solversparam(5:end);
end

function filename = appbinfile(app)
filename = app.datapath + "/datain" + model_strn(app) + "/app.bin";
end
