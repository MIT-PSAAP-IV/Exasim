function  exasim(pde,mesh)


if isa(pde, Array)
    nmodels = length(pde);      
else
    nmodels = 1;   
end
res = []
if nmodels==1
    # search compilers and set options
    pde = Gencode.setcompilers(pde);
    physicsparamcases, hasPhysicsParamSweep = normalizephysicsparamsweep(pde);
    if hasPhysicsParamSweep
        pde.physicsparam = reshape(physicsparamcases[1,:], 1, :);
    end

    # generate input files and store them in datain folder
    pde, mesh, master, dmd = Preprocessing.preprocessing(pde,mesh);
    writeappbase = hasPhysicsParamSweep ? writeapptemplate(pde) : nothing;

    # resolve auto (-1) modelid -> 100 + modelnumber so this model can coexist
    # with others in one working dir (model 0 stays at 100, byte-identical).
    pde.modelid = Gencode.resolve_modelid(pde);

    # generate source codes and store them in app folder
    Gencode.gencode(pde);

    compilerstr = Gencode.cmakecompile(pde); # use cmake to compile source codes
    #compilerstr = Main.compilepdemodel(pde); # use cmake to compile source codes

    # When pde.exportapp is set, package a relocatable "data transfer app"
    # bundle INSTEAD of running the simulation here: a production run can be
    # prohibitively expensive on the local machine, so we export it to be built
    # and run elsewhere. pde.buildandrun (default 1) controls whether exportapp
    # builds+runs the bundle in a scratch dir to verify it; set it to 0 to
    # export without any local build/run.
    if isdefined(pde, :exportapp) && !isempty(pde.exportapp)
        buildandrun = (isdefined(pde, :buildandrun) ? pde.buildandrun : 1) != 0;
        Gencode.exportapp(pde, pde.exportapp; build=buildandrun);
        runstr = "";
        sol = [];
    else
        if hasPhysicsParamSweep
            ncases = size(physicsparamcases, 1);
            sol = Array{Any, 1}(undef, ncases);
            runstr = Array{Any, 1}(undef, ncases);
            res = Array{Any, 1}(undef, ncases);
            pde.paramcaseoutputdirs = Array{String, 1}(undef, ncases);
            warmstart = isdefined(pde, :physicsparamwarmstart) && pde.physicsparamwarmstart == 1;
            previousoutdir = "";

            for icase = 1:ncases
                pdecase = deepcopy(writeappbase);
                pdecase.physicsparam = reshape(physicsparamcases[icase,:], 1, :);
                pdecase.dataoutpath = paramcaseoutputdir(pdecase, icase);
                mkpath(pdecase.dataoutpath);
                writephysicsparamcase(pdecase.dataoutpath, pdecase.physicsparam);

                if warmstart && icase > 1
                    writephysicsparamwarmstart(pdecase, dmd, previousoutdir);
                end
                pdecase = Preprocessing.writeapp(pdecase, appbinfile(pdecase));
                runstr[icase] = Gencode.runcode(pdecase, 1);
                sol[icase] = Postprocessing.fetchsolution(pdecase, master, dmd, pdecase.dataoutpath);
                if pdecase.saveResNorm == 1
                    fn = joinpath(pdecase.dataoutpath, "out_residualnorms0.bin");
                    tm = reinterpret(Float64,read(fn));
                    ne = Int64(round(length(tm)/4));
                    res[icase] = reshape(tm,(4,ne))';
                else
                    res[icase] = [];
                end
                pde.paramcaseoutputdirs[icase] = pdecase.dataoutpath;
                previousoutdir = pdecase.dataoutpath;
            end
            writephysicsparamsweepmanifest(dataoutbasedir(pde), physicsparamcases, pde.paramcaseoutputdirs, warmstart);
        else
            runstr = Gencode.runcode(pde, 1);

            # get solution from output files in this model's dataout dir
            strn = Gencode.model_strn(pde);
            doutdir = isempty(strn) ? joinpath(pde.datapath, "dataout") : joinpath(pde.datapath, "dataout", strn);
            sol = Postprocessing.fetchsolution(pde,master,dmd, doutdir);
            if pde.saveResNorm == 1
                fn = "dataout/out_residualnorms0.bin";
                res = reinterpret(Float64,read(fn));
                ne = Int64(round(length(res)/4));
                res = reshape(res,(4,ne));
                res = res';
            end
        end
    end
else
    master = Array{Any, 1}(undef, nmodels);
    dmd = Array{Any, 1}(undef, nmodels);
    sol = Array{Any, 1}(undef, nmodels);
    res = Array{Any, 1}(undef, nmodels);

    # Combined multi-PDE through the external-model path: model m occupies slot
    # m (modelnumber=m-1 -> distinct modelid 100+slot, kernel dir, datain/dataout
    # subdir), so the N models generate without clobbering and link into ONE
    # exasimapp (one provider that dispatches all ids). Replaces gencodeall.
    for m = 1:nmodels
        pde[m].modelnumber = m - 1;
        pde[m] = Gencode.setcompilers(pde[m]);
        pde[m], mesh[m], master[m], dmd[m] = Preprocessing.preprocessing(pde[m], mesh[m]);
        pde[m].modelid = Gencode.resolve_modelid(pde[m]);
        Gencode.gencode(pde[m]);
    end

    compilerstr = Gencode.cmakecompile_combined(pde);
    runstr = Gencode.runcode_combined(pde);

    # get solution from each model's own dataout subdir
    for m = 1:nmodels
        strn = Gencode.model_strn(pde[m]);
        doutdir = isempty(strn) ? joinpath(pde[m].datapath, "dataout") : joinpath(pde[m].datapath, "dataout", strn);
        sol[m] = Postprocessing.fetchsolution(pde[m], master[m], dmd[m], doutdir);
    end
end

return sol,pde,mesh,master,dmd,compilerstr,runstr,res

end

function normalizephysicsparamsweep(pde)
    base = vec(Float64.(pde.physicsparam));
    if !emptysweepspec(pde.physicsparamsweep)
        cases = physicsparamsweepcases(pde.physicsparamsweep, length(base));
        return cases, size(cases, 1) > 1;
    end

    raw = Float64.(pde.physicsparam);
    if ndims(raw) == 2 && size(raw,1) > 1 && size(raw,2) > 1
        validatephysicsparamcases(raw, size(raw,2));
        return raw, true;
    end

    return reshape(base, 1, :), false;
end

emptysweepspec(spec) = spec === nothing || (applicable(isempty, spec) && isempty(spec))

function physicsparamsweepcases(spec, nparam)
    if spec isa Dict
        if haskey(spec, "samples")
            return physicsparamsweepcases(spec["samples"], nparam);
        elseif haskey(spec, :samples)
            return physicsparamsweepcases(spec[:samples], nparam);
        elseif haskey(spec, "values")
            return physicsparamsweepcases(spec["values"], nparam);
        elseif haskey(spec, :values)
            return physicsparamsweepcases(spec[:values], nparam);
        elseif haskey(spec, "grid")
            return physicsparamgridcases(spec["grid"], nparam);
        elseif haskey(spec, :grid)
            return physicsparamgridcases(spec[:grid], nparam);
        else
            error("physicsparamsweep Dict must contain samples, values, or grid.");
        end
    elseif spec isa NamedTuple
        if haskey(spec, :samples)
            return physicsparamsweepcases(spec.samples, nparam);
        elseif haskey(spec, :values)
            return physicsparamsweepcases(spec.values, nparam);
        elseif haskey(spec, :grid)
            return physicsparamgridcases(spec.grid, nparam);
        else
            error("physicsparamsweep NamedTuple must contain samples, values, or grid.");
        end
    elseif spec isa AbstractMatrix
        cases = Float64.(spec);
    elseif spec isa AbstractVector && !isempty(spec) && first(spec) isa AbstractVector
        cases = zeros(Float64, length(spec), nparam);
        for i = 1:length(spec)
            v = vec(Float64.(spec[i]));
            if length(v) != nparam
                error("physicsparamsweep case $i has $(length(v)) parameters; expected $nparam.");
            end
            cases[i,:] = v;
        end
    elseif spec isa AbstractVector
        values = Float64.(spec);
        cases = nparam == 1 && length(values) > 1 ? reshape(values, :, 1) : reshape(values, 1, :);
    else
        error("physicsparamsweep must be numeric, a vector of vectors, a Dict, or a NamedTuple.");
    end
    validatephysicsparamcases(cases, nparam);
    return cases;
end

function physicsparamgridcases(grid, nparam)
    if length(grid) != nparam
        error("physicsparamsweep.grid must contain one value vector per physics parameter.");
    end
    products = collect(Iterators.product(grid...));
    cases = zeros(Float64, length(products), nparam);
    for i = 1:length(products)
        cases[i,:] = collect(products[i]);
    end
    validatephysicsparamcases(cases, nparam);
    return cases;
end

function validatephysicsparamcases(cases, nparam)
    if ndims(cases) != 2 || size(cases,2) != nparam
        error("Each physicsparamsweep row must contain $nparam physics parameters.");
    end
    if any(!isfinite, cases)
        error("physicsparamsweep cases must contain finite numeric values.");
    end
end

function paramcaseoutputdir(pde, icase)
    return joinpath(dataoutbasedir(pde), "paramcase_" * lpad(string(icase), 4, "0"));
end

function dataoutbasedir(pde)
    strn = Gencode.model_strn(pde);
    return isempty(strn) ? joinpath(pde.datapath, "dataout") : joinpath(pde.datapath, "dataout", strn);
end

function writephysicsparamcase(outdir, values)
    open(joinpath(outdir, "physicsparam.txt"), "w") do io
        println(io, join(string.(vec(values)), " "));
    end
end

function writephysicsparamwarmstart(pde, dmd, previousoutdir)
    udg = getsolutions(joinpath(previousoutdir, "outudg"), dmd);
    udg = udg[:,:,:,end];
    wdg = pde.ncw > 0 ? getsolutions(joinpath(previousoutdir, "outwdg"), dmd) : nothing;
    wdg === nothing || (wdg = wdg[:,:,:,end]);
    strn = Gencode.model_strn(pde);
    datain = joinpath(pde.datapath, "datain", strn);

    for iproc = 1:length(dmd)
        solfile = pde.mpiprocs == 1 ? joinpath(datain, "sol.bin") :
                                      joinpath(datain, "sol" * string(iproc) * ".bin");
        elempart = dmd[iproc].elempart[:];
        rewritewarmstartsolfile(solfile, udg[:,:,elempart],
            wdg === nothing ? nothing : wdg[:,:,elempart]);
    end
end

function rewritewarmstartsolfile(solfile, udgpart, wdgpart)
    data = reinterpret(Float64, read(solfile));
    isempty(data) && error("Warm-start solution file $solfile is empty.");

    nsizeLen = Int(data[1]);
    nsizeStart = 2;
    nsizeEnd = nsizeStart + nsizeLen - 1;
    nsize = Int.(data[nsizeStart:nsizeEnd]);
    ndimsStart = nsizeEnd + 1;
    ndimsEnd = ndimsStart + nsize[1] - 1;
    xdgStart = ndimsEnd + 1;
    xdgEnd = xdgStart + nsize[2] - 1;
    udgStart = xdgEnd + 1;
    udgEnd = udgStart + nsize[3] - 1;
    odgStart = udgEnd + 1;
    odgEnd = odgStart + nsize[4] - 1;
    wdgStart = odgEnd + 1;
    wdgEnd = wdgStart + nsize[5] - 1;

    udgflat = Float64.(vec(udgpart));
    if length(udgflat) != nsize[3]
        error("Warm-start udg size mismatch in $solfile: got $(length(udgflat)) values, expected $(nsize[3]).");
    end
    newnsize = copy(nsize);
    newnsize[3] = length(udgflat);
    wdgflat = data[wdgStart:wdgEnd];

    if wdgpart !== nothing
        wdgflat = Float64.(vec(wdgpart));
        if nsize[5] != 0 && length(wdgflat) != nsize[5]
            error("Warm-start wdg size mismatch in $solfile: got $(length(wdgflat)) values, expected $(nsize[5]).");
        end
        newnsize[5] = length(wdgflat);
    end
    data = vcat(data[1:nsizeStart-1], Float64.(newnsize),
        data[ndimsStart:xdgEnd], udgflat, data[odgStart:odgEnd],
        wdgflat, data[wdgEnd+1:end]);

    open(solfile, "w") do io
        write(io, data);
    end
end

function writephysicsparamsweepmanifest(baseout, cases, outdirs, warmstart)
    open(joinpath(baseout, "physicsparam_sweep_manifest.txt"), "w") do io
        println(io, "ncases ", size(cases,1));
        println(io, "nparam ", size(cases,2));
        println(io, "physicsparamwarmstart ", warmstart ? 1 : 0);
        for i = 1:size(cases,1)
            println(io, "case ", i, " ", outdirs[i], " ", join(string.(vec(cases[i,:])), " "));
        end
    end
end

function writeapptemplate(app)
    template = deepcopy(app);
    template.flag = reshape(template.flag[20:end], 1, :);
    template.problem = reshape(template.problem[29:end], 1, :);
    template.factor = reshape(template.factor[6:end], 1, :);
    template.solversparam = reshape(template.solversparam[5:end], 1, :);
    return template;
end

function appbinfile(pde)
    strn = Gencode.model_strn(pde);
    return joinpath(pde.datapath, "datain", strn, "app.bin");
end
