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

    # generate input files and store them in datain folder
    pde, mesh, master, dmd = Preprocessing.preprocessing(pde,mesh);

    # resolve auto (-1) modelid -> 100 + modelnumber so this model can coexist
    # with others in one working dir (model 0 stays at 100, byte-identical).
    pde.modelid = Gencode.resolve_modelid(pde);

    # generate source codes and store them in app folder
    Gencode.gencode(pde);

    compilerstr = Gencode.cmakecompile(pde); # use cmake to compile source codes
    #compilerstr = Main.compilepdemodel(pde); # use cmake to compile source codes

    runstr = Gencode.runcode(pde, 1);

    # optionally package a relocatable "data transfer app" bundle (the
    # local build+run above doubles as the bundle's verification step).
    if isdefined(pde, :exportapp) && !isempty(pde.exportapp)
        Gencode.exportapp(pde, pde.exportapp; build=true);
    end

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

