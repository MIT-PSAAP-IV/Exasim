"""Julia version of the axisymmetric ideal-gas ISOQ mesh-adaptivity case."""

repository = dirname(dirname(dirname(dirname(@__DIR__))))
source_install = joinpath(dirname(repository), "exasim_install")
canonical_kernels = joinpath(@__DIR__, "canonical_kernels")
if !haskey(ENV, "EXASIM_PREFIX") &&
   isfile(joinpath(source_install, "lib", "cmake", "Exasim", "ExasimConfig.cmake"))
    ENV["EXASIM_PREFIX"] = source_install
end
pushfirst!(LOAD_PATH, joinpath(repository, "frontends", "Julia", "Exasim"))
using DelimitedFiles
using Exasim

include(joinpath(@__DIR__, "pdemodel.jl"))
include(joinpath(@__DIR__, "isoq_mesh.jl"))
include(joinpath(@__DIR__, "pdemodel_solid.jl"))
using .IsoqSolidMesh

function build_case(run_directory=joinpath(@__DIR__, "julia_run"))
    pde, _ = Exasim.initializeexasim()
    pde.model = "ModelD"
    # All frontends compile the same checked-in generated model sources.
    pde.codegenerator = "prebuilt"
    pde.kernelsource = joinpath(canonical_kernels, "fluid")
    pde.canonicalmaster = joinpath(@__DIR__, "canonical_mesh", "fluid_master.bin")
    pde.modelfile = "pdemodel_fluid"
    pde.platform = "cpu"
    pde.mpiprocs = 4
    pde.porder = 2
    pde.pgauss = 2 * pde.porder
    pde.hybrid = 1
    pde.debugmode = 0
    pde.saveParaview = 1
    pde.saveSolBouFreq = 1
    pde.ibs = 3
    pde.saveSolBouLoc = 1
    mkpath(run_directory)
    pde.datapath = run_directory
    pde.builddir = joinpath(run_directory, ".exasim")
    pde.buildpath = pde.builddir

    gam, reynolds, prandtl, mach_inf = 1.4, 1.56e5, 0.71, 7.6
    reference_temperature, wall_temperature = 266.5, 300.0
    pressure_inf = 1.0 / (gam * mach_inf^2)
    temperature_inf = pressure_inf / (gam - 1.0)
    density_inf, momentum_x_inf, momentum_y_inf = 1.0, 1.0, 0.0
    energy_inf = 0.5 + pressure_inf / (gam - 1.0)

    pde.AV = 2
    pde.AVcontinuationIter = 8
    pde.AVcontinuationLogScale = 1.5
    pde.AVcoeffStart = 0.002
    pde.AVcoeffEnd = 0.00002
    # Canonical continuation schedule shared by MATLAB, Python, and Julia.
    pde.avparam1 = [
        0.002, 0.0015034374019939735, 0.0011026529166442065,
        0.0007791726457939881, 0.000518085978384667,
        0.0003073582956121101, 0.00013727624306105775, 0.0,
    ]
    pde.avparam2 = [
        0.0, 1.3727624306105766e-06, 3.0735829561211013e-06,
        5.180859783846669e-06, 7.791726457939882e-06,
        1.1026529166442065e-05, 1.5034374019939733e-05, 2.0e-05,
    ]
    pde.AVdistfunction = 1
    pde.distanceboundaryconditions = [3]
    pde.AVsmoothingMethod = 1
    pde.AVHelmholtzCoeff = 0.0001
    av_max_divergence, av_distance_coefficient = 20.0, 100.0

    pde.meshadaptenabled = 0
    pde.meshadaptfield = 2
    pde.meshadaptavcomponent = 1
    pde.meshadaptalpha = 0.5
    pde.meshadaptqmin = 0.2
    pde.meshadaptqmax = 0.8
    pde.meshadaptHelmholtzCoeff = 0.001
    pde.meshadaptforcescale = 0.25
    pde.meshadaptsmoothingpasses = 30
    pde.meshadaptboundaryconditions = [3, 3, 3, 2]

    pde.physicsparam = reshape(
        [
            gam, reynolds, prandtl, mach_inf,
            density_inf, momentum_x_inf, momentum_y_inf, energy_inf,
            temperature_inf, reference_temperature, wall_temperature,
            av_max_divergence, av_distance_coefficient,
            pde.AVcoeffStart, pde.AVcoeffEnd,
        ],
        1, :,
    )
    pde.tau = [4.0]
    pde.GMRESrestart = 500
    pde.GMRESortho = 1
    pde.linearsolvertol = 1.0e-6
    pde.linearsolveriter = 1000
    pde.preconditioner = 1
    pde.RBdim = 0
    pde.ppdegree = 0
    pde.NLtol = 1.0e-6
    pde.NLiter = 10
    pde.matvectol = 1.0e-6

    # Keep the optional serialized extension blocks identical to MATLAB.
    pde.flag = zeros(Int, 1, 0); pde.problem = zeros(Int, 1, 0)
    pde.factor = zeros(Float64, 1, 0); pde.solversparam = zeros(Float64, 1, 0)
    pde.stgdata = zeros(Float64, 1, 0); pde.stgparam = zeros(Float64, 1, 0)
    pde.stgib = zeros(Int, 1, 0); pde.dae_dt = Float64[]

    mesh = load_canonical_mesh("fluid")
    mesh.boundarycondition = reshape([4, 2, 1, 3], :, 1)
    return pde, mesh
end


function run_isoq_case(build_fluid,matching,case_directory;run_directory=joinpath(case_directory,"julia_run"))
    precursor=joinpath(run_directory,"precursor"); fluid,fluid_mesh=build_fluid(precursor)
    fluid_solution,fluid,fluid_mesh,fluid_master,fluid_dmd=Exasim.exasim(fluid,fluid_mesh)[1:5]
    ndims(fluid_solution) == 4 && (fluid_solution = fluid_solution[:,:,:,end])
    fluid_mesh.udg=fluid_solution; vdgbase=joinpath(precursor,"dataout","outvdg")
    if isfile(vdgbase*"_np0.bin"); fluid_mesh.odg=Exasim.Postprocessing.getsolution(vdgbase,fluid_dmd,fluid_master.npe); end
    coupled=joinpath(run_directory,"coupled"); mkpath(coupled); setpaths(p,path)=(p.datapath=path;p.builddir=joinpath(path,".exasim");p.buildpath=p.builddir)
    # The fluid uses quadrilaterals and the solid uses triangles, so both the
    # coincident and independently meshed interfaces require the type-1 path.
    setpaths(fluid,coupled); fluid.modelnumber=0; fluid.interfacetype=1
    fluid_mesh.boundarycondition=reshape([4,2,1,7],:,1); fluid_mesh.interfacecondition=reshape([0,0,0,1],:,1)
    fluid.interfacefluxmap=[4]; fluid.ibs=7; fluid.distanceboundaryconditions=[7]
    fluid.AVdistfunction=0; fluid.AVsmoothingMethod=0; fluid.AV=0
    fluid.AVcontinuationIter=0; fluid.avparam1=Float64[]; fluid.avparam2=Float64[]
    params=vec(fluid.physicsparam); params[end-1:end]=[0.0,fluid.AVcoeffEnd]; fluid.physicsparam=reshape(params,1,:)
    solid,_=Exasim.initializeexasim(); setpaths(solid,coupled)
    solid.model="ModelD"; solid.codegenerator="prebuilt"; solid.kernelsource=joinpath(canonical_kernels,"solid")
    solid.canonicalmaster=joinpath(@__DIR__,"canonical_mesh","solid_master.bin")
    solid.modelfile="pdemodel_solid"
    solid.platform=fluid.platform; solid.mpiprocs=2; solid.hybrid=1
    solid.debugmode=fluid.debugmode; solid.porder=fluid.porder; solid.pgauss=fluid.pgauss; solid.Cxxpreprocessing=1
    solid.GMRESrestart=fluid.GMRESrestart; solid.GMRESortho=fluid.GMRESortho; solid.linearsolvertol=fluid.linearsolvertol
    solid.linearsolveriter=fluid.linearsolveriter; solid.preconditioner=fluid.preconditioner; solid.RBdim=fluid.RBdim; solid.ppdegree=fluid.ppdegree
    solid.NLtol=fluid.NLtol; solid.NLiter=fluid.NLiter; solid.matvectol=fluid.matvectol; solid.modelnumber=1; solid.interfacetype=1; solid.interfacefluxmap=[1]
    solid.flag=zeros(Int,1,0); solid.problem=zeros(Int,1,0); solid.factor=zeros(Float64,1,0)
    solid.solversparam=zeros(Float64,1,0); solid.stgdata=zeros(Float64,1,0); solid.stgparam=zeros(Float64,1,0)
    solid.stgib=zeros(Int,1,0); solid.dae_dt=Float64[]
    gam,mach,tref,twall=params[1],params[4],params[10],params[11]; tinf=1.0/(gam*(gam-1.0)*mach^2); solid_temperature=twall/tref*tinf
    solid.physicsparam=reshape([0.01,solid_temperature],1,:); solid.tau=[1.0]
    solid_mesh=load_canonical_mesh("solid")
    solid_mesh.boundarycondition=reshape([3,2,1],:,1)
    return Exasim.exasim([fluid,solid],[fluid_mesh,solid_mesh])
end

function main()
    return run_isoq_case(build_case, false, @__DIR__)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
