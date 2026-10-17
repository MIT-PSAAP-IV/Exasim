"""Julia version of the axisymmetric ideal-gas ISOQ mesh-adaptivity case."""

repository = dirname(dirname(dirname(dirname(@__DIR__))))
source_install = joinpath(dirname(repository), "exasim_install")
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
    pde.modelfile = ""
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

    mesh = make_isoq_mesh(pde.porder; radial_shift=5.0e-4)
    mesh.boundarycondition = reshape([4, 2, 1, 3], :, 1)
    distance = wall_distance(mesh, pde.porder)
    # ODG layout: wall distance, filtered AV sensor, mesh-adaptation pressure.
    mesh.odg = zeros(Float64, size(distance, 1), 3, size(distance, 3))
    mesh.odg[:, 1:1, :] .= distance
    npe, _, ne = size(mesh.dgnodes)
    # Match MATLAB's HDG initialization exactly: conservative variables are
    # followed by the two zero gradient blocks in the 12-component UDG array.
    mesh.udg = zeros(Float64, npe, 12, ne)
    mesh.udg[:, 1, :] .= density_inf
    mesh.udg[:, 2, :] .= momentum_x_inf .* tanh.(
        av_distance_coefficient .* distance[:, 1, :]
    )
    mesh.udg[:, 3, :] .= momentum_y_inf
    near_wall_temperature = temperature_inf .* (
        wall_temperature / reference_temperature - 1.0
    ) .* exp.(-av_distance_coefficient .* distance[:, 1, :]) .+ temperature_inf
    mesh.udg[:, 4, :] .= near_wall_temperature .+ 0.5 .* (
        mesh.udg[:, 2, :].^2 .+ mesh.udg[:, 3, :].^2
    )
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
    params=vec(fluid.physicsparam); params[end-1:end]=[0.0,1.5*fluid.AVcoeffEnd]; fluid.physicsparam=reshape(params,1,:); fluid.NLiter=20
    solid,_=Exasim.initializeexasim(); setpaths(solid,coupled)
    solid.model="ModelD"; solid.modelfile="pdemodel_solid"; solid.platform=fluid.platform; solid.mpiprocs=2; solid.hybrid=1
    solid.debugmode=fluid.debugmode; solid.porder=fluid.porder; solid.pgauss=fluid.pgauss; solid.Cxxpreprocessing=0
    solid.GMRESrestart=fluid.GMRESrestart; solid.GMRESortho=fluid.GMRESortho; solid.linearsolvertol=fluid.linearsolvertol
    solid.linearsolveriter=fluid.linearsolveriter; solid.preconditioner=fluid.preconditioner; solid.RBdim=fluid.RBdim; solid.ppdegree=fluid.ppdegree
    solid.NLtol=fluid.NLtol; solid.NLiter=20; solid.matvectol=fluid.matvectol; solid.modelnumber=1; solid.interfacetype=1; solid.interfacefluxmap=[1]
    gam,mach,tref,twall=params[1],params[4],params[10],params[11]; tinf=1.0/(gam*(gam-1.0)*mach^2); solid_temperature=twall/tref*tinf
    solid.physicsparam=reshape([0.01,solid_temperature],1,:); solid.tau=[1.0]
    solid_mesh=IsoqSolidMesh.make_isoq_solid(solid,fluid_mesh,matching;mesh_size=0.0028)
    solid_mesh.udg=zeros(size(solid_mesh.dgnodes,1),3,size(solid_mesh.dgnodes,3)); solid_mesh.udg[:,1,:].=solid_temperature
    return Exasim.exasim([fluid,solid],[fluid_mesh,solid_mesh])
end

function main()
    return run_isoq_case(build_case, false, @__DIR__)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
