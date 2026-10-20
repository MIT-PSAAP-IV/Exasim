"""Julia frontend for the five-species reacting-flow cylinder example."""

using Exasim
include("pdemodel.jl")
include("reacting_mesh.jl")

function build_case(av_coefficient=0.2)
    pde,_=Exasim.initializeexasim(); pde.model="ModelD"; pde.modelfile="pdemodel"; pde.platform="gpu"; pde.mpiprocs=1; pde.hybrid=1
    pde.porder=3; pde.pgauss=6; pde.tau=[1.0]
    species=[1.00555e-9,3.5318161606e-4,1.58723747e-5,1.16691299088e-3,1.103201281e-5]
    momentum_x,momentum_y,total_energy=9.213932,0.0,33563.20282790763
    rho_ref,velocity_ref,temperature_ref=0.001547,5976.0,901.0; energy_ref=rho_ref*velocity_ref^2
    mu_ref,kappa_ref,cp_ref,length_ref=3.8196e-5,0.076685370133649,1182.1920097928833,0.045
    reynolds=rho_ref*length_ref*velocity_ref/mu_ref; prandtl=mu_ref*cp_ref/kappa_ref
    physics=zeros(23); physics[1:5].=species; physics[6:8].=[momentum_x,momentum_y,total_energy]; physics[15]=pde.porder; physics[19:23].=[1.e13,1.e4,prandtl,reynolds,1.0]
    pde.physicsparam=reshape(physics,1,:); pde.externalparam=reshape([rho_ref,velocity_ref,energy_ref,temperature_ref,mu_ref,kappa_ref,cp_ref,length_ref],1,:)
    pde.GMRESrestart=200; pde.linearsolveriter=200; pde.linearsolvertol=1.e-8; pde.NLtol=1.e-4; pde.NLiter=30
    pde.RBdim=0; pde.ppdegree=20; pde.precMatrixType=2; pde.ptcMatrixType=0; pde.matvectol=1.e-6; pde.AV=1
    mesh=make_mesh(pde.porder); distance=wall_distance(mesh,pde.porder); npe,_,ne=size(mesh.dgnodes)
    initial=zeros(npe,8,ne); initial[:,1:5,:].=reshape(species./rho_ref,1,5,1)
    initial[:,6,:].=momentum_x/(rho_ref*velocity_ref).*tanh.(4.0.*distance[:,1,:]); initial[:,7,:].=0.0
    internal=(total_energy-0.5*sum(species)*velocity_ref^2)/energy_ref
    initial[:,8,:].=internal.+0.5.*(initial[:,6,:].^2.+initial[:,7,:].^2)./sum(initial[:,1:5,:],dims=2)[:,1,:]
    mesh.udg=initial; mesh.wdg=ones(npe,1,ne); mesh.odg=zeros(npe,2,ne)
    mesh.odg[:,1,:].=av_coefficient.*tanh.(25.0.*distance[:,1,:]); mesh.odg[:,2,:].=element_scales(mesh).*tanh.(1000.0.*distance[:,1,:])
    pde,mesh
end

function main()
    solution=nothing; pde=nothing; mesh=nothing
    for coefficient in (0.2,0.1,0.05,0.03,0.015,0.01)
        pde,mesh=build_case(coefficient)
        if solution !== nothing; mesh.udg=solution; pde.NLtol=1.e-6; end
        solution,pde,mesh=Exasim.exasim(pde,mesh)[1:3]
    end
    println("Julia reacting-flow solution: ",size(solution)); solution,pde,mesh
end

if abspath(PROGRAM_FILE)==@__FILE__; main(); end
