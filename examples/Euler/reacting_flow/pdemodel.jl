"""Julia five-species reacting Euler model."""

include("reacting_kernels.jl")
const NS = 5
const NCH = 8

mass(u,q,w,v,x,t,mu,eta) = ones(NCH)
function flux(u,q,w,v,x,t,mu,eta)
    species = u[1:NS]; density = sum(species)
    momentum_x, momentum_y, total_energy = u[NS+1:NS+3]
    velocity_x, velocity_y = momentum_x/density, momentum_y/density
    p = pressure(w[1]*eta[4], species .* eta[1]) / eta[3]
    enthalpy = total_energy/density + p/density
    qx, qy, artificial = -q[1:NCH], -q[NCH+1:2NCH], v[1]
    result = Matrix{Any}(undef,NCH,2)
    for species_index in 1:NS
        result[species_index,1] = species[species_index]*velocity_x - artificial*qx[species_index]
        result[species_index,2] = species[species_index]*velocity_y - artificial*qy[species_index]
    end
    result[NS+1,1] = momentum_x*velocity_x+p-artificial*qx[NS+1]
    result[NS+2,1] = momentum_y*velocity_x-artificial*qx[NS+2]
    result[NS+3,1] = momentum_x*enthalpy-artificial*qx[NS+3]
    result[NS+1,2] = momentum_x*velocity_y-artificial*qy[NS+1]
    result[NS+2,2] = momentum_y*velocity_y+p-artificial*qy[NS+2]
    result[NS+3,2] = momentum_y*enthalpy-artificial*qy[NS+3]
    result
end
function source(u,q,w,v,x,t,mu,eta)
    densities = u[1:NS].*eta[1]
    vcat(production_rates(densities,w[1]*eta[4]) ./ (eta[1]*eta[2]/eta[8]), zeros(3))
end
function boundary_states(u,uhat,n)
    prescribed = [0.00000065,0.22830098,0.01026010,0.75430704,0.00713123,0.996653279785810,0.0,0.607507866832124]
    inflow, outflow = prescribed-uhat, u-uhat
    wall_state = copy(u); normal_momentum = u[NS+1]*n[1]+u[NS+2]*n[2]
    wall_state[NS+1] -= n[1]*normal_momentum; wall_state[NS+2] -= n[2]*normal_momentum
    wall = wall_state-uhat
    [inflow inflow wall wall inflow outflow]
end
fbouhdg(u,q,w,v,x,t,mu,eta,uhat,n,tau) = boundary_states(u,uhat,n)
fbou(u,q,w,v,x,t,mu,eta,uhat,n,tau) = 0 .* boundary_states(u,uhat,n)
ubou(u,q,w,v,x,t,mu,eta,uhat,n,tau) = 0 .* boundary_states(u,uhat,n)
initu(x,mu,eta) = vcat(mu[1:NS]./eta[1], mu[NS+1:NS+2]./(eta[1]*eta[2]), mu[NS+3]/eta[3])
initw(x,mu,eta) = [1.0]
function eos(u,q,w,v,x,t,mu,eta)
    densities = u[1:NS].*eta[1]; momentum = u[NS+1:NS+2].*(eta[1]*eta[2]); energy = u[NS+3]*eta[3]
    internal = energy - 0.5*sum(momentum.*momentum)/sum(densities)
    temperature_residual(w[1]*eta[4],densities./MW,internal)
end
function sourcew(u,q,w,v,x,t,mu,eta)
    densities = u[1:NS].*eta[1]; momentum = u[NS+1:NS+2].*(eta[1]*eta[2]); energy = u[NS+3]*eta[3]
    internal = mu[23]*(energy-0.5*sum(momentum.*momentum)/sum(densities))
    temperature_residual(w[1]*eta[4],densities./MW,internal)
end
