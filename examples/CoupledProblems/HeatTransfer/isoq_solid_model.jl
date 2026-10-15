module IsoqSolidModel

include(joinpath(@__DIR__, "solid_heat_model.jl"))
using .SolidHeatModel: mass, flux, fbou, ubou, fbouhdg, fint, initu
source(u, q, w, v, x, t, mu, eta) = -mu[1] * q[2] / x[2]

end
