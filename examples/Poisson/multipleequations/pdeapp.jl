# import the Exasim frontend (installed via `cmake --install` under
# <prefix>/share/exasim/julia, or Pkg.develop'd from frontends/Julia/Exasim)
using Exasim

pde = Array{Any, 1}(undef, 2);
mesh = Array{Any, 1}(undef, 2);

# create pde and mesh for each PDE model
include("pdeapp1.jl"); 
include("pdeapp2.jl"); 

# call exasim to generate and run C++ code to solve the PDE models
sol,pde,mesh,master,dmd,compilerstr,runstr = exasim(pde,mesh);

# visualize the numerical solution of the PDE model using Paraview
for m = 1:length(pde)
    pde[m].visscalars = ["temperature", 1];  # list of scalar fields for visualization
    pde[m].visvectors = ["temperature gradient", [2, 3]]; # list of vector fields for visualization
    pde[m].visfilename = "dataout" * string(m) * "/output";  
    vis(sol[m],pde[m],mesh[m]); # visualize the numerical solution
end
print("Done!");
