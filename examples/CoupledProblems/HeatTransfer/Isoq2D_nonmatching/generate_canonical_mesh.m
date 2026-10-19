% Regenerate the frontend-neutral mesh and initial-state payload.
% Run this script from this example directory with MATLAB.

caseDir = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(fileparts(fileparts(caseDir))));
run(fullfile(repo,'frontends','Matlab','exasim_setup.m'));
addpath(caseDir);

[pde,~] = initializeexasim();
pde.platform = "cpu";
pde.porder = 2;
pde.pgauss = 4;
pde.mpiprocs = 1;
pde.nd = 2;

fluid = mkmesh_isoq2d(pde.porder,5.0e-4);
fluid.boundarycondition = [4 2 1 3];
master = Master(pde);
distance = meshdist3(fluid.f,fluid.dgnodes,master.perm,[4]);
fluid.vdg = zeros(size(fluid.dgnodes,1),3,size(fluid.dgnodes,3));
fluid.vdg(:,1,:) = distance;

gam = 1.4;
Minf = 7.6;
Tref = 266.5;
Twall = 300.0;
pinf = 1/(gam*Minf^2);
Tinf = pinf/(gam-1);
fluid.udg = initu(fluid,{1.0,1.0,0.0,0.5+pinf/(gam-1),0,0,0,0,0,0,0,0});
fluid.udg(:,2,:) = fluid.udg(:,2,:).*tanh(100.0*distance);
fluid.udg(:,3,:) = fluid.udg(:,3,:).*tanh(100.0*distance);
temperature = Tinf*(Twall/Tref-1)*exp(-100.0*distance)+Tinf;
fluid.udg(:,4,:) = temperature + ...
    0.5*(fluid.udg(:,2,:).^2+fluid.udg(:,3,:).^2);

pde2 = pde;
pde2.Cxxpreprocessing = 0;
[solid,~] = mkmesh_isoqsolid(pde2,fluid,0.0028);
solidTemperature = Twall/Tref*Tinf;
solid.udg = zeros(size(solid.dgnodes,1),3,size(solid.dgnodes,3));
solid.udg(:,1,:) = solidTemperature;

root = fullfile(caseDir,'canonical_mesh');
if ~exist(root,'dir'), mkdir(root); end
write_array(root,'fluid_p',fluid.p);
write_array(root,'fluid_t',double(fluid.t-1));
write_array(root,'fluid_f',double(fluid.f));
write_array(root,'fluid_dgnodes',fluid.dgnodes);
write_array(root,'fluid_udg',fluid.udg);
write_array(root,'fluid_vdg',fluid.vdg);
write_array(root,'solid_p',solid.p);
write_array(root,'solid_t',double(solid.t-1));
write_array(root,'solid_f',double(solid.f));
write_array(root,'solid_dgnodes',solid.dgnodes);
write_array(root,'solid_udg',solid.udg);

fprintf('canonical fluid elements: %d\n',size(fluid.t,2));
fprintf('canonical solid elements: %d\n',size(solid.t,2));

function write_array(root,name,value)
dims = zeros(1,3);
dims(1:ndims(value)) = size(value);
fid = fopen(fullfile(root,name + ".bin"),'wb');
assert(fid >= 0);
cleanup = onCleanup(@() fclose(fid));
fwrite(fid,dims,'float64');
fwrite(fid,double(value(:)),'float64');
end
