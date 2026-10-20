run(fullfile(fileparts(mfilename('fullpath')), '..', '..', '..', 'frontends', 'Matlab', 'exasim_setup.m'));

[pde,mesh] = initializeexasim();
pde.model = "ModelD";
pde.modelfile = "pdemodel";
pde.porder = 2;
pde.torder = 2;
pde.nstage = 2;
pde.dt = 0.05*ones(1,200);
pde.soltime = 0:199;
pde.visdt = 0.05;
pde.pgauss = 2*pde.porder;
pde.tau = 1.0;
pde.linearsolvertol = 1e-4;
pde.ppdegree = 6;
pde.RBdim = 0;
pde.GMRESrestart = 50;
pde.linearsolveriter = 51;
pde.precMatrixType = 2;
pde.NLiter = 2;

massIon = 16*1.67e-27;
kB = 1.38e-23;
pde.physicsparam = [massIon, kB, 1.6e-19, 8.7];
pde.mpiprocs = 1;
pde.hybrid = 1;

Re = 6378;
R0 = (Re+100)/Re;
R1 = (Re+500)/Re;
[mesh.p,mesh.t,mesh.dgnodes] = sphereCube(pde.porder,R0,R1,15,10);
mesh.boundaryexpr = {@(p) abs(sum(p.^2,1)-R0^2)<1e-6, @(p) abs(sum(p.^2,1)-R1^2)<1e-6};
mesh.boundarycondition = [1;2];

npe = size(mesh.dgnodes,1);
ne = size(mesh.dgnodes,3);
nvalues = npe*ne;
filename = fullfile(fileparts(mfilename('fullpath')), 'input.nc');
if ~isfile(filename)
    error('ionTest requires %s, which is not distributed with the repository.', filename);
end

mesh.vdg = zeros(npe,14,ne);
mesh.udg = zeros(npe,5,ne);
vnames = {'EX','EY','EZ','BX','BY','BZ','U','V','W','NU','P','L','H','C'};
for i = 1:numel(vnames)
    mesh.vdg(:,i,:) = reshape(read_first_record(filename,vnames{i},nvalues),npe,1,ne);
end
numberDensity = read_first_record(filename,'N',nvalues);
mesh.udg(:,1,:) = reshape(numberDensity,npe,1,ne);
mesh.udg(:,2,:) = reshape(massIon*numberDensity.*read_first_record(filename,'VX',nvalues),npe,1,ne);
mesh.udg(:,3,:) = reshape(massIon*numberDensity.*read_first_record(filename,'VY',nvalues),npe,1,ne);
mesh.udg(:,4,:) = reshape(massIon*numberDensity.*read_first_record(filename,'VZ',nvalues),npe,1,ne);
mesh.udg(:,5,:) = reshape(kB*numberDensity.*read_first_record(filename,'T',nvalues),npe,1,ne);

[sol,pde,mesh] = exasim(pde,mesh);
disp('Done!');

function values = read_first_record(filename,name,count)
raw = ncread(filename,name);
if numel(raw) == count
    values = raw(:);
    return;
end
sz = size(raw);
pointdim = find(sz == count,1);
if isempty(pointdim)
    error('Variable %s has no dimension matching the %d DG points.',name,count);
end
indices = repmat({1},1,ndims(raw));
indices{pointdim} = ':';
values = raw(indices{:});
values = values(:);
end
