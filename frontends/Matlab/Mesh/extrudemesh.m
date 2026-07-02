function mesh = extrudemesh(mesh2d,zz)

if nargin < 2
    error('extrudemesh requires mesh2d and zz.');
end

if ~isfield(mesh2d,'p') || ~isfield(mesh2d,'t') || ~isfield(mesh2d,'dgnodes')
    error('mesh2d must contain p, t, and dgnodes.');
end

if ~isfield(mesh2d,'porder')
    error('mesh2d must contain porder.');
end

if size(mesh2d.p,1) ~= 2
    error('mesh2d.p must have two coordinate components.');
end

zz = zz(:).';
if numel(zz) < 2
    error('zz must contain at least two extrusion coordinates.');
end

porder = mesh2d.porder;
plc1d = masternodes(porder,1,1);

nz = length(zz)-1;
tz = [(1:nz); (2:nz+1)]';
dz = zeros(length(plc1d),nz);
for i = 1:nz
    pz = zz(tz(i,:));
    dz(:,i) = (pz(2)-pz(1))*plc1d + pz(1);
end

nxy = size(mesh2d.p,2);
pz = repmat(zz,[nxy 1]);
mesh.p = [repmat(mesh2d.p',[nz+1 1]) pz(:)]';

t2d = mesh2d.t';
[ne2d, nv2d] = size(t2d);

mesh.t = zeros(ne2d*nz, nv2d*2);
for i = 1:nz
    mesh.t(ne2d*(i-1)+1:ne2d*i,:) = [t2d+(i-1)*nxy t2d+i*nxy];    
end
mesh.t = mesh.t';

[np2d,nd2d,ne2d_dg] = size(mesh2d.dgnodes);
if nd2d ~= 2
    error('mesh2d.dgnodes must have two coordinate components.');
end
if ne2d_dg ~= ne2d
    error('mesh2d.t and mesh2d.dgnodes have inconsistent element counts.');
end

xy = extrudesol(mesh2d.dgnodes, porder, nz);
zdg = extrudecoord(zz, porder, np2d, 1, ne2d);
mesh.dgnodes = cat(2,xy,zdg);

end
