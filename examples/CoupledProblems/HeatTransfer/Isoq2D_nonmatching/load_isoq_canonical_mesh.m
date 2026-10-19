function mesh = load_isoq_canonical_mesh(domain)
%LOAD_ISOQ_CANONICAL_MESH Load the frontend-neutral ISOQ mesh payload.

root = fullfile(fileparts(mfilename('fullpath')),'canonical_mesh');
[~,mesh] = initializeexasim();
mesh.p = read_array(root,domain + "_p");
mesh.t = read_array(root,domain + "_t") + 1;
mesh.f = read_array(root,domain + "_f");
mesh.dgnodes = read_array(root,domain + "_dgnodes");
mesh.porder = 2;
mesh.nd = 2;
mesh.nodetype = 1;
mesh.elemtype = double(domain == "fluid");
[mesh.plocal,mesh.tlocal,mesh.plocfc,mesh.tlocfc, ...
    mesh.permnode,mesh.permedge,mesh.permface] = ...
    mkmasternodes(mesh.porder,mesh.nd,mesh.elemtype,mesh.nodetype);
mesh.perm = mesh.permedge;
mesh.telem = mesh.tlocal;

if domain == "fluid"
    mesh.udg = read_array(root,"fluid_udg");
    mesh.vdg = read_array(root,"fluid_vdg");
    ymin = min(mesh.p(2,:));
    xmax = max(mesh.p(1,:));
    mesh.boundaryexpr = { ...
        @(q) abs(q(2,:)-ymin) < 1.0e-6, ...
        @(q) q(1,:) > xmax-1.0e-4, ...
        @(q) q(1,:) < -1.0e-3 | q(2,:) > 0.1, ...
        @(q) abs(q(1,:)) < 20.0+1.0e-6};
else
    mesh.udg = read_array(root,"solid_udg");
    face = getelemface(2,0);
    [localFace,element] = find(mesh.f == 1);
    vertices = [];
    for i = 1:numel(element)
        vertices = [vertices; mesh.t(face(:,localFace(i)),element(i))]; %#ok<AGROW>
    end
    vertices = unique(vertices);
    startx = min(mesh.p(1,vertices));
    endy = max(mesh.p(2,vertices));
    ymin = min(mesh.p(2,:));
    xmax = max(mesh.p(1,:));
    scale = max(1.0,max(max(mesh.p,[],2)-min(mesh.p,[],2)));
    tol = 1.0e-10*scale;
    mesh.boundaryexpr = { ...
        @(q) ~(abs(q(2,:)-ymin)<tol & q(1,:)>startx+tol | ...
               abs(q(1,:)-xmax)<tol & q(2,:)<endy-tol), ...
        @(q) abs(q(2,:)-ymin) < tol, ...
        @(q) abs(q(1,:)-xmax) < tol};
    mesh.interfacecondition = [1;0;0];
end
end

function value = read_array(root,name)
filename = fullfile(root,name + ".bin");
fid = fopen(filename,'rb');
if fid < 0
    error('Unable to open canonical mesh array %s.',filename);
end
cleanup = onCleanup(@() fclose(fid));
dims = fread(fid,3,'float64=>double').';
if numel(dims) ~= 3 || any(dims < 0) || any(dims ~= floor(dims))
    error('Invalid canonical mesh header in %s.',filename);
end
dims = dims(dims > 0);
if numel(dims) < 2
    error('Invalid canonical mesh rank in %s.',filename);
end
value = fread(fid,prod(dims),'float64=>double');
if numel(value) ~= prod(dims)
    error('Truncated canonical mesh array %s.',filename);
end
value = reshape(value,dims);
end
