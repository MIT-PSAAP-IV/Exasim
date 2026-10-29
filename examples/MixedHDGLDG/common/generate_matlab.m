function kernels = generate_matlab(caseDir, buildDir, nd, ncu, nparam)
% Native symbolic MATLAB generation, not a Text2Code export.
% Hybrid=1 generates both native LDG kernels and the HDG derivatives. The
% common runtime input, not this generator, selects the actual rank scheme.
% These five models use no ODG/WDG or generated visualization/QoI callbacks.
caseDir = char(caseDir); buildDir = string(buildDir);
if ~isfolder(caseDir) || ~isfile(fullfile(caseDir, 'pdemodel.m'))
    error('hdgldg:model', 'Missing native MATLAB pdemodel.m.');
end
if ~startsWith(buildDir, '/tmp/') && ~startsWith(buildDir, '/private/tmp/')
    error('hdgldg:output', 'Model build directory must be absolute under /tmp.');
end
if isfolder(buildDir) || isfile(buildDir)
    error('hdgldg:output', 'Model build directory must be new.');
end
if ~ismember(nd, [2 3]) || ncu < 1 || nparam < 0
    error('hdgldg:dimensions', 'Invalid ModelD dimensions.');
end
previousPath = path;
cleanup = onCleanup(@() path(previousPath));
% The source-tree setup only adds frontend folders to this session's path.
run(fullfile(fileparts(fileparts(fileparts(fileparts(mfilename('fullpath'))))), ...
    'frontends', 'Matlab', 'exasim_setup.m'));
addpath(caseDir, '-begin');
clear pdemodel;
[pde, ~] = initializeexasim();
pde.model = "ModelD"; pde.modelfile = "pdemodel";
pde.hybrid = 1; pde.modelnumber = 0;
pde.nd = nd; pde.ncx = nd; pde.ncu = ncu; pde.nch = ncu;
pde.ncq = nd*ncu; pde.nc = (nd+1)*ncu;
pde.nco = 0; pde.ncw = 0; pde.nce = 0;
pde.nsca = 0; pde.nvec = 0; pde.nten = 0;
pde.nbqoi = 0; pde.nvqoi = 0; pde.nsurfq = 0; pde.nmaterialstate = 0;
% Values are symbolic placeholders only; the solver reads common parameters.
pde.physicsparam = zeros(1,nparam); pde.uinf = zeros(1,0); pde.tau = 1;
pde.builddir = buildDir; pde.buildpath = buildDir; pde.datapath = buildDir;
kkgencode(pde);
kernels = fullfile(buildDir, 'kernels');
end
