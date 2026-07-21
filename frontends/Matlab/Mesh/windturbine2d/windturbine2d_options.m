function opts = windturbine2d_options(varargin)
%WINDTURBINE2D_OPTIONS Default parameters for the 2-D turbine mesh workflow.
%
%   opts = windturbine2d_options()
%   opts = windturbine2d_options('Nb', 4, 'Rrotor', 6)

root = local_exasim_root();

opts.Nb = 3;
opts.Rrotor = 5;
opts.Rfar = 10;
opts.backgroundMeshMode = 'graded';
opts.backgroundMeshSize = 0.8;
opts.backgroundMeshSizeFar = 1.0;
opts.backgroundMeshSizeNearBlade = 0.35;
opts.backgroundMeshSizeInterface = [];
opts.backgroundMeshDistMin = [];
opts.backgroundMeshSizeTransition = 2.0;
opts.backgroundMeshGrowthRate = 1.3;
opts.backgroundMeshMinSize = 0.25;
opts.backgroundMeshMaxSize = 1.2;
opts.farfieldBoundarySpacing = [];
opts.minFarfieldPoints = 32;
opts.maxFarfieldPoints = 256;
opts.plotFarfieldBoundary = false;
opts.backgroundElementType = "tri";
opts.backgroundElemtype = [];
opts.preserveInterfaceSegments = true;
opts.allowNonconformingQuadInterface = false;
opts.porder = 2;
opts.nodetype = 1;
opts.gmsh = "gmsh";
opts.gmshVersion = "";
opts.workdir = fullfile(tempdir, 'exasim_windturbine2d');
opts.runGmsh = true;
opts.plot = true;
opts.plotDiagnostics = false;

opts.airfoilFile = fullfile(root, 'examples', 'NavierStokes', ...
    'eppler3d', 'epp387_smoothed');
opts.airfoilScale = 1;
opts.airfoilAngleOffset = 0;

opts.cmesh.nxw = 11;
opts.cmesh.nflr = 21;
opts.cmesh.nflf = 21;
opts.cmesh.nfuf = 23;
opts.cmesh.nfur = 29;
opts.cmesh.nr = 15;
opts.cmesh.TEC = 2;
opts.cmesh.spr = [10, 10, 10, 10, 10, 10, 10]*2;
opts.cmesh.yref = [0.1 0.3];
opts.cmesh.lw = 0.5;
opts.cmesh.ll = 0.01;
opts.cmesh.wakeopts = struct('expansionRatio', 2.0, ...
    'nstations', opts.cmesh.nxw + 1, 'verbose', false);

opts.mergeTolerance = 1e-7;
opts.boundaryTolerance = 1e-7;

if mod(numel(varargin), 2) ~= 0
    error('windturbine2d_options expects name/value pairs.');
end
for i = 1:2:numel(varargin)
    opts = local_set_option(opts, varargin{i}, varargin{i+1});
end

end

function root = local_exasim_root()
root = fileparts(mfilename('fullpath'));
for i = 1:8
    if exist(fullfile(root, 'examples'), 'dir') && ...
            exist(fullfile(root, 'frontends'), 'dir')
        return;
    end
    root = fileparts(root);
end
error('Could not locate Exasim repository root.');
end

function opts = local_set_option(opts, name, value)
parts = split(string(name), '.');
if numel(parts) == 1
    opts.(parts{1}) = value;
elseif numel(parts) == 2
    opts.(parts{1}).(parts{2}) = value;
else
    error('Unsupported nested option name: %s.', string(name));
end
end
