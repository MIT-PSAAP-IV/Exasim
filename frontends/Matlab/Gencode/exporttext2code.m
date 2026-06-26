function dest = exporttext2code(pde, mesh, dest)
%EXPORTTEXT2CODE Export a self-contained Text2Code application package.
%
%   dest = exporttext2code(pde, mesh, dest) writes the files needed by the
%   Exasim text2code executable to reconstruct the application without the
%   original MATLAB script:
%
%     pdemodel.txt
%     pdeapp.txt
%     grid.bin
%     xdg.bin, udg.bin, vdg.bin, wdg.bin when present in mesh
%
%   This differs from exportapp: exportapp packages generated kernels and a
%   C++ driver ready to build, while exporttext2code packages the higher-level
%   Text2Code inputs used to regenerate those kernels.

if nargin < 3 || isempty(dest)
    if isfield(pde, 'exporttext2code') && ~isempty(pde.exporttext2code)
        dest = pde.exporttext2code;
    elseif isfield(pde, 'exportt2c') && ~isempty(pde.exportt2c)
        dest = pde.exportt2c;
    else
        error("exporttext2code: no destination (pass dest or set pde.exporttext2code).");
    end
end

dest = string(exporttext2code_abspath(dest));
if exist(char(dest), 'dir') == 0
    mkdir(char(dest));
end

fprintf("Export Exasim Text2Code package to %s ...\n", dest);

% writeinputfile writes pdeapp.txt plus grid/field binaries in the current
% directory. Reset file fields so the exported package is self-contained and
% uses deterministic relative file names.
pdeout = pde;
pdeout = exporttext2code_infer_dimensions(pdeout, mesh);
if (~isfield(pdeout, 'model') || isempty(pdeout.model)) && isfield(pdeout, 'pdemodel')
    pdeout.model = pdeout.pdemodel;
end

% The model DSL is the source of generated kernels for Text2Code.
genpdemodel(pdeout, dest + "/pdemodel.txt");
pdeout.modelfile = "pdemodel";
if isfield(pdeout, "physicsparamsweep") && ~isempty(pdeout.physicsparamsweep)
    pdeout.physicsparamcases = exporttext2code_sweepcases(pdeout.physicsparamsweep, numel(pdeout.physicsparam));
end
for key = ["meshfile", "xdgfile", "udgfile", "vdgfile", "wdgfile"]
    if isfield(pdeout, key)
        pdeout = rmfield(pdeout, key);
    end
end

olddir = pwd();
cleanup = onCleanup(@() cd(olddir));
cd(char(dest));
writeinputfile("pdeapp.txt", pdeout, mesh);

write_text2code_readme(dest);
fprintf("Exported Text2Code package: %s\n", dest);
end

function pde = exporttext2code_infer_dimensions(pde, mesh)
if isfield(mesh, 'p') && ~isempty(mesh.p)
    nd = size(mesh.p, 1);
    pde.nd = nd;
    pde.ncx = nd;
elseif isfield(pde, 'nd')
    nd = pde.nd;
else
    nd = 1;
end
pde = exporttext2code_infer_model_dimensions(pde);
model = "ModelD";
if isfield(pde, 'model') && ~isempty(pde.model)
    model = string(pde.model);
elseif isfield(pde, 'pdemodel') && ~isempty(pde.pdemodel)
    model = string(pde.pdemodel);
end
if lower(model) == "modelc"
    nc = pde.ncu;
else
    nc = pde.ncu * (nd + 1);
end
if ~isfield(pde, 'nc') || isempty(pde.nc) || pde.nc < nc
    pde.nc = nc;
end
pde.ncq = max(pde.nc - pde.ncu, 0);
if isfield(mesh, 'vdg') && ~isempty(mesh.vdg)
    pde.nco = size(mesh.vdg, 2);
end
if isfield(mesh, 'wdg') && ~isempty(mesh.wdg)
    pde.ncw = size(mesh.wdg, 2);
end
end

function pde = exporttext2code_infer_model_dimensions(pde)
if ~isfield(pde, 'modelfile') || isempty(pde.modelfile)
    if isfield(pde, 'model') && ~isempty(pde.model)
        pde.modelfile = pde.model;
    else
        pde.modelfile = "pdemodel";
    end
end

pdemodelfun = str2func(pde.modelfile);
pdem = pdemodelfun();

nuinf = numel(pde.externalparam);
nparam = numel(pde.physicsparam);
xdgsym = sym('xdg', [pde.ncx 1]);
uinfsym = sym('uinf', [nuinf 1]);
paramsym = sym('param', [nparam 1]);

if isfield(pdem, 'initu')
    udgsym = pdem.initu(xdgsym, paramsym, uinfsym);
    pde.ncu = numel(udgsym(:));
elseif ~isfield(pde, 'ncu') || isempty(pde.ncu)
    pde.ncu = 1;
end

if isfield(pdem, 'initv')
    odgsym = pdem.initv(xdgsym, paramsym, uinfsym);
    pde.nco = numel(odgsym(:));
elseif ~isfield(pde, 'nco') || isempty(pde.nco)
    pde.nco = 0;
end

if isfield(pdem, 'initw')
    wdgsym = pdem.initw(xdgsym, paramsym, uinfsym);
    pde.ncw = numel(wdgsym(:));
elseif ~isfield(pde, 'ncw') || isempty(pde.ncw)
    pde.ncw = 0;
end
end

function cases = exporttext2code_sweepcases(spec, nparam)
if isnumeric(spec)
    if isvector(spec) && nparam == 1
        cases = spec(:);
    else
        cases = spec;
    end
elseif iscell(spec)
    cases = zeros(numel(spec), nparam);
    for i = 1:numel(spec)
        v = spec{i};
        if numel(v) ~= nparam
            error("physicsparamsweep case %d has %d parameters; expected %d.", i, numel(v), nparam);
        end
        cases(i,:) = reshape(v, 1, []);
    end
elseif isstruct(spec)
    if isfield(spec, 'samples')
        cases = exporttext2code_sweepcases(spec.samples, nparam);
    elseif isfield(spec, 'values')
        cases = exporttext2code_sweepcases(spec.values, nparam);
    elseif isfield(spec, 'grid')
        if ~iscell(spec.grid) || numel(spec.grid) ~= nparam
            error("physicsparamsweep.grid must be a cell array with one value vector per physics parameter.");
        end
        grids = cellfun(@(v) v(:), spec.grid, 'UniformOutput', false);
        [meshgrids{1:nparam}] = ndgrid(grids{:});
        cases = zeros(numel(meshgrids{1}), nparam);
        for j = 1:nparam
            cases(:,j) = meshgrids{j}(:);
        end
    else
        error("physicsparamsweep struct must contain samples, values, or grid.");
    end
else
    error("physicsparamsweep must be numeric, a cell array, or a struct.");
end
if size(cases,2) ~= nparam
    error("Each physicsparamsweep row must contain %d physics parameters.", nparam);
end
if any(~isfinite(cases(:)))
    error("physicsparamsweep cases must contain finite numeric values.");
end
end

function p = exporttext2code_abspath(p)
p = char(string(p));
if ~isempty(p) && (p(1) == filesep || contains(p, ':\'))
    % Already absolute on Unix or Windows.
else
    p = fullfile(pwd(), p);
end
end

function write_text2code_readme(dest)
txt = [
"# Exasim Text2Code Export"; ...
""; ...
"This directory contains the high-level Text2Code inputs exported from an Exasim frontend."; ...
""; ...
"Generated files:"; ...
""; ...
"- `pdemodel.txt`: PDE model definition consumed by Text2Code."; ...
"- `pdeapp.txt`: application, mesh, solver, output, and runtime configuration."; ...
"- `grid.bin`: mesh coordinates and connectivity."; ...
"- `xdg.bin`, `udg.bin`, `vdg.bin`, `wdg.bin`: optional field data written only when present."; ...
""; ...
"Regenerate the application with:"; ...
""; ...
"```sh"; ...
"/path/to/exasim-prefix/bin/text2code pdeapp.txt"; ...
"```"; ...
""; ...
"The `vdg.bin` file stores external variables. In backend data structures these are also called `odg`."; ...
""];

fid = fopen(char(dest + "/README.md"), 'w');
if fid == -1
    error("Unable to write %s.", dest + "/README.md");
end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', txt);
end
