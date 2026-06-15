% exasim_setup — put the Exasim MATLAB frontend on the path.
%
% Usage:
%   run('<prefix>/share/exasim/matlab/exasim_setup.m')      % installed
%   run('<repo>/frontends/Matlab/exasim_setup.m')           % source tree
%
% Runtime data (datain/, dataout/) is written under the working directory
% (override with pde.datapath); generated code and the solver build live in
% the hidden pde.builddir (default <cwd>/.exasim).

srcdir = fileparts(mfilename('fullpath'));
addpath(fullfile(srcdir, 'Gencode'));
addpath(fullfile(srcdir, 'master'));
addpath(fullfile(srcdir, 'Mesh'));
addpath(fullfile(srcdir, 'Mesh', 'boundaryexpressions'));
addpath(fullfile(srcdir, 'Mesh', 'mkmesh'));
addpath(fullfile(srcdir, 'Mesh', 'cmesh'));
addpath(fullfile(srcdir, 'Mesh', 'lesmesh'));
addpath(fullfile(srcdir, 'Mesh', 'surfmesh'));
addpath(fullfile(srcdir, 'HDG'));
addpath(fullfile(srcdir, 'Preprocessing'));
addpath(fullfile(srcdir, 'Postprocessing'));
addpath(fullfile(srcdir, 'Utilities'));
addpath(fullfile(srcdir, 'STG'));

% Make external tools in common nonstandard prefixes visible without
% clobbering the user's PATH.
setenv('PATH', [getenv('PATH') ':/usr/bin:/usr/local/bin:/opt/local/bin:/opt/homebrew/bin']);
setenv('EXASIM_PREFIX', '/Users/cuongnguyen/Documents/GitHub/PSAAP/Exasim/local');

fprintf("==> Exasim MATLAB frontend ...\n");
clear srcdir;
