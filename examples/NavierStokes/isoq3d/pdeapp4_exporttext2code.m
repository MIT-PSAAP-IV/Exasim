function pdeapp4_exporttext2code(res, nz, outdir, meshname)
%PDEAPP4_EXPORTTEXT2CODE  Build + export a v4 "true-geometry" isoq3d fluid mesh.
%
%   pdeapp4_exporttext2code(res, nz, outdir, meshname)
%
%   Reconstruction of the missing v4 export driver that produced the committed
%   mesh_fluid/*_truegeom packages.  Mirrors pdeapp3_exporttext2code.m through
%   mesh/state construction, but:
%     * builds the wall-on-profile + axis-butterfly-plug mesh via mkmesh_isoq3d4
%       (5 boundaries [4,4,2,3,1]; NO 0.002 m standoff, NO inner cylinder);
%     * emits a UNIFORM FREESTREAM initial condition (the original v3 rotated a
%       2D solution from solp2.mat, which is absent -- a uniform start is a
%       stable seed for a scaling-study mesh; the fluid may need a few extra
%       pseudo-time steps to settle);
%     * writes grid/xdg/udg/vdg.bin directly with writebin (the committed
%       format) instead of calling exporttext2code, whose MATLAB string-array
%       machinery does not run under Octave.
%
%   res      = [n1 m1 n2 m2] shell subdivisions (default [48 80 36 80] = medium)
%   nz       = azimuthal element layers over [0,pi/2], even (default 4)
%   outdir   = directory to write <meshname>/{grid,xdg,udg,vdg}.bin + header
%   meshname = subdir + header tag (default 'medium_truegeom')
%
%   Element-count model (validated vs committed coarse/medium/fine):
%     ne = (n1*m1 + n2*m2)*nz  +  3*(nz/2)^2 * m1

if nargin < 1 || isempty(res);      res = [48 80 36 80]; end
if nargin < 2 || isempty(nz);       nz  = 4;             end
if nargin < 3 || isempty(outdir);   outdir = pwd();      end
if nargin < 4 || isempty(meshname); meshname = 'medium_truegeom'; end

porder = 2;

% --- physics parameters (from the committed truegeom header) ---
% [gam Re Pr Minf  rinf ruinf rvinf rwinf rEinf  Tinf  Tref Twall]
physicsparam = [1.4, 156000, 0.71, 7.6, 1, 1, 0, 0, 0.53092, 0.030916, 266.5, 300];
Uinf = physicsparam(5:9);   % uniform conserved freestream [rho rhou rhov rhow rhoE]

% --- build the v4 mesh ---
mesh = mkmesh_isoq3d4(porder, res, nz);
ne = size(mesh.t,2); np = size(mesh.p,2); npe = size(mesh.dgnodes,1);

% --- fields ---
udg = zeros(npe,5,ne); for k=1:5; udg(:,k,:) = Uinf(k); end   % uniform freestream IC
vdg = reshape(mesh.dist, [npe,1,ne]);                          % wall-distance (odg)

% --- write the Text2Code binary package (committed writebin format) ---
od = fullfile(outdir, meshname);
if exist(od,'dir')==0; mkdir(od); end
p = mesh.p; t = mesh.t; dg = mesh.dgnodes;
writebin(fullfile(od,'grid.bin'), [size(p) size(t) p(:)' t(:)']);
writebin(fullfile(od,'xdg.bin'),  [size(dg) dg(:)']);
writebin(fullfile(od,'udg.bin'),  [size(udg) udg(:)']);
writebin(fullfile(od,'vdg.bin'),  [size(vdg) vdg(:)']);

fprintf('pdeapp4: wrote %s  ne=%d np=%d npe=%d nz=%d res=%s\n', od, ne, np, npe, nz, mat2str(res));
end
