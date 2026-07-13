function [UDGspm,UDG2d] = spanwiseaverage(UDGavg,n1,nz,UDG,nodeIndex,spanIndex)
%SPANWISEAVERAGE Compute the spanwise average of an Exasim solution array.
%
%   UDGspm = spanwiseaverage(UDGavg)
%   UDGspm = spanwiseaverage(UDGavg,n1,nz)
%   [UDGspm,UDG2d] = spanwiseaverage(UDGavg,n1,nz,UDG,nodeIndex,spanIndex)
%
% UDGavg is expected to have size [npe,nc,ne], where npe = n1*n2 and
% ne = ne2*nz. The function reshapes the solution as
% [n1,n2,nc,ne2,nz] and averages over the spanwise element direction nz
% and the second nodal direction n2.
%
% If UDG is provided, the function also extracts a two-dimensional slice
% after reshaping UDG to [n1,n2,nc,ne2,nz]:
%   UDG2d = squeeze(UDG(:,nodeIndex,:,:,spanIndex))
% The defaults are nodeIndex = 3 and spanIndex = nz.
%
% Example:
%   n1 = 16;
%   nz = 12;
%   gamma = 1.4;
%
%   UDG = getsolution("eppler3d/dataout/outudg_t2000",dmd,64);
%   UDGavg = getmeansolution("eppler3d/dataout/outsolavg",dmd,64);
%   [UDGspm,UDG2d] = spanwiseaverage(UDGavg,n1,nz,UDG,3,10);
%
%   mesh2d.xpe = mesh2d.plocal;
%   mesh2d.telem = mesh2d.tlocal;
%   figure(1); clf; scaplot(mesh2d, UDGspm(:,2,:)./UDGspm(:,1,:),[],2);
%   colormap('jet'); colorbar;
%   figure(1); clf; scaplot(mesh2d, eulereval3d(UDGspm,'p',gamma,Ma),[],2);
%   colormap('jet'); colorbar;
%   figure(1); clf; scaplot(mesh2d, eulereval3d(UDGspm,'vm',gamma,Ma),[],2);
%   colormap('jet'); colorbar;
%   figure(1); clf; scaplot(mesh2d, eulereval3d(UDG2d,'u',gamma,Ma),[],2,1);
%   colormap('jet'); colorbar;

if nargin < 2 || isempty(n1)
    n1 = 16;
end
if nargin < 3 || isempty(nz)
    nz = 12;
end

if ndims(UDGavg) ~= 3
    error('UDGavg must have size [npe,nc,ne].');
end

npe = size(UDGavg,1);
nc = size(UDGavg,2);
ne = size(UDGavg,3);

if mod(npe,n1) ~= 0
    error('size(UDGavg,1) = %d is not divisible by n1 = %d.', npe, n1);
end
if mod(ne,nz) ~= 0
    error('size(UDGavg,3) = %d is not divisible by nz = %d.', ne, nz);
end

n2 = npe/n1;
ne2 = ne/nz;

UDGavg = reshape(UDGavg, [n1 n2 nc ne2 nz]);
UDGspm = squeeze(mean(mean(UDGavg,5),2));

UDG2d = [];
if nargin >= 4 && ~isempty(UDG)
    if nargin < 5 || isempty(nodeIndex)
        nodeIndex = 3;
    end
    if nargin < 6 || isempty(spanIndex)
        spanIndex = nz;
    end
    if ndims(UDG) ~= 3
        error('UDG must have size [npe,nc,ne].');
    end
    if any(size(UDG) ~= [npe nc ne])
        error('UDG must have the same size as UDGavg.');
    end
    if nodeIndex < 1 || nodeIndex > n2
        error('nodeIndex = %d is outside the valid range [1,%d].', nodeIndex, n2);
    end
    if spanIndex < 1 || spanIndex > nz
        error('spanIndex = %d is outside the valid range [1,%d].', spanIndex, nz);
    end
    UDG = reshape(UDG, [n1 n2 nc ne2 nz]);
    UDG2d = squeeze(UDG(:,nodeIndex,:,:,spanIndex));
end

end
