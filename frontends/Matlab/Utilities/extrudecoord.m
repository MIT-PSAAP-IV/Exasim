function zdg = extrudecoord(zz, porder, np2d, nc, ne2d)

%EXTRUDECOORD Replicate high-order extrusion coordinates in extrudesol ordering.
%   zdg = extrudecoord(zz,porder) returns the high-order coordinates in each
%   extrusion interval with size [porder+1, length(zz)-1].
%
%   zdg = extrudecoord(zz,porder,np2d,nc,ne2d) expands those coordinates to
%   match extrudesol output layout: [np2d*(porder+1), nc, ne2d*(length(zz)-1)].

if nargin ~= 2 && nargin ~= 5
    error('extrudecoord requires either 2 or 5 input arguments.');
end

zz = zz(:).';
if numel(zz) < 2
    error('zz must contain at least two extrusion coordinates.');
end

nz = length(zz) - 1;

plc1d = masternodes(porder,1,1);
np1d = length(plc1d);
zdg = zeros(np1d,nz);
tz = [(1:nz); (2:nz+1)]';
for i = 1:nz
    pzdg = zz(tz(i,:));
    zdg(:,i) = (pzdg(2)-pzdg(1))*plc1d + pzdg(1);
end

if nargin == 2
    return;
end

if np2d < 1 || nc < 1 || ne2d < 1
    error('np2d, nc, and ne2d must be positive integers.');
end

zdg = reshape(zdg,[1 np1d 1 1 nz]);
zdg = repmat(zdg,[np2d 1 nc ne2d 1]);
zdg = reshape(zdg,[np2d*np1d,nc,ne2d*nz]);

end
