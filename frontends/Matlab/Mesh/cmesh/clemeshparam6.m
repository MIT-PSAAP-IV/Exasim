function [p,t] = clemeshparam6( nxw, nflr, nflf, nfuf, nfur, nr, sps, spr, yref)
%CMESHPARAM  Creates mesh in parametric space for airfoil c-type grids
%        __________________________________ ______
%       |      |      |      |      |      |      |
%       |      |      |      |      |      |      |
%    nr |      |      |      |      |      |      |
%       |      |      |      |      |      |      |
%       |______|______|______|______|______|______|
%          nxw   nflr   nflf   nfuf   nfur
%
%   nxw  : number of subdivison in the wake
%   nflr : number of subdivision in the lower foil (rear)
%   nflf : number of subdivision in the lower foil (front)
%   nfuf : number of subdivisions in the upper foil (front)
%   nfur : number of subdivisions in the upper foil (rear)
%   nr   : number of subdivisions in the radial direction
%   sps(id) : streamwise size control
%     sps(1)  - ratio between the first and last elements in the wake
%     sps(2)  - ratio between mid-chord and trailing edge element (lower)
%     sps(3)  - ratio between mid-chord and leading edge element (lower)
%     sps(4)  - ratio between mid-chord and leading edge element (upper)
%     sps(5)  - ratio between mid-chord and trailing edge element (upper)
%     sps(6)  - ratio between the first and last elements in the wake (lower far field)
%     sps(7)  - ratio between mid-chord and trailing edge element (lower far field)
%     sps(8)  - ratio between mid-chord and leading edge element (lower far field)
%     sps(9)  - ratio between mid-chord and leading edge element (upper far field)
%     sps(10) - ratio between mid-chord and trailing edge element (upper far field)
%     sps(11) - ratio between the first and last elements in the wake (upper far field)
%   sps(id) : radial size control
%     spr(1) - ratio between far-field and wake element (lower)
%     spr(2) - ratio between far-field and trailing edge (lower)
%     spr(3) - ratio between far-field and mid-chord (lower)
%     spr(4) - ratio between far-field and leading edge
%     spr(5) - ratio between far-field and mid-chord (upper)
%     spr(6) - ratio between far-field and trailing edge (upper)
%     spr(7) - ratio between far-field and wake element (upper)

% TEC = 15;
% sps = [TEC, 1, 1, 1, 1, TEC, 1, 1, 1, 1, TEC];
% spr = [10, 10, 10, 10, 10, 10, 10]*25;
% [x,y] = cmeshparam6(nxw, nflr, nflf, nfuf, nfur, nr, ...
%                     [TEC, 1, 1, 1, 1, TEC, 1, 1, 1, 1, TEC], ...
%                     [10, 10, 10, 10, 10, 10, 10]*25);

X = cell(6,1); p = cell(6,1); t = cell(6,1);

X{1} = block([ nxw,nr],[1/sps(1),  1/sps(6), spr(1), spr(2)], [-2.0, 0.0; -1.0, 0.0; -2.0, 1.0; -1.0, 1.0]);
X{2} = block([nflr,nr],[  sps(2),    sps(7), spr(2), spr(3)], [-1.0, 0.0; -0.5, 0.0; -1.0, 1.0; -0.5, 1.0]);
X{3} = block([nflf,nr],[1/sps(3),  1/sps(8), spr(3), spr(4)], [-0.5, 0.0;  0.0, 0.0; -0.5, 1.0;  0.0, 1.0]);
X{4} = block([nfuf,nr],[  sps(4),    sps(9), spr(4), spr(5)], [ 0.0, 0.0;  0.5, 0.0;  0.0, 1.0;  0.5, 1.0]);
X{5} = block([nfur,nr],[1/sps(5), 1/sps(10), spr(5), spr(6)], [ 0.5, 0.0;  1.0, 0.0;  0.5, 1.0;  1.0, 1.0]);
X{6} = block([ nxw,nr],[  sps(1),   sps(11), spr(6), spr(7)], [ 1.0, 0.0;  2.0, 0.0;  1.0, 1.0;  2.0, 1.0]);

for i = 1:6
  x = squeeze(X{i}(1,:,:)); y = squeeze(X{i}(2,:,:));
  [p{i}, t{i}] = refinecartgrid(x, y, yref);
end
p0 = p{1}; p0(:,1) = p0(:,1)+2;
p{1} = mapp(p0,[-2.0, 0.0; -1.0, 0.0; -2.0, 0.5; -1.0, 1.0]);
p0 = p{6}; p0(:,1) = p0(:,1)-1;
p{6} = mapp(p0,[ 1.0, 0.0;  2.0, 0.0;  1.0, 1.0;  2.0, 0.5]);

% figure(1); clf; hold on;
% for i = 1:6
%   simpplot(p{i},t{i});
% end
% axis on; axis equal; axis tight;
