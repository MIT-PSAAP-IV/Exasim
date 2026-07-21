function [x,y] = cmeshparam6( nxw, nflr, nflf, nfuf, nfur, nr, sps, spr)
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

X1 = block([ nxw,nr],[1/sps(1),  1/sps(6), spr(1), spr(2)], [-2.0, 0.0; -1.0, 0.0; -2.0, 0.5; -1.0, 1.0]);
X2 = block([nflr,nr],[  sps(2),    sps(7), spr(2), spr(3)], [-1.0, 0.0; -0.5, 0.0; -1.0, 1.0; -0.5, 1.0]);
X3 = block([nflf,nr],[1/sps(3),  1/sps(8), spr(3), spr(4)], [-0.5, 0.0;  0.0, 0.0; -0.5, 1.0;  0.0, 1.0]);
X4 = block([nfuf,nr],[  sps(4),    sps(9), spr(4), spr(5)], [ 0.0, 0.0;  0.5, 0.0;  0.0, 1.0;  0.5, 1.0]);
X5 = block([nfur,nr],[1/sps(5), 1/sps(10), spr(5), spr(6)], [ 0.5, 0.0;  1.0, 0.0;  0.5, 1.0;  1.0, 1.0]);
X6 = block([ nxw,nr],[  sps(1),   sps(11), spr(6), spr(7)], [ 1.0, 0.0;  2.0, 0.0;  1.0, 1.0;  2.0, 0.5]);

% figure(3); clf; hold on;
% x = X1(1,:,:); y = X1(2,:,:); plot(x(:),y(:),'o');
% x = X2(1,:,:); y = X2(2,:,:); plot(x(:),y(:),'o');
% x = X3(1,:,:); y = X3(2,:,:); plot(x(:),y(:),'o');
% x = X4(1,:,:); y = X4(2,:,:); plot(x(:),y(:),'o');
% x = X5(1,:,:); y = X5(2,:,:); plot(x(:),y(:),'o');
% x = X6(1,:,:); y = X6(2,:,:); plot(x(:),y(:),'o');
%
%
% Xa = block([ nxw,nr],[1/sps(1),  1/sps(6), spr(1), spr(2)], [-2.0, 0.0; -1.0, 0.0; -2.0, 1.0; -1.0, 1.0]);
% Xb = block([ nxw,nr],[  sps(1),   sps(11), spr(6), spr(7)], [ 1.0, 0.0;  2.0, 0.0;  1.0, 1.0;  2.0, 1.0]);
%
% x1 = squeeze(Xa(1,:,:)); y1 = squeeze(Xa(2,:,:));
% x2 = squeeze(X2(1,:,:)); y2 = squeeze(X2(2,:,:));
% x3 = squeeze(X3(1,:,:)); y3 = squeeze(X3(2,:,:));
% x4 = squeeze(X4(1,:,:)); y4 = squeeze(X4(2,:,:));
% x5 = squeeze(X5(1,:,:)); y5 = squeeze(X5(2,:,:));
% x6 = squeeze(Xb(1,:,:)); y6 = squeeze(Xb(2,:,:));
%
% [p1,t1] = refinecartgrid(x1, y1, [0.02]);
% p0 = p1; p0(:,1) = p0(:,1)+2;
% p1 = mapp(p0,[-2.0, 0.0; -1.0, 0.0; -2.0, 0.5; -1.0, 1.0]);
%
% [p2,t2] = refinecartgrid(x2, y2, [0.02 0.1]);
% [p3,t3] = refinecartgrid(x3, y3, [0.02 0.1]);
% [p4,t4] = refinecartgrid(x4, y4, [0.02 0.1]);
% [p5,t5] = refinecartgrid(x5, y5, [0.02 0.1]);
%
% [p6,t6] = refinecartgrid(x6, y6, [0.02]);
% p0 = p6; p0(:,1) = p0(:,1)-1;
% p6 = mapp(p0,[ 1.0, 0.0;  2.0, 0.0;  1.0, 1.0;  2.0, 0.5]);
%
% figure(4); clf; hold on;
% simpplot(p1,t1);
% simpplot(p2,t2);
% simpplot(p3,t3);
% simpplot(p4,t4);
% simpplot(p5,t5);
% simpplot(p6,t6);
% axis on; axis equal; axis tight;

% X1 = block([ nxw,nr],[1/sps(1),  1/sps(6), spr(1), spr(2)], [-2.5, 0.0; -1.0, 0.0; -2.5, 0.7; -1.0, 1.4]);
% X2 = block([nflr,nr],[  sps(2),    sps(7), spr(2), spr(3)], [-1.0, 0.0; -0.5, 0.0; -1.0, 1.4; -0.5, 1.4]);
% X3 = block([nflf,nr],[1/sps(3),  1/sps(8), spr(3), spr(4)], [-0.5, 0.0;  0.0, 0.0; -0.5, 1.4;  0.0, 1.4]);
% X4 = block([nfuf,nr],[  sps(4),    sps(9), spr(4), spr(5)], [ 0.0, 0.0;  0.5, 0.0;  0.0, 1.4;  0.5, 1.4]);
% X5 = block([nfur,nr],[1/sps(5), 1/sps(10), spr(5), spr(6)], [ 0.5, 0.0;  1.0, 0.0;  0.5, 1.4;  1.0, 1.4]);
% X6 = block([ nxw,nr],[  sps(1),   sps(11), spr(6), spr(7)], [ 1.0, 0.0;  2.5, 0.0;  1.0, 1.4;  2.5, 0.7]);

x = [squeeze(X1(1,:,:))', squeeze(X2(1,2:end,:))', squeeze(X3(1,2:end,:))', ...
                          squeeze(X4(1,2:end,:))', squeeze(X5(1,2:end,:))', squeeze(X6(1,2:end,:))'];
y = [squeeze(X1(2,:,:))', squeeze(X2(2,2:end,:))', squeeze(X3(2,2:end,:))', ...
                          squeeze(X4(2,2:end,:))', squeeze(X5(2,2:end,:))', squeeze(X6(2,2:end,:))'];

%figure(4); clf; plot(x(:),y(:),'o');

% X1 = block([nxw,nr],[1/sps(1), 1/sps(4), spr(1), spr(2)], [-2,0;-1,0;-2,1;-1,1]);
% X2 = block([nflr,nr],[  sps(2),   sps(5), spr(2), spr(3)], [-1,0; 0,0;-1,1; 0,1]);
% X3 = block([nfur,nr],[1/sps(3), 1/sps(6), spr(3), spr(4)], [ 0,0; 1,0; 0,1; 1,1]);
% X4 = block([nxw,nr],[  sps(1),   sps(7), spr(4), spr(5)], [ 1,0; 2,0; 1,1; 2,1]);
%
% x = [squeeze(X1(1,:,:))', squeeze(X2(1,2:end,:))', squeeze(X3(1,2:end,:))', squeeze(X4(1,2:end,:))'];
% y = [squeeze(X1(2,:,:))', squeeze(X2(2,2:end,:))', squeeze(X3(2,2:end,:))', squeeze(X4(2,2:end,:))'];
