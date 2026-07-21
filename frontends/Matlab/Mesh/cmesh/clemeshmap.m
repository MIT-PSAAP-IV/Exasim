function [p,t,xdg] = clemeshmap( xf, yf, p, t, lw, ll, porder)

if nargin < 7
  porder = 1;
end

small = 1.e-8;
shift = 0.004;                   % 0.004 Good for SD7003

%nx = 41;
%ny = 21;

%[xr,yr] = read_foil();

dxu = xf(2)-xf(1);
dyu = yf(2)-yf(1);
dxl = xf(end-1)-xf(end);
dyl = yf(end-1)-yf(end);
tau = acos((dxu*dxl+dyu*dyl)/sqrt(dxu*dxu+dyu*dyu)/sqrt(dxl*dxl+dyl*dyl));
n   = 2-tau/pi;

chord = max(xf)-min(xf);
xmo = 0.5*(max(xf)+min(xf));
ymo = yf(1);
xn = 2.0*(1.0+shift)*(xf-xmo)/chord-shift;
yn = 2.0*(1.0+shift)*(yf-ymo)/chord;

% Near circle
[x,y] = trefftz_inv( xn, yn, n, 1/n, 0,1);

% Center
xm = 0.5*(max(x)+min(x));
ym = 0.5*(max(y)+min(y));
fc = (max(x)-min(x)+max(y)-min(y))/4;
x = (x-xm)/fc;
y = (y-ym)/fc;

% Now a circle
[A,B] = GetTG( 200, x, y);

% figure(1); clf; hold on;
% for i = 1:6
%   simpplot(p{i},t{i});
% end
% axis on; axis equal; axis tight;

% Start mesh generation
elemtype = 1; nodetype = 1;
bndexpr={'true'};
xdg = cell(6,1);
npe = (porder+1)^2;
for i = 1:6
  mesh = mkmesh(p{i},t{i},porder,bndexpr,elemtype,nodetype);
  xdg{i} = reshape(permute(mesh.dgnodes,[1 3 2]), [npe*size(mesh.dgnodes,3) 2]);
end

p = maptrefftzmesh(p, lw, ll, small, A, B, fc, xm, ym, n, shift, chord, xmo, ymo);
xdg = maptrefftzmesh(xdg, lw, ll, small, A, B, fc, xm, ym, n, shift, chord, xmo, ymo);

figure(1); clf; hold on;
for i = 1:6
  simpplot(p{i},t{i});
end
axis on; axis equal; axis tight;

figure(2); clf; hold on;
for i = 1:6
  simpplot(p{i},t{i});
  plot(xdg{i}(:,1), xdg{i}(:,2), '.b');
end
axis on; axis equal; axis tight;

for i = 1:6
  ne = numel(xdg{i})/(npe*2);
  xdg{i} = permute(reshape(xdg{i}, [npe ne 2]), [1 3 2]);
end


% xdg{6}(:,1) = (xdg{6}(:,1)-1)*(sqrt(lw+1)-1)+1;
% xdg{1}(:,1) = (xdg{1}(:,1)+1)*(sqrt(lw+1)-1)-1;
%
% p{6}(:,1) = (p{6}(:,1)-1)*(sqrt(lw+1)-1)+1;
% p{1}(:,1) = (p{1}(:,1)+1)*(sqrt(lw+1)-1)-1;
% for i = 1:6
%   p{i}(:,2) = p{i}(:,2)*sqrt(ll)+small;
%   ix = (p{i}(:,1) < 0);
%   p{i}(ix,1) = -p{i}(ix,1);
%   p{i}(ix,2) = -p{i}(ix,2);
%
%   zr = complex(p{i}(:,1),p{i}(:,2));
%   wr = zr.^2;
%
%   % Now a circle
%   [xc,yc] = trefftz_inv( 2.0*(real(wr)-0.5), 2.0*imag(wr), 2.0, 0.5, 0, 0);
%
%   % Now a near circle
%   [xg, yg] = TG( 2.0*xc, 2.0*yc, A, B);
%
%   % Finally the real thing
%   [xgf, ygf] = trefftz( xg*fc+xm, yg*fc+ym, n, 1/n, 0);
%
%   xgf = (xgf+shift)*chord/(2.0*(1.0+shift)) + xmo;
%   ygf = ygf*chord/(2.0*(1.0+shift)) + ymo;
%
%   p{i}(:,1) = xgf;
%   p{i}(:,2) = ygf;
% end

% ix = xr(1,:)>1;
% xr(:,ix) = (xr(:,ix)-1)*(sqrt(lw+1)-1)+1;
%
% ix = xr(1,:)<-1;
% xr(:,ix) = (xr(:,ix)+1)*(sqrt(lw+1)-1)-1;
%
% yr = yr*sqrt(ll)+small;
% ix = (xr < 0);
% xr(ix) = -xr(ix);
% yr(ix) = -yr(ix);
%
%
% % x^2 Transformation
% zr = complex(xr,yr);
% wr = zr.^2;
%
% % Now a circle
% [xc,yc] = trefftz_inv( 2.0*(real(wr)-0.5), 2.0*imag(wr), 2.0, 0.5, 0, 0);
%
% % Now a near circle
% [xg, yg] = TG( 2.0*xc, 2.0*yc, A, B);
%
% % Finally the real thing
% [xgf, ygf] = trefftz( xg*fc+xm, yg*fc+ym, n, 1/n, 0);
%
% xgf = (xgf+shift)*chord/(2.0*(1.0+shift)) + xmo;
% ygf = ygf*chord/(2.0*(1.0+shift)) + ymo;
%
% figure(3);clf;
% surf(xgf, ygf, 0*xgf);
% view(2), axis equal;
%

function p = maptrefftzmesh(p, lw, ll, small, A, B, fc, xm, ym, n, shift, chord, xmo, ymo)
%MAPTREFFTZMESH Map computational mesh blocks to a Trefftz airfoil.
%
%   p = MAPTREFFTZMESH(p, lw, ll, small, A, B, fc, xm, ym, n, ...
%                      shift, chord, xmo, ymo)
%
% Inputs
%   p      - Cell array of mesh blocks. Each p{i} is an N-by-2 array.
%   lw     - Streamwise stretching parameter.
%   ll     - Wall-normal stretching parameter.
%   small  - Wall offset.
%   A, B   - TG transformation parameters.
%   fc     - Scaling factor for the near-circle transformation.
%   xm, ym - Translation before the final Trefftz transform.
%   n      - Trefftz exponent.
%   shift  - Chordwise shift.
%   chord  - Airfoil chord length.
%   xmo    - x-coordinate offset.
%   ymo    - y-coordinate offset.
%
% Output
%   p      - Transformed mesh blocks.

% Stretch the left and right blocks
p{6}(:,1) = (p{6}(:,1) - 1) * (sqrt(lw + 1) - 1) + 1;
p{1}(:,1) = (p{1}(:,1) + 1) * (sqrt(lw + 1) - 1) - 1;

% Map each block
for i = 1:numel(p)

    % Wall-normal stretching
    p{i}(:,2) = p{i}(:,2) * sqrt(ll) + small;

    % Reflect lower half-plane
    ix = p{i}(:,1) < 0;
    p{i}(ix,1) = -p{i}(ix,1);
    p{i}(ix,2) = -p{i}(ix,2);

    % Square mapping
    zr = complex(p{i}(:,1), p{i}(:,2));
    wr = zr.^2;

    % Map to a circle
    [xc, yc] = trefftz_inv( ...
        2*(real(wr) - 0.5), ...
        2*imag(wr), ...
        2.0, 0.5, 0, 0);

    % Map to a near circle
    [xg, yg] = TG(2*xc, 2*yc, A, B);

    % Final Trefftz mapping
    [xgf, ygf] = trefftz( ...
        xg * fc + xm, ...
        yg * fc + ym, ...
        n, 1/n, 0);

    % Scale and translate to the physical airfoil
    xgf = (xgf + shift) * chord / (2 * (1 + shift)) + xmo;
    ygf = ygf * chord / (2 * (1 + shift)) + ymo;

    % Store mapped coordinates
    p{i}(:,1) = xgf;
    p{i}(:,2) = ygf;
end

function [x,y] = trefftz( x1, y1, n, cx, cy)
z1 = complex( x1, y1);
cc = complex( cx, cy);
A = ((z1-cc)./(z1+cc)).^n;
z = ((1+A)./(1-A))*n*cc;
x = real(z);
y = imag(z);

function [x,y] = trefftz_inv( x1, y1, n, cx, cy, track)
z1 = complex( x1, y1);
cc = complex( cx, cy);
A = ((z1-n*cc)./(z1+n*cc));
if track
   R = abs(A);
   T = angle(A);
   for k = 2:size(T,1)
       d(1) = T(k) + 2*pi;
       d(2) = T(k);
       d(3) = T(k) - 2*pi;
       [minv,j] = min(abs(d-T(k-1)));
       T(k) = d(j);
   end
   B = ((R).^(1/n)).*exp(i*T/n);
else
   B = A.^(1/n);
end
z = ((1+B)./(1-B))*cc;
x = real(z);
y = imag(z);

function [x,y] = TG( xc, yc, A, B)
N = size(A,2)-1;
zc = complex( xc, yc);
e = complex(zeros(size(zc)),zeros(size(zc)));
for j=1:N+1
    e = e + (A(j)+i*B(j)).*zc.^(1-j);
end
ix = isnan(e);
e(ix) = 0;
z = zc.*exp(e);
x = real(z);
y = imag(z);


function [A,B] = GetTG( N, x, y)
lr = log(sqrt(x.^2 + y.^2));
th = atan2(y,x);
for k=2:size(th,1)
    if (th(k) < th(k-1))
        th(k) = th(k) + 2*pi;
    end
end
thi = [th(1:end-1) - 2*pi; th(1:end); th(2:end)+2*pi];
lri = lr([1:end-1,1:end,2:end]);
A = ones(1,N+1);
B = zeros(1,N+1);
Y = complex(zeros(1,2*N), zeros(1,2*N));
tt = 0:pi/N:2*pi;
tt = tt(1:end-1);
Anew = 0*A;
Bnew = B;
while norm(A-Anew) + norm(B-Bnew) > 1.e-15,
    %disp(norm(A-Anew) + norm(B-Bnew));
    A = Anew;
    B = Bnew;
    B(1) = th(1) - sum(B(2:N+1));
    B(N+1) = 0;
    Y(1) = 2*N*B(1);
    Y(2:N) = N*(B(2:N)+ i*A(2:N));
    Y(N+1) = 2*N*B(N+1);
    Y(N+2:2*N) = conj(Y(N:-1:2));
    zt = tt + ifft(Y);
    rr = spline(thi, lri, zt);
    Y = fft(rr);
    Anew(1) = real(Y(1))/(2*N);
    Anew(2:N) = real(Y(2:N))/N;
    Anew(N+1) = real(Y(N+1))/(2*N);
    Bnew(1) = B(1);
    Bnew(2:N) = -imag(Y(2:N))/N;
    Bnew(N+1) = 0;
end
