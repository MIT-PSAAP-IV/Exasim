function [Cx,Cy] = surfaceForceCoefficients(Cp,Cf,x,y,master)
%SURFACEFORCECOEFFICIENTS Integrate pressure and skin-friction forces.
%
%   [Cx,Cy] = surfaceForceCoefficients(Cp,Cf,x,y,master)
%
% Cp, Cf, x, and y contain face nodal data with logical size npf x nf.
% They may be provided as npf x nf or npf x 1 x nf arrays. The face shape
% functions and quadrature weights are taken from master.shapfgt,
% master.shapft, or master.shapfc and master.gwf or master.gwfc.
%
% The chord is assumed to be one. Cp is assumed to follow the Exasim
% convention Cp = -2*(p - pinf), so the pressure contribution is Cp*n.
% Cf is assumed to be signed in the local face tangent direction.

Cp = faceMatrix(Cp,'Cp');
Cf = faceMatrix(Cf,'Cf');
x = faceMatrix(x,'x');
y = faceMatrix(y,'y');

if ~isequal(size(Cp),size(Cf),size(x),size(y))
    error('Cp, Cf, x, and y must have the same logical size npf x nf.');
end

[npf,nf] = size(Cp);
[shapft,dshapft,gwf] = faceOperators(master,npf);

% Use the ordered surface loop orientation to choose the outward normal.
xf = mean(x,1).';
yf = mean(y,1).';
signedArea = 0.5*sum(xf.*yf([2:end 1]) - xf([2:end 1]).*yf);

Cx = 0.0;
Cy = 0.0;
for f = 1:nf
    cpg = shapft*Cp(:,f);
    cfg = shapft*Cf(:,f);

    dxds = dshapft*x(:,f);
    dyds = dshapft*y(:,f);
    jac = hypot(dxds,dyds);
    if any(jac <= eps(max(jac)))
        error('Degenerate face detected while computing force coefficients.');
    end

    sx = dxds./jac;
    sy = dyds./jac;
    if signedArea < 0
        nx = -sy;
        ny = sx;
    else
        nx = sy;
        ny = -sx;
    end

    wjac = gwf(:).*jac;
    Cx = Cx + sum((cpg.*nx + cfg.*sx).*wjac);
    Cy = Cy + sum((cpg.*ny + cfg.*sy).*wjac);
end

end

function A = faceMatrix(A,name)
if ismatrix(A)
    return;
end

if ndims(A) == 3 && size(A,2) == 1
    A = squeeze(A(:,1,:));
elseif ndims(A) == 3 && size(A,3) == 1
    A = A(:,:,1);
else
    error('%s must have logical size npf x nf.', name);
end

if isvector(A)
    A = A(:);
end
end

function [shapft,dshapft,gwf] = faceOperators(master,npf)
if ~isstruct(master)
    error('master must be a structure.');
end

if isfield(master,'shapfgt')
    shap = master.shapfgt;
elseif isfield(master,'shapft')
    shap = master.shapft;
elseif isfield(master,'shapfc')
    shap = master.shapfc;
else
    error('master.shapfgt, master.shapft, or master.shapfc is required.');
end

if size(shap,2) == npf
    shapft = squeeze(shap(:,:,1));
    dshapft = squeeze(shap(:,:,2));
else
    shapft = squeeze(shap(:,:,1))';
    dshapft = squeeze(shap(:,:,2))';
end

if isfield(master,'gwf')
    gwf = master.gwf;
elseif isfield(master,'gwfc')
    gwf = master.gwfc;
else
    error('master.gwf or master.gwfc is required.');
end

if size(shapft,2) ~= npf
    error('Face shape functions are incompatible with npf = %d.', npf);
end
if numel(gwf) ~= size(shapft,1)
    error('Face quadrature weights are incompatible with face shape functions.');
end
end
