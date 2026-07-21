function [p,t]=fixmesh(p,t)
%FIXMESH  Remove duplicated/unused nodes and fix element orientation.
%   [P,T]=FIXMESH(P,T)

% NOTE: This function works for triangles, quadrangles and tetrahedra.

% Remove duplicated nodes:
snap=max(max(p,[],1)-min(p,[],1),[],2)*1024*eps;
[foo,ix,jx]=unique(round(p/snap)*snap,'rows');
p=p(ix,:);
t=jx(t);
if size(t,2) == 1, t = t'; end  % This lines ensures the function works for one element

% Remove elements collapsed by duplicate-node snapping or degenerate input.
v = elementmeasure(p,t);
vtol = max(max(abs(v)),1)*1024*eps;
keep = abs(v) > vtol;
if any(~keep)
    warning('%d zero-volume/area elements in mesh.t have been removed.', nnz(~keep));
    t = t(keep,:);
end
if isempty(t)
    error('fixmesh removed all elements because they have zero volume/area.');
end

% Remove nodes that are not contained in t:
[pix,ix,jx]=unique(t);
t=reshape(jx,size(t));
p=p(pix,:);

nv = size(t,2); % # vertices
nd = size(p,2); % # dimensions

if ((nd==2) && (nv==3)) || ((nd==3) && (nv==4))
    v = simpvol(p,t);
    flip=v<0;
    t(flip,[1,2])=t(flip,[2,1]);
elseif nd == 2 && nv == 4
    v = quadarea2d(p,t);
    flip = v < 0;
    t(flip,[1,2,3,4]) = t(flip,[4,3,2,1]);
end

function v = elementmeasure(p,t)
nv = size(t,2);
nd = size(p,2);
if ((nd==2) && (nv==3)) || ((nd==3) && (nv==4))
    v = simpvol(p,t);
elseif nd == 2 && nv == 4
    v = quadarea2d(p,t);
else
    error('fixmesh not valid for this type of elements.');
end

function v = quadarea2d(p,t)
x = reshape(p(t',1), size(t,2), [])';
y = reshape(p(t',2), size(t,2), [])';
v = 0.5*sum(x.*y(:,[2:end 1]) - y.*x(:,[2:end 1]), 2);
