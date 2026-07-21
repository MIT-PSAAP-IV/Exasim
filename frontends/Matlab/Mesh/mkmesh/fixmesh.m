function [p,t]=fixmesh(p,t)
%FIXMESH  Remove duplicated/unused nodes and fix element orientation.
%   [P,T]=FIXMESH(P,T)

% NOTE: This function works well for triangles, quadrangles and tetraedra.
% For hexaedra, this function can only recover elements whose bottom and up
% faces are numbered in a wrong (clockwise) way. TANGLED HEXES are not
% fixed with this routine !!!

disp('Fixing mesh...')

% Remove duplicated nodes:
snap=max(max(p,[],1)-min(p,[],1),[],2)*1024*eps;
[foo,ix,jx]=unique(round(p/snap)*snap,'rows');
if size(p,1) ~= length(ix); warning('Some vertices in mesh.p have been removed.'); end
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

if (size(t,2) == 3 && size(p,2) == 2) || (size(t,2) == 4 && size(p,2) == 3)          % Simplices
    v = simpvol(p,t);
    flip=v<0;
    t(flip,[1,2])=t(flip,[2,1]);
elseif (size(t,2) == 4 && size(p,2) == 2)      % Quads
    v = quadarea2d(p,t);
    flip = v < 0;
    t(flip,[1,2,3,4])=t(flip,[4,3,2,1]);
elseif (size(t,2) == 8 && size(p,2) == 3)      % Hex
    V12 = p(t(:,2),:) - p(t(:,1),:);
    V14 = p(t(:,4),:) - p(t(:,1),:);
    VAC = 1./4. * (p(t(:,5),:) + p(t(:,6),:) + p(t(:,7),:) + p(t(:,8),:)...
                 -(p(t(:,1),:) + p(t(:,2),:) + p(t(:,3),:) + p(t(:,4),:)));
    N1 = cross(V12,V14,2);
    flip = dot(N1,VAC,2)<0.;
    t(flip,[1,2,3,4])=t(flip,[4,3,2,1]);
    t(flip,[5,6,7,8])=t(flip,[8,7,6,5]);
    fprintf('%d Hexes have been reordered.\n',sum(flip))
else
    error('fixmesh not valid for this type of elements.');
end

if any(flip); warning('Some vertices in mesh.t have been reordered to meet code requirements.'); end
end

function v = elementmeasure(p,t)
%ELEMENTMEASURE Signed area/volume measure used to detect collapsed cells.
if (size(t,2) == 3 && size(p,2) == 2) || (size(t,2) == 4 && size(p,2) == 3)          % Simplices
    v = simpvol(p,t);
elseif (size(t,2) == 4 && size(p,2) == 2)      % Quads
    v = quadarea2d(p,t);
elseif (size(t,2) == 8 && size(p,2) == 3)      % Hex
    V12 = p(t(:,2),:) - p(t(:,1),:);
    V14 = p(t(:,4),:) - p(t(:,1),:);
    VAC = 1./4. * (p(t(:,5),:) + p(t(:,6),:) + p(t(:,7),:) + p(t(:,8),:)...
                 -(p(t(:,1),:) + p(t(:,2),:) + p(t(:,3),:) + p(t(:,4),:)));
    v = dot(cross(V12,V14,2),VAC,2);
else
    error('fixmesh not valid for this type of elements.');
end
end

function v = quadarea2d(p,t)
%QUADAREA2D Signed polygon area for 2-D quadrilateral elements.
x = reshape(p(t',1), size(t,2), [])';
y = reshape(p(t',2), size(t,2), [])';
v = 0.5*sum(x.*y(:,[2:end 1]) - y.*x(:,[2:end 1]), 2);
end
