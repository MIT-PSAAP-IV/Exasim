function [vn, vt] = normaltangentvelocity(v, nl)
%NORMALTANGENTVELOCITY Decompose velocity into normal and tangential parts.
%
%   [vn, vt] = normaltangentvelocity(v, nl)
%
% v has size np x nd, np x nd x m, np x nd x ne, or np x nd x ne x m.
% nl has size np x nd or np x nd x ne. For nd = 2, vn = v dot n and
% vt = v dot t are scalar fields with t = [-n_y, n_x]. For nd = 3, vn is
% the vector normal component (v dot nl) nl, and vt is the vector tangential
% component v - vn.

validateInputs(v, nl);

nl = expandNormals(v, nl);
dotvn = sum(v.*nl, 2);

if size(v, 2) == 2
    tl = cat(2, -nl(:,2,:,:), nl(:,1,:,:));
    vn = dotvn;
    vt = sum(v.*tl, 2);
    return;
end

vn = dotvn.*nl;
vt = v - vn;

end

function validateInputs(v, nl)
if ~isnumeric(v) || ~isnumeric(nl)
    error('v and nl must be numeric arrays.');
end
if ndims(v) < 2 || ndims(v) > 4
    error('v must have size np x nd, np x nd x m, np x nd x ne, or np x nd x ne x m.');
end
if ndims(nl) ~= 2 && ndims(nl) ~= 3
    error('nl must have size np x nd or np x nd x ne.');
end
if size(v, 2) ~= size(nl, 2)
    error('v and nl must have the same physical dimension nd.');
end
if size(v, 1) ~= size(nl, 1)
    error('v and nl must have matching np dimensions.');
end
if ndims(nl) == 3 && ndims(v) >= 3 && size(v, 3) ~= size(nl, 3)
    error('v and nl must have matching ne dimensions when nl has size np x nd x ne.');
end
if ndims(nl) == 3 && ndims(v) == 2
    error('nl must have size np x nd when v has size np x nd.');
end
end

function nl = expandNormals(v, nl)
if isequal(size(v), size(nl))
    return;
end

if ndims(nl) == 2
    if ndims(v) == 2
        return;
    end
    reps = [1 1 size(v, 3)];
    if ndims(v) == 4
        reps = [reps size(v, 4)];
    end
    nl = repmat(nl, reps);
    return;
end

if ndims(v) == 4
    nl = repmat(nl, [1 1 1 size(v, 4)]);
    return;
end
end
