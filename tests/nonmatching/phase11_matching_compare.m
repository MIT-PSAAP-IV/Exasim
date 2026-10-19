function relative = phase11_matching_compare(elemtype, ranksPerModel, porder)
%PHASE11_MATCHING_COMPARE Compare complete type-0/type-1 3D solutions.

if nargin < 3
    if elemtype == 0, porder = 1; else, porder = 2; end
end
conformal = phase11_preflight(elemtype, ranksPerModel, 0, porder);
nonmatching = phase11_preflight(elemtype, ranksPerModel, 1, porder);
relative = zeros(2,1);
for model = 1:2
    delta = nonmatching.sol{model} - conformal.sol{model};
    relative(model) = norm(delta(:))/max(norm(conformal.sol{model}(:)), 1);
end
assert(max(relative) < 5e-12, ...
    '3D matching type-0/type-1 solution mismatch: %.17e', max(relative));
fprintf('PASS phase11_matching_compare %s %d+%d relative %.17e %.17e\n', ...
    conformal.family, ranksPerModel, ranksPerModel, relative(1), relative(2));
end
