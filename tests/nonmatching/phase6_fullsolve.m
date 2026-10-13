function phase6_fullsolve(ranksPerModel)
%PHASE6_FULLSOLVE Compatibility wrapper for the Phase 6 coupled acceptance test.
%
% [nonmatching] Phase 9 generalized the Phase 6 harness to phase9_fullsolve.m
% (arbitrary k1+k2 splits, pde.interfacetype instead of pde.problem, complete
% Newton/GMRES histories, and the production cmakecompile(pde{1}, k1+k2) call).
% The former 1+1 compile workaround (max(2, mpiprocs)) is gone: cmakecompile now
% selects the MPI variant from the total coupled rank count.
%
% EXASIM_PHASE6_PREFIX remains accepted as the install prefix.

if nargin == 0
    ranksPerModel = 1;
end
phase9_fullsolve('matching', ranksPerModel, ranksPerModel);
if ranksPerModel == 1
    phase9_fullsolve('nonlinear', 1, 1);
end
fprintf('PASS phase6_fullsolve %d+%d\n', ranksPerModel, ranksPerModel);
end
