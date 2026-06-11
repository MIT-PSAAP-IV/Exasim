% Compatibility shim for legacy pdeapps that run this script after computing
% cdir/ii. The frontend entry point is now frontends/Matlab/exasim_setup.m
% (installed to <prefix>/share/exasim/matlab/exasim_setup.m).
ExasimPath = cdir(1:(ii+5));
run(char(ExasimPath + "/frontends/Matlab/exasim_setup.m"));
clear ExasimPath;
