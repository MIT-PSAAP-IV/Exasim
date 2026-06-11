function [datapath, builddir] = exasimbuilddirs(sharedbuild)
if nargin < 1
    sharedbuild = 1;
end

if sharedbuild == 1
    cdir = pwd(); 
    ii = strfind(cdir, "Exasim");
    sharedroot = cdir(1:(ii+5)) + "/examples/exasimfe/";
    if strlength(sharedroot) > 0
        datapath = sharedroot;
        builddir = sharedroot;
        return;
    end
end

localroot = string(fullfile(pwd(), "exasim"));
datapath = localroot;
builddir = localroot;
end

