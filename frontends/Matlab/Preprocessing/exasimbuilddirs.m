function [datapath, builddir] = exasimbuilddirs(sharedbuild)
if nargin < 1
    sharedbuild = 1;
end

if sharedbuild == 1
    sharedroot = string(fullfile(exasim_path(), "examples", "exasimfe"));
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
