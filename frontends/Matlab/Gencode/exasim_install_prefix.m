function prefix = exasim_install_prefix()
% Locate the Exasim install this frontend belongs to.
%
% The frontend is installed to <prefix>/share/exasim/matlab by
% `cmake --install`, so the prefix is normally derived from this file's
% location. Override with the EXASIM_PREFIX environment variable (e.g. for
% a source-tree checkout, point it at <repo>/build/install).

env = getenv('EXASIM_PREFIX');
if ~isempty(env)
    if isprefix(env)
        prefix = string(env);
        return;
    end
    error("EXASIM_PREFIX=%s does not contain lib/cmake/Exasim/ExasimConfig.cmake", env);
end

here = fileparts(mfilename('fullpath'));   % .../share/exasim/matlab/Gencode
% Installed layout: <prefix>/share/exasim/matlab/Gencode
cand = fullfile(here, '..', '..', '..', '..');
if isprefix(cand)
    prefix = string(absolutepath(cand));
    return;
end
% Source-tree layout: <repo>/frontends/Matlab/Gencode with the superbuild
% installed to <repo>-build/install (sibling of the repo).
repodir = absolutepath(fullfile(here, '..', '..', '..'));
cand = fullfile(repodir, 'local');
if isprefix(cand)
    prefix = string(absolutepath(cand));
    return;
end
[repoparent, reponame] = fileparts(repodir);
cand = fullfile(repoparent, [reponame '-build'], 'install');
if isprefix(cand)
    prefix = string(absolutepath(cand));
    return;
end
cand = fullfile(repodir, 'build', 'install');
if isprefix(cand)
    prefix = string(absolutepath(cand));
    return;
end
error("Cannot locate an Exasim install. Build and install Exasim " + ...
      "(cmake -S <repo> -B <repo>-build, build, install) " + ...
      "and/or set EXASIM_PREFIX to the install prefix.");
end

function tf = isprefix(p)
tf = exist(fullfile(p, 'lib', 'cmake', 'Exasim', 'ExasimConfig.cmake'), 'file') == 2;
end

function p = absolutepath(p)
d = dir(p);
if ~isempty(d)
    p = d(1).folder;
end
end
