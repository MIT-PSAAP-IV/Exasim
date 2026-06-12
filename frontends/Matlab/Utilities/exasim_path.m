function exasimpath = exasim_path()
%EXASIM_PATH Return the absolute path to the Exasim repo root inferred from pwd.

cdir = pwd();
ii = strfind(cdir, "Exasim");
if isempty(ii)
    error("exasim_path:NotFound", ...
        "Current directory %s does not contain 'Exasim' in its path.", cdir);
end

ii = ii(end);
exasimpath = cdir(1:(ii+5));

end
