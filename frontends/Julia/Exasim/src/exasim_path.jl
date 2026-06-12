function exasim_path()
    cdir = pwd()
    idx = findlast("Exasim", cdir)
    idx === nothing && error("exasim_path: current directory $cdir does not contain 'Exasim' in its path.")
    return cdir[firstindex(cdir):last(idx)]
end
