function exasimbuilddirs(sharedbuild=1)
    if sharedbuild == 1
        sharedroot = joinpath(exasim_path(), "examples", "exasimfe")
        if !isempty(sharedroot)
            return sharedroot, sharedroot
        end
    end

    localroot = joinpath(pwd(), "exasim")
    return localroot, localroot
end
