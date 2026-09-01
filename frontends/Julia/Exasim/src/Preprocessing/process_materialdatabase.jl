function process_materialdatabase(pde, output_dir, output_name="materialdatabase.bin")
    source = strip(String(_materialdatabase_get(pde, "materialdatabase", "")))
    isempty(source) && return nothing

    isfile(source) || error("pde.materialdatabase file not found: $source")

    destination = joinpath(output_dir, String(output_name))
    ext = lowercase(splitext(source)[2])
    if ext == ".bin"
        cp(source, destination; force=true)
    elseif ext == ".dat"
        values = _read_materialdatabase_dat_values(source)
        open(destination, "w") do io
            write(io, Float64.(values))
        end
    else
        error("Unsupported pde.materialdatabase format '$ext'. Expected .dat or .bin.")
    end
    return destination
end

function _materialdatabase_get(pde, key, default)
    if pde isa AbstractDict
        return get(pde, key, get(pde, Symbol(key), default))
    end
    sym = Symbol(key)
    return hasproperty(pde, sym) ? getproperty(pde, sym) : default
end

function _read_materialdatabase_dat_values(filename)
    values = Float64[]
    for rawline in eachline(filename)
        line = replace(rawline, r"(#|%|//).*$" => "")
        line = strip(line)
        isempty(line) && continue
        append!(values, parse.(Float64, split(line)))
    end
    return values
end
