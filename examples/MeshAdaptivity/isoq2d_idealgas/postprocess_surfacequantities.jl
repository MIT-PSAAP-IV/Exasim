"""Read ISOQ wall surfacequantities saved by Exasim."""

using Printf

function count_surface_ranks(prefix::AbstractString)
    nranks = 0
    while isfile(prefix * "bouinfo_np$(nranks).bin")
        nranks += 1
    end
    return nranks
end

function average_duplicate_points(x, y, nx, ny, cp, cf, cq)
    scale = max(maximum(x)-minimum(x), maximum(y)-minimum(y), 1.0)
    tol = 1.0e-12 * scale
    sums = Dict{Tuple{Int64,Int64},Vector{Float64}}()
    order = Tuple{Int64,Int64}[]
    for i in eachindex(x)
        key = (round(Int64, x[i]/tol), round(Int64, y[i]/tol))
        if !haskey(sums, key)
            sums[key] = zeros(8)
            push!(order, key)
        end
        sums[key] .+= [x[i], y[i], nx[i], ny[i], cp[i], cf[i], cq[i], 1.0]
    end
    n = length(order)
    xo = zeros(n); yo = zeros(n); nxo = zeros(n); nyo = zeros(n)
    cpo = zeros(n); cfo = zeros(n); cqo = zeros(n)
    for (i, key) in enumerate(order)
        s = sums[key]
        c = s[8]
        xo[i] = s[1]/c
        yo[i] = s[2]/c
        nxo[i] = s[3]/c
        nyo[i] = s[4]/c
        cpo[i] = s[5]/c
        cfo[i] = s[6]/c
        cqo[i] = s[7]/c
    end
    return xo, yo, nxo, nyo, cpo, cfo, cqo
end

function wall_arclength(x, y)
    order = sortperm(collect(zip(x, y)))
    xs = x[order]
    ys = y[order]
    s = zeros(length(xs))
    for i in 2:length(xs)
        s[i] = s[i-1] + hypot(xs[i]-xs[i-1], ys[i]-ys[i-1])
    end
    return s, order
end

function write_surface_csv(filename, arclength, x, y, nx, ny, cp, cf, cq)
    open(filename, "w") do io
        println(io, "s,x,y,nx,ny,Cp,Cf,Cq")
        for i in eachindex(arclength)
            println(io, join((
                @sprintf("%.17e", arclength[i]),
                @sprintf("%.17e", x[i]),
                @sprintf("%.17e", y[i]),
                @sprintf("%.17e", nx[i]),
                @sprintf("%.17e", ny[i]),
                @sprintf("%.17e", cp[i]),
                @sprintf("%.17e", cf[i]),
                @sprintf("%.17e", cq[i])), ","))
        end
    end
end

function postprocess_surfacequantities(pde; wallib::Int=3)
    prefix = joinpath(pde.datapath, "dataout", "out")
    nranks = pde.mpiprocs
    if !isfile(prefix * "bouinfo_np$(nranks-1).bin")
        nranks = count_surface_ranks(prefix)
    end
    nranks >= 1 || error("No surface output files found with prefix $prefix")

    surf = readsurfacequantities(prefix, nranks, pde.saveSolBouLoc)
    haskey(surf, wallib) || error("Boundary ID $wallib is absent from saved surface output.")
    wall = surf[wallib]
    tstep = size(wall[:values], 4)
    values = wall[:values][:, :, :, tstep]
    xgeo = wall[:x][:, :, :, tstep]
    ngeo = wall[:n][:, :, :, tstep]

    x = vec(xgeo[:, :, 1])
    y = vec(xgeo[:, :, 2])
    nx = vec(ngeo[:, :, 1])
    ny = vec(ngeo[:, :, 2])
    cp = vec(values[:, :, 1])
    cf = vec(values[:, :, 2])
    cq = vec(values[:, :, 3])
    x, y, nx, ny, cp, cf, cq = average_duplicate_points(x, y, nx, ny, cp, cf, cq)

    all(isfinite, vcat(x, y, nx, ny, cp, cf, cq)) ||
        error("Saved wall coordinates, normals, or quantities contain NaN/Inf.")

    arclength, order = wall_arclength(x, y)
    x = x[order]; y = y[order]
    nx = nx[order]; ny = ny[order]
    cp = cp[order]; cf = cf[order]; cq = cq[order]

    plotdir = joinpath(pde.datapath, "surfacequantities_plots")
    mkpath(plotdir)
    csvfile = joinpath(plotdir, "surfacequantities.csv")
    write_surface_csv(csvfile, arclength, x, y, nx, ny, cp, cf, cq)

    println("Surface quantities read from $prefix with $nranks rank file(s).")
    println("Boundary ID $wallib, save step $tstep, $(length(arclength)) unique points.")
    println("Cp range = [$(minimum(cp)), $(maximum(cp))].")
    println("Cf range = [$(minimum(cf)), $(maximum(cf))].")
    println("Cq range = [$(minimum(cq)), $(maximum(cq))].")
    println("Surface data written to $plotdir.")
    return Dict(
        :s => arclength,
        :x => [x y],
        :n => [nx ny],
        :Cp => cp,
        :Cf => cf,
        :Cq => cq,
        :plotdir => plotdir)
end
