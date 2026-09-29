"""
    readsurfacequantities(prefix, nranks, saveSolBouLoc=nothing) -> Dict{Int,Dict{Symbol,Any}}

Read the pointwise surface quantities written by the backend when the model
defines `surfacequantities` and `pde.saveSolBouFreq > 0` (boundaries `pde.ibs`).

`prefix` is the output prefix, e.g. `joinpath(pde.datapath, "dataout", "out")`;
the files `prefix * "bouinfo_np<r>.bin"`, `prefix * "bousurf_np<r>.bin"`, ...
are read for ranks r = 0..nranks-1. `saveSolBouLoc` optionally checks the run's
evaluation location (0 face nodes, 1 face Gauss points); the files say which.

Returns a Dict keyed by boundary index `ib`, each entry a Dict with
- `:values` `[np, nf, nsurfq, nsteps]` (np = npf for saveSolBouLoc = 0, ngf for 1)
- `:x`      `[np, nf, ncx, nsteps]` coordinates of those points at each save
- `:n`      `[np, nf, nd, nsteps]`  unit normals at those points at each save
- `:dA`     `[np, nf, nsteps]` Gauss weight * face jacobian (saveSolBouLoc = 1 only, else
            `nothing`), so `sum(values[:, :, i, t] .* dA[:, :, t])` integrates quantity i
- `:saveSolBouLoc` 0 or 1
Faces are concatenated over ranks and blocks with the same `ib`. Geometry is saved with
every record, so it follows a moving or adapted mesh.
"""
function readsurfacequantities(prefix::AbstractString, nranks::Integer, saveSolBouLoc=nothing)
    header(fn) = (h = open(fn, "r") do io; read!(io, Vector{Float64}(undef, 3)); end;
                  (round.(Int, h), filesize(fn) ÷ 8 - 3))
    payload(fn) = open(fn, "r") do io
        read!(io, Vector{Float64}(undef, filesize(fn) ÷ 8))[4:end]
    end

    # Pass 1: layout -- blocks per rank, faces per boundary, the common dimensions.
    blocks = Vector{Vector{Tuple{Int,Int}}}(undef, nranks)
    nftot = Dict{Int,Int}()
    dims = nothing
    for r in 0:nranks-1
        info = payload(prefix * "bouinfo_np$(r).bin")
        hinfo, _ = header(prefix * "bouinfo_np$(r).bin")
        blocks[r+1] = [(round(Int, info[2k+1]), round(Int, info[2k+2])) for k in 0:hinfo[1]-1]
        nfbou = sum(b[2] for b in blocks[r+1]; init=0)
        nfbou > 0 || continue   # this rank owns no face on the ibs boundaries
        hs, nsv = header(prefix * "bousurf_np$(r).bin")
        hg, ngv = header(prefix * "bousurfgeo_np$(r).bin")
        hx, _ = header(prefix * "bouxdg_np$(r).bin")
        hn, _ = header(prefix * "boundg_np$(r).bin")
        (hs[2] == nfbou && hg[2] == nfbou && hg[1] == hs[1]) ||
            error("readsurfacequantities: bousurf/bousurfgeo/bouinfo disagree on rank $r")
        loc = hg[3] - hx[3] - hn[3]   # 0: [x, n] at nodes, 1: [x, n, dA] at Gauss points
        loc in (0, 1) || error("readsurfacequantities: unexpected bousurfgeo width $(hg[3]) on rank $r")
        reclen = hs[1] * nfbou * hs[3]
        nsteps = reclen > 0 ? nsv ÷ reclen : 0
        nsteps * hs[1] * nfbou * hg[3] == ngv ||
            error("readsurfacequantities: bousurfgeo does not hold one record per save on rank $r")
        d = (hs[1], hs[3], hx[3], hn[3], nsteps, loc)
        if dims === nothing
            dims = d
        elseif d != dims
            error("readsurfacequantities: ranks disagree on the output layout (rank $r)")
        end
        for (ib, nf) in blocks[r+1]
            nftot[ib] = get(nftot, ib, 0) + nf
        end
    end
    out = Dict{Int,Dict{Symbol,Any}}()
    dims === nothing && return out
    np, nsq, ncx, nd, nsteps, loc = dims
    saveSolBouLoc === nothing || Int(saveSolBouLoc) == loc ||
        error("readsurfacequantities: files were written with saveSolBouLoc = $loc")

    # Allocate every result once.
    for (ib, nf) in nftot
        out[ib] = Dict{Symbol,Any}(
            :values => zeros(np, nf, nsq, nsteps),
            :x => zeros(np, nf, ncx, nsteps),
            :n => zeros(np, nf, nd, nsteps),
            :dA => loc == 1 ? zeros(np, nf, nsteps) : nothing,
            :saveSolBouLoc => loc)
    end
    ngeo = ncx + nd + loc

    # Pass 2: fill by face offset per boundary.
    offset = Dict(ib => 0 for ib in keys(nftot))
    for r in 0:nranks-1
        blks = blocks[r+1]
        nfbou = sum(b[2] for b in blks; init=0)
        nfbou > 0 || continue
        sdata = payload(prefix * "bousurf_np$(r).bin")
        gdata = payload(prefix * "bousurfgeo_np$(r).bin")
        srec = np * nfbou * nsq
        grec = np * nfbou * ngeo
        sk = 0; gk = 0
        for (ib, nf) in blks
            e = out[ib]; o = offset[ib]; m = np * nf
            cols = o+1:o+nf
            for t in 1:nsteps
                s0 = (t-1) * srec + sk
                e[:values][:, cols, :, t] = reshape(view(sdata, s0+1:s0+m*nsq), np, nf, nsq)
                g0 = (t-1) * grec + gk
                e[:x][:, cols, :, t] = reshape(view(gdata, g0+1:g0+m*ncx), np, nf, ncx); g0 += m*ncx
                e[:n][:, cols, :, t] = reshape(view(gdata, g0+1:g0+m*nd), np, nf, nd);   g0 += m*nd
                loc == 1 && (e[:dA][:, cols, t] = reshape(view(gdata, g0+1:g0+m), np, nf))
            end
            sk += m * nsq; gk += m * ngeo
            offset[ib] = o + nf
        end
    end
    return out
end
