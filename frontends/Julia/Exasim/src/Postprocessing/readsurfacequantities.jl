"""
    readsurfacequantities(prefix, nranks, saveSolBouLoc=nothing) -> Dict{Int,Dict{Symbol,Any}}

Read the pointwise surface quantities written by the backend when the model
defines `surfacequantities` and `pde.saveSolBouFreq > 0` (boundaries `pde.ibs`).

`prefix` is the output prefix, e.g. `joinpath(pde.datapath, "dataout", "out")`;
the files `prefix * "bouinfo_np<r>.bin"`, `prefix * "bousurf_np<r>.bin"`, ...
are read for ranks r = 0..nranks-1 (ranks without an info file are skipped).
`saveSolBouLoc` (0 face nodes, 1 face Gauss points) is the run's evaluation
location; `nothing` infers 1 when `outbousurfgeo_np0.bin` exists. Pass it
explicitly if the output directory may hold stale files from another run.

Returns a Dict keyed by boundary index `ib`, each entry a Dict with
- `:values` `[np, nf, nsurfq, nsteps]` (np = npf for saveSolBouLoc = 0, ngf for 1)
- `:x`      `[np, nf, ncx]` coordinates of those points
- `:n`      `[np, nf, nd]`  unit normals at those points
- `:dA`     `[np, nf]` Gauss weight * face jacobian (saveSolBouLoc = 1 only, else
            `nothing`), so `sum(values[:, :, i, t] .* dA)` integrates quantity i
- `:saveSolBouLoc` 0 or 1
Faces are concatenated over ranks and blocks with the same `ib`.
"""
function readsurfacequantities(prefix::AbstractString, nranks::Integer, saveSolBouLoc=nothing)
    rd(fn) = open(fn, "r") do io
        read!(io, Vector{Float64}(undef, filesize(fn) ÷ 8))
    end
    loc = saveSolBouLoc === nothing ? Int(isfile(prefix * "bousurfgeo_np0.bin")) : Int(saveSolBouLoc)
    loc in (0, 1) || error("readsurfacequantities: saveSolBouLoc must be 0 or 1")
    out = Dict{Int,Dict{Symbol,Any}}()
    catdim(a, b, d) = a === nothing ? b : cat(a, b; dims=d)
    function add!(ib, key, arr, d)
        e = get!(out, ib, Dict{Symbol,Any}(:dA => nothing, :saveSolBouLoc => loc))
        e[key] = catdim(get(e, key, nothing), arr, d)
    end
    for r in 0:nranks-1
        finfo = prefix * "bouinfo_np$(r).bin"
        isfile(finfo) || continue
        info = rd(finfo)
        nblk = Int(info[1])
        blocks = [(Int(info[3+2k+1]), Int(info[3+2k+2])) for k in 0:nblk-1]
        nftot = sum(b[2] for b in blocks; init=0)
        nftot > 0 || continue   # this rank owns no face on the ibs boundaries

        fsurf = prefix * "bousurf_np$(r).bin"
        a = rd(fsurf)
        np, nfb, nsq = Int(a[1]), Int(a[2]), Int(a[3])
        nfb == nftot || error("readsurfacequantities: $fsurf has nfbou=$nfb, bouinfo says $nftot")
        reclen = np * nfb * nsq
        data = a[4:end]
        nsteps = reclen > 0 ? length(data) ÷ reclen : 0
        data = reshape(data[1:reclen*nsteps], reclen, nsteps)
        off = 0
        for (ib, nf) in blocks
            n = np * nf * nsq
            add!(ib, :values, reshape(data[off+1:off+n, :], np, nf, nsq, nsteps), 2)
            off += n
        end

        # outboundg's header gives nd; outbouxdg's gives ncx.
        hx, hn = rd(prefix * "bouxdg_np$(r).bin"), rd(prefix * "boundg_np$(r).bin")
        ncx, nd = Int(hx[3]), Int(hn[3])
        if loc == 0
            # face nodes: the geometry is exactly outbouxdg / outboundg
            for (a, key, nc) in ((hx, :x, ncx), (hn, :n, nd))
                npf = Int(a[1])
                npf == np || error("readsurfacequantities: rank $r has $np points per face but npf=$npf; pass saveSolBouLoc=1?")
                off = 3
                for (ib, nf) in blocks
                    n = npf * nf * nc
                    add!(ib, key, reshape(a[off+1:off+n], npf, nf, nc), 2)
                    off += n
                end
            end
        else
            # face Gauss points: per block [x | n | dA] from outbousurfgeo
            a = rd(prefix * "bousurfgeo_np$(r).bin")
            ngf = Int(a[1])
            ngf == np || error("readsurfacequantities: bousurfgeo/bousurf point counts differ on rank $r")
            off = 3
            for (ib, nf) in blocks
                m = ngf * nf
                add!(ib, :x, reshape(a[off+1:off+m*ncx], ngf, nf, ncx), 2); off += m*ncx
                add!(ib, :n, reshape(a[off+1:off+m*nd], ngf, nf, nd), 2);   off += m*nd
                add!(ib, :dA, reshape(a[off+1:off+m], ngf, nf), 2);          off += m
            end
        end
    end
    return out
end
