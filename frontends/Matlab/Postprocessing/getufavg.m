function uf = getufavg(base, nprocs, npf, ncu)

uf = [];
for i = 1:nprocs
    fname = base + "_np" + string(i-1) + ".bin";
    fid = fopen(fname, 'r');
    if fid < 0
        continue;
    end
    tm = fread(fid, 'double').';
    nsteps = tm(end);
    tm = tm(1:end-1)/nsteps;
    tm = reshape(tm, npf, [], ncu);
    if isempty(uf)
        uf = tm;
    else
        uf = cat(2, uf, tm);
    end
    fclose(fid);
end
