function surf = readsurfacequantities(fileprefix, nranks, saveSolBouLoc)
%READSURFACEQUANTITIES Read the pdemodel surfacequantities written on the ibs boundaries.
%
%   surf = readsurfacequantities(fileprefix, nranks)
%   surf = readsurfacequantities(fileprefix, nranks, saveSolBouLoc)
%
%   fileprefix    output path prefix, e.g. [pde.buildpath '/dataout/out']; the
%                 files read are [fileprefix 'bou*_np<r>.bin'], r = 0..nranks-1.
%   nranks        number of MPI ranks (output files) of the run (default 1).
%   saveSolBouLoc 0 = values at face nodes, 1 = at face Gauss points. Optional:
%                 by default it is 1 iff outbousurfgeo_np0.bin exists (pass it
%                 explicitly if a stale file from an earlier run may be present).
%
%   Returns a struct array with one entry per boundary id ib, with faces
%   concatenated over ranks and over the face blocks of that boundary:
%     surf(k).ib           boundary index
%     surf(k).values       [np, nf, nsurfq, nsteps]  (np = npf, or ngf if loc 1)
%     surf(k).x            [np, nf, ncx]  at the points of values
%     surf(k).n            [np, nf, nd]   at the points of values
%     surf(k).dA           [np, nf]       (loc 1 only) Gauss weight * face
%                          jacobian, so sum(f.*dA,'all') integrates f
%     surf(k).saveSolBouLoc
%
%   File formats (float64, 3-double header each):
%     outbouinfo_np<r>.bin    [nblk 2 0], then (ib, nf) per block in file order
%     outbousurf_np<r>.bin    [np nfbou nsurfq], then per saved step the blocks'
%                             [np, nf_b, nsurfq] arrays concatenated
%     outbouxdg_np<r>.bin     [npf nfbou ncx], per block [npf, nf_b, ncx]
%     outboundg_np<r>.bin     [npf nfbou nd],  per block [npf, nf_b, nd]
%     outbousurfgeo_np<r>.bin [ngf nfbou ncx+nd+1], per block x [ngf*nf_b*ncx],
%                             n [ngf*nf_b*nd], dA [ngf*nf_b]   (loc 1 only)

if nargin < 2 || isempty(nranks), nranks = 1; end
fileprefix = char(fileprefix);
if nargin < 3 || isempty(saveSolBouLoc)
    saveSolBouLoc = double(isfile([fileprefix 'bousurfgeo_np0.bin']));
end
if ~any(saveSolBouLoc == [0 1])
    error('readsurfacequantities: saveSolBouLoc must be 0 or 1.');
end

ids = [];
acc = struct('values', {}, 'x', {}, 'n', {}, 'dA', {});

for r = 0:nranks-1
    tag = ['_np' num2str(r) '.bin'];
    info = readbin([fileprefix 'bouinfo' tag]);
    nblk = info(1);
    blk = reshape(info(4:3+2*nblk), 2, nblk);   % rows: ib, nf
    if nblk == 0, continue; end

    s = readbin([fileprefix 'bousurf' tag]);
    np = s(1); nfbou = s(2); nsq = s(3);
    if sum(blk(2,:)) ~= nfbou
        error('readsurfacequantities: %s disagrees with outbouinfo on the face count.', ['outbousurf' tag]);
    end
    rec = np*nfbou*nsq;
    data = s(4:end);
    nsteps = floor(numel(data)/max(rec,1));
    if rec > 0 && numel(data) ~= nsteps*rec
        warning('readsurfacequantities: %s has a partial trailing record (ignored).', ['outbousurf' tag]);
    end
    data = reshape(data(1:nsteps*rec), rec, nsteps);

    if saveSolBouLoc == 1
        g = readbin([fileprefix 'bousurfgeo' tag]);
        ngf = g(1); ncg = g(3);
        if ngf ~= np || g(2) ~= nfbou
            error('readsurfacequantities: outbousurfgeo header does not match outbousurf.');
        end
        geo = g(4:end);
        nd = readheader([fileprefix 'boundg' tag]);   % [npf nfbou nd]
        nd = nd(3);
        ncx = ncg - nd - 1;
    else
        x = readbin([fileprefix 'bouxdg' tag]);
        n = readbin([fileprefix 'boundg' tag]);
        if x(1) ~= np || n(1) ~= np
            error(['readsurfacequantities: outbousurf has %d points per face but outbouxdg has %d; ' ...
                   'pass saveSolBouLoc explicitly.'], np, x(1));
        end
        ncx = x(3); nd = n(3);
        x = x(4:end); n = n(4:end);
    end

    o = 0;   % offset (in faces*np) into a record / nodal geometry files
    og = 0;  % offset into the Gauss geometry file
    for b = 1:nblk
        ib = blk(1,b); nf = blk(2,b);
        m = np*nf;
        vals = reshape(data(o*nsq + (1:m*nsq), :), np, nf, nsq, nsteps);
        if saveSolBouLoc == 1
            xb = reshape(geo(og + (1:m*ncx)), np, nf, ncx);
            nb = reshape(geo(og + m*ncx + (1:m*nd)), np, nf, nd);
            dAb = reshape(geo(og + m*(ncx+nd) + (1:m)), np, nf);
            og = og + m*ncg;
        else
            xb = reshape(x(o*ncx + (1:m*ncx)), np, nf, ncx);
            nb = reshape(n(o*nd + (1:m*nd)), np, nf, nd);
            dAb = [];
        end
        o = o + m;

        k = find(ids == ib, 1);
        if isempty(k)
            ids(end+1) = ib; %#ok<AGROW>
            k = numel(ids);
            acc(k).values = vals; acc(k).x = xb; acc(k).n = nb; acc(k).dA = dAb;
        else
            if size(acc(k).values, 4) ~= nsteps
                error('readsurfacequantities: ranks disagree on the number of saved steps.');
            end
            acc(k).values = cat(2, acc(k).values, vals);
            acc(k).x = cat(2, acc(k).x, xb);
            acc(k).n = cat(2, acc(k).n, nb);
            acc(k).dA = cat(2, acc(k).dA, dAb);
        end
    end
end

[ids, order] = sort(ids);
acc = acc(order);
surf = struct('ib', {}, 'values', {}, 'x', {}, 'n', {}, 'dA', {}, 'saveSolBouLoc', {});
for k = 1:numel(ids)
    surf(k).ib = ids(k);
    surf(k).values = acc(k).values;
    surf(k).x = acc(k).x;
    surf(k).n = acc(k).n;
    surf(k).dA = acc(k).dA;
    surf(k).saveSolBouLoc = saveSolBouLoc;
end

end

function h = readheader(filename)
fid = fopen(filename, 'r');
if fid < 0
    error('readsurfacequantities: cannot open %s', filename);
end
h = fread(fid, 3, 'double');
fclose(fid);
end

function a = readbin(filename)
fid = fopen(filename, 'r');
if fid < 0
    error('readsurfacequantities: cannot open %s', filename);
end
a = fread(fid, inf, 'double');
fclose(fid);
end
