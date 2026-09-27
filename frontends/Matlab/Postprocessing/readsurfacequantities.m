function surf = readsurfacequantities(fileprefix, nranks, saveSolBouLoc)
%READSURFACEQUANTITIES Read the pdemodel surfacequantities written on the ibs boundaries.
%
%   surf = readsurfacequantities(fileprefix, nranks)
%   surf = readsurfacequantities(fileprefix, nranks, saveSolBouLoc)
%
%   fileprefix    output path prefix, e.g. [pde.buildpath '/dataout/out']; the
%                 files read are [fileprefix 'bou*_np<r>.bin'], r = 0..nranks-1.
%   nranks        number of MPI ranks (output files) of the run (default 1).
%   saveSolBouLoc optional check of the run's evaluation location (0 = face
%                 nodes, 1 = face Gauss points); the files themselves say which.
%
%   Returns a struct array with one entry per boundary id ib, with faces
%   concatenated over ranks and over the face blocks of that boundary:
%     surf(k).ib           boundary index
%     surf(k).values       [np, nf, nsurfq, nsteps]  (np = npf, or ngf if loc 1)
%     surf(k).x            [np, nf, ncx, nsteps] coordinates at each save
%     surf(k).n            [np, nf, nd, nsteps]  normals at each save
%     surf(k).dA           [np, nf, nsteps] (loc 1 only, else []) Gauss weight *
%                          face jacobian, so sum(values(:,:,i,t).*dA(:,:,t),'all')
%                          integrates quantity i
%     surf(k).saveSolBouLoc
%   Geometry is saved with every record, so it follows a moving or adapted mesh.
%
%   File formats (float64, 3-double header each):
%     outbouinfo_np<r>.bin    [nblk 2 0], then (ib, nf) per block in file order
%     outbousurf_np<r>.bin    [np nfbou nsurfq], then per save the blocks'
%                             [np, nf_b, nsurfq] arrays concatenated
%     outbousurfgeo_np<r>.bin [np nfbou ncx+nd+loc], then per save per block
%                             x [np*nf_b*ncx], n [np*nf_b*nd] (+ dA [np*nf_b] if loc 1)
%     outbouxdg / outboundg   headers give ncx / nd

if nargin < 2 || isempty(nranks), nranks = 1; end
fileprefix = char(fileprefix);

% Pass 1: layout -- blocks per rank, faces per boundary, the common dimensions.
blocks = cell(nranks, 1);
dims = [];
for r = 0:nranks-1
    info = readbin([fileprefix 'bouinfo_np' num2str(r) '.bin']);
    nblk = round(info(1));
    b = reshape(round(info(4:3+2*nblk)), 2, nblk)';   % rows (ib, nf)
    blocks{r+1} = b;
    if isempty(b) || sum(b(:,2)) == 0, continue; end
    nfbou = sum(b(:,2));
    [hs, nsv] = header([fileprefix 'bousurf_np' num2str(r) '.bin']);
    [hg, ngv] = header([fileprefix 'bousurfgeo_np' num2str(r) '.bin']);
    hx = header([fileprefix 'bouxdg_np' num2str(r) '.bin']);
    hn = header([fileprefix 'boundg_np' num2str(r) '.bin']);
    if hs(2) ~= nfbou || hg(2) ~= nfbou || hg(1) ~= hs(1)
        error('readsurfacequantities: bousurf/bousurfgeo/bouinfo disagree on rank %d.', r);
    end
    loc = hg(3) - hx(3) - hn(3);       % 0: [x n] at nodes, 1: [x n dA] at Gauss points
    if ~any(loc == [0 1])
        error('readsurfacequantities: unexpected bousurfgeo width %d on rank %d.', hg(3), r);
    end
    reclen = hs(1)*nfbou*hs(3);
    nsteps = 0; if reclen > 0, nsteps = floor(nsv/reclen); end
    if nsteps*hs(1)*nfbou*hg(3) ~= ngv
        error('readsurfacequantities: bousurfgeo does not hold one record per save on rank %d.', r);
    end
    d = [hs(1) hs(3) hx(3) hn(3) nsteps loc];   % np nsq ncx nd nsteps loc
    if isempty(dims)
        dims = d;
    elseif any(d ~= dims)
        error('readsurfacequantities: ranks disagree on the output layout (rank %d).', r);
    end
end
surf = struct('ib', {}, 'values', {}, 'x', {}, 'n', {}, 'dA', {}, 'saveSolBouLoc', {});
if isempty(dims), return; end
np = dims(1); nsq = dims(2); ncx = dims(3); nd = dims(4); nsteps = dims(5); loc = dims(6);
if nargin >= 3 && ~isempty(saveSolBouLoc) && saveSolBouLoc ~= loc
    error('readsurfacequantities: files were written with saveSolBouLoc = %d.', loc);
end

% Allocate every result once.
allb = vertcat(blocks{:});
ids = unique(allb(:,1))';
for k = 1:numel(ids)
    nf = sum(allb(allb(:,1) == ids(k), 2));
    surf(k).ib = ids(k);
    surf(k).values = zeros(np, nf, nsq, nsteps);
    surf(k).x = zeros(np, nf, ncx, nsteps);
    surf(k).n = zeros(np, nf, nd, nsteps);
    if loc == 1, surf(k).dA = zeros(np, nf, nsteps); else, surf(k).dA = []; end
    surf(k).saveSolBouLoc = loc;
end
ngeo = ncx + nd + loc;

% Pass 2: fill by face offset per boundary.
offset = zeros(1, numel(ids));
for r = 0:nranks-1
    b = blocks{r+1};
    if isempty(b) || sum(b(:,2)) == 0, continue; end
    nfbou = sum(b(:,2));
    sdata = readbin([fileprefix 'bousurf_np' num2str(r) '.bin']);    sdata = sdata(4:end);
    gdata = readbin([fileprefix 'bousurfgeo_np' num2str(r) '.bin']); gdata = gdata(4:end);
    srec = np*nfbou*nsq; grec = np*nfbou*ngeo;
    sk = 0; gk = 0;
    for j = 1:size(b,1)
        k = find(ids == b(j,1)); nf = b(j,2); m = np*nf;
        cols = offset(k) + (1:nf);
        for t = 1:nsteps
            s0 = (t-1)*srec + sk;
            surf(k).values(:, cols, :, t) = reshape(sdata(s0 + (1:m*nsq)), np, nf, nsq);
            g0 = (t-1)*grec + gk;
            surf(k).x(:, cols, :, t) = reshape(gdata(g0 + (1:m*ncx)), np, nf, ncx);
            g0 = g0 + m*ncx;
            surf(k).n(:, cols, :, t) = reshape(gdata(g0 + (1:m*nd)), np, nf, nd);
            if loc == 1
                g0 = g0 + m*nd;
                surf(k).dA(:, cols, t) = reshape(gdata(g0 + (1:m)), np, nf);
            end
        end
        sk = sk + m*nsq; gk = gk + m*ngeo;
        offset(k) = offset(k) + nf;
    end
end
end

function a = readbin(filename)
fid = fopen(filename, 'r');
if fid < 0, error('readsurfacequantities: cannot open %s', filename); end
a = fread(fid, inf, 'double');
fclose(fid);
end

function [h, nvals] = header(filename)
fid = fopen(filename, 'r');
if fid < 0, error('readsurfacequantities: cannot open %s', filename); end
h = round(fread(fid, 3, 'double'))';
fclose(fid);
d = dir(filename);
nvals = d.bytes/8 - 3;
end
