function uhat = getuhat(basename, nproc)
%GETUHAT Assemble HDG trace solution from per-rank dataout/uhat_np*.bin files.
%
% Each rank file stores:
%   header: [ncu, npf, nf_rank]
%   data:   ncu x npf x nf_rank x nsteps

if nargin < 2
    nproc = 1;
end

%basename = pde.datapath + "/dataout/outuhat";

uhat = [];
for r = 0:nproc-1
    [ncu, npf, nf, nsteps, block] = read_rank(rankfile(basename, r));

    if r == 0
        uhat = block;
    else
        if size(block,1) ~= size(uhat,1) || size(block,2) ~= size(uhat,2) || ...
           size(block,4) ~= size(uhat,4)
            error("getuhat:IncompatibleRankData", ...
                  "Rank %d has incompatible uhat dimensions.", r);
        end
        uhat = cat(3, uhat, block);
    end
end

end

function fname = rankfile(base, r)
fname = base + "_np" + string(r) + ".bin";
end