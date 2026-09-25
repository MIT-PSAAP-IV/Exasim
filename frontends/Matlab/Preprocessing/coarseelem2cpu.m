function elem2cpu = coarseelem2cpu(dmd, ne)
%COARSEELEM2CPU Recover zero-based element owner ranks from a DMD partition.

nproc = numel(dmd);
elem2cpu = -ones(ne, 1);

for i = 1:nproc
    owned = sum(dmd{i}.elempartpts(1:2));
    elem = dmd{i}.elempart(1:owned);
    elem2cpu(elem) = i - 1;
end

if any(elem2cpu < 0)
    error('coarseelem2cpu: some coarse elements do not have an owner rank.');
end

end
