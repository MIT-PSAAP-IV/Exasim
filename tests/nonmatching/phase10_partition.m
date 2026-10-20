function results = phase10_partition()
%PHASE10_PARTITION Compare mixed-family solutions in global element order.
splits = [1 1; 2 2; 4 4; 1 3; 3 1];
results = struct('orientation', {}, 'k1', {}, 'k2', {}, 'relative', {});
for orientation = {'tri-quad','quad-tri'}
    ref = phase10_fullsolve(orientation{1}, 1, 1, 2, 2, 'A', false, 0);
    scale = max([max(abs(ref.sol{1}(:))) max(abs(ref.sol{2}(:))) 1]);
    for s = 1:size(splits,1)
        if all(splits(s,:) == [1 1]), out = ref;
        else, out = phase10_fullsolve(orientation{1}, splits(s,1), splits(s,2), 2, 2, 'A', false, 0); end
        diff = max([max(abs(out.sol{1}(:)-ref.sol{1}(:))) max(abs(out.sol{2}(:)-ref.sol{2}(:)))])/scale;
        assert(diff <= 1e-12, 'Phase 10 linear partition difference %.3e exceeds 1e-12', diff);
        results(end+1) = struct('orientation', orientation{1}, 'k1', splits(s,1), ...
            'k2', splits(s,2), 'relative', diff); %#ok<AGROW>
        fprintf('PHASE10 partition %s %d+%d relative %.6e\n', ...
            orientation{1}, splits(s,1), splits(s,2), diff);
    end
end
fprintf('PASS phase10_partition\n');
end
