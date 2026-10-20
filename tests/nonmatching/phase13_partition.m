function phase13_partition
%PHASE13_PARTITION Manual unequal-rank and nonlinear partition invariance sweep.
splits=[1 1;1 3;3 1;2 2]; orders={'A','B'};
for o=1:2
 sequence=[orders{o} '-M2']; if orders{o}=='A',sequence='A-M1';end
 reference=[];
 for s=1:size(splits,1)
  out=phase13_fullsolve(orders{o},sequence,2,2,splits(s,1),splits(s,2),true);
  for model=1:2
   counts=out.interfaceElements{model};
   assert(numel(counts)==splits(s,model),'model %d rank count does not match split',model);
   countsText=strtrim(sprintf('%d ',counts));
   assert(all(counts>0), ...
    'model %d split %d+%d has a rank without interface elements: [%s]', ...
    model,splits(s,1),splits(s,2),countsText);
  end
  values=cellfun(@(u)u(:),out.sol,'UniformOutput',false); values=vertcat(values{:});
  if isempty(reference),reference=values;else
   relative=norm(values-reference)/max(norm(reference),realmin);
   fprintf('PHASE13 partition %s %d+%d relative %.9e\n',orders{o},splits(s,1),splits(s,2),relative);
   assert(relative<=1e-10,'partition dependence exceeds 1e-10');
  end
 end
end
fprintf('PASS phase13_partition\n');
end
