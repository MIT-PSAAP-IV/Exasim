function results=phase13_convergence
%PHASE13_CONVERGENCE Manual A-M1 normative and mixed-sequence diagnostic sweep.
sequences={'A-M1','A-M1-fine','A-M2','B-M2'}; results=struct([]);
for p=1:4
 if p==1, levels=[2 4 8 16]; else, levels=[1 2 4 8]; end
 strict=p+1-0.2;
 fprintf('PHASE13 thresholds p=%d: tet rate >= %.6f; E rate >= min(%.6f,r_hex-0.1)\n',p,strict,strict);
 for s=1:numel(sequences)
  seq=sequences{s}; ordering=seq(1); e1=[]; e2=[]; E=[]; hex=[];
  for N=levels
   out=phase13_fullsolve(ordering,seq,p,N,1,1,false);
   e1(end+1)=out.e1; e2(end+1)=out.e2; E(end+1)=out.E; %#ok<AGROW>
   if strcmp(seq,'A-M1'), base=phase13_hex_baseline(p,N); hex(end+1)=base.error; end %#ok<AGROW>
   results=[results out]; %#ok<AGROW>
  end
  r1=log2(e1(1:end-1)./e1(2:end)); r2=log2(e2(1:end-1)./e2(2:end)); rE=log2(E(1:end-1)./E(2:end));
  fprintf('PHASE13 %s p=%d errors e1={%s} e2={%s} E={%s}\n',seq,p,num2str(e1,' %.9e'),num2str(e2,' %.9e'),num2str(E,' %.9e'));
  fprintf('PHASE13 %s p=%d rates e1={%s} e2={%s} E={%s}\n',seq,p,num2str(r1,' %.6f'),num2str(r2,' %.6f'),num2str(rE,' %.6f'));
  if strcmp(seq,'A-M1')
   rhex=log2(hex(1:end-1)./hex(2:end)); threshold=min(strict,rhex(end)-0.1);
   fprintf('PHASE13 A-M1 p=%d baseline={%s} rates={%s} final E threshold=%.6f\n',p,num2str(hex,' %.9e'),num2str(rhex,' %.6f'),threshold);
   assert(rE(end)>=threshold && r1(end)>=strict,'Phase 13 Amendment 1 gate failed for p=%d',p);
  end
 end
end
fprintf('PASS phase13_convergence\n');
end
