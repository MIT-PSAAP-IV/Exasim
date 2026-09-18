function dmd = meshpartition2(t,f,t2t,bcm,dim,elemtype,porder,nproc,metis,Cxxpreprocessing,elem2cpu)

if nargin < 11
    elem2cpu = [];
end

disp('run elementpartition...');  
dmd = elementpartition2(t,t2t,nproc,metis,elem2cpu);

if Cxxpreprocessing==0
    disp('run facepartition...');  
    dmd = facepartition2(dmd,t,f,bcm,dim,elemtype,porder,nproc);
else
    for i = 1:nproc   
        fi = f(:,dmd{i}.elempart); 
        dmd{i}.bf = 0*f(:,dmd{i}.elempart);
        for j=1:length(bcm)
          ind = fi==j;
          dmd{i}.bf(ind) = bcm(j);
        end      
    end    
end
